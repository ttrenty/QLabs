from bit import count_trailing_zeros

from gpu import thread_idx, block_dim, block_idx, global_idx, barrier
from layout import Layout, LayoutTensor

alias dtype = DType.float32
alias GATE_SIZE = 2
alias NUMBER_CONTROL_BITS = 1


fn qubit_wise_multiply_inplace_gpu[
    state_vector_size: Int,
    gate_set_size: Int,
    circuit_number_control_gates: Int,
    number_control_bits: Int,  # TODO allow more flexibility like with CPU
    use_one_thread: Bool = False,
](
    gate_set_re: LayoutTensor[
        mut=False, dtype, Layout.row_major(gate_set_size, GATE_SIZE, GATE_SIZE)
    ],
    gate_set_im: LayoutTensor[
        mut=False, dtype, Layout.row_major(gate_set_size, GATE_SIZE, GATE_SIZE)
    ],
    gate_index: Int,
    gate_size: Int,
    target_qubit: Int,
    quantum_state_re: LayoutTensor[
        mut=True, dtype, Layout.row_major(state_vector_size)
    ],
    quantum_state_im: LayoutTensor[
        mut=True, dtype, Layout.row_major(state_vector_size)
    ],
    number_qubits: Int,
    quantum_state_out_re: LayoutTensor[
        mut=True, dtype, Layout.row_major(state_vector_size)
    ],
    quantum_state_out_im: LayoutTensor[
        mut=True, dtype, Layout.row_major(state_vector_size)
    ],
    control_bits_circuit: LayoutTensor[
        mut=False,
        DType.int32,
        Layout.row_major(circuit_number_control_gates, NUMBER_CONTROL_BITS, 2),
    ],
    current_control_gate_circuit: LayoutTensor[
        mut=True, DType.int32, Layout.row_major(1)
    ],
) -> None:
    """Applies a quantum gate to specific qubits in the quantum state.

    It will apply the gate starting from the target qubit assuming that the other
    qubits that the gate acts on are following the target qubit.

    Parameters:
        state_vector_size: Size of the quantum state vector (2^number_qubits).
        gate_set_size: Size of the gate set (number of unique gates).
        circuit_number_control_gates: Number of control gates in the circuit.
        number_control_bits: Number of control bits for a control gate.
        use_one_thread: If True, only the first thread will perform the operation.
                    If False, all threads will participate in the operation.

    Args:
        gate_set_re: All unique gates applied in the circuit, real part.
        gate_set_im: All unique gates applied in the circuit, imaginary part.
        gate_index: Index of the gate in the gate set to apply.
        gate_size: Size of the gate (2^n, where n is the number of qubits the gate acts on).
        target_qubit: The index of the target qubit to apply the gate to.
        quantum_state_re: Real part of the quantum state vector.
        quantum_state_im: Imaginary part of the quantum state vector.
        number_qubits: Total number of qubits in the quantum state.
        quantum_state_out_re: Output real part of the quantum state vector after applying the gate.
        quantum_state_out_im: Output imaginary part of the quantum state vector after applying the gate.
        control_bits_circuit: Control bits, where each control bit contains
                            [wire_index, flag] (1 for control, 0 for anti-control).
        current_control_gate_circuit: Current control gate circuit index,
                                used to track the position in the control_bits_circuit.
    """
    # global_i = block_dim.x * block_idx.x + thread_idx.x
    global_i = global_idx.x
    # local_i = thread_idx.x

    size_of_half_block: Int = 1 << target_qubit  # 2^target_qubit

    @parameter
    if use_one_thread:
        if global_i > 0:
            return  # Skip this thread if it is not the first one
    else:
        if global_i < state_vector_size:
            # Only threads whose index has a '0' at the target_qubit position will do the work.
            # These are the 'i1' indices.
            is_i1_thread = (global_i & size_of_half_block) == 0
            if not is_i1_thread:
                return  # Skip this thread if it is not an 'i1' thread

    if (target_qubit < 0) or (target_qubit >= number_qubits):
        print(
            "Error: target_qubit index out of bounds. Must be between 0 and",
            number_qubits - 1,
            "(Skipping gate application)",
        )
        return

    inclusion_mask: Int = 0
    desired_value_mask: Int = 0

    # CPU like implementation
    @parameter
    for control_qubit in range(number_control_bits):
        wire_index, flag = (
            control_bits_circuit[
                Int(current_control_gate_circuit[0]), control_qubit, 0
            ],
            control_bits_circuit[
                Int(current_control_gate_circuit[0]), control_qubit, 1
            ],
        )
        current_control_gate_circuit[0] += 1
        bit: Int = 1 << Int(
            wire_index
        )  # efficient way of computing 2^wire_index
        inclusion_mask |= bit  # turn on the bit
        if flag == 1:
            desired_value_mask |= bit  # turn on the bit

    # # GPU implementation
    # if global_i < number_control_bits:
    #     wire_index, flag = (
    #         control_bits_circuit[
    #             Int(current_control_gate_circuit[0]), global_i, 0
    #         ],
    #         control_bits_circuit[
    #             Int(current_control_gate_circuit[0]), global_i, 1
    #         ],
    #     )
    #     current_control_gate_circuit[0] += 1
    #     bit: Int = 1 << Int(
    #         wire_index
    #     )  # efficient way of computing 2^wire_index
    #     inclusion_mask |= bit  # turn on the bit
    #     if flag == 1:
    #         desired_value_mask |= bit  # turn on the bit

    # copies all amplitudes from quantum_state to quantum_state_out
    @parameter
    if use_one_thread:
        # CPU like implementation
        for i in range(state_vector_size):
            quantum_state_out_re[i] = quantum_state_re[i]
            quantum_state_out_im[i] = quantum_state_im[i]

        target_qubits_count: Int = count_trailing_zeros(gate_size)
        size_of_block: Int = size_of_half_block << target_qubits_count
        for block_start in range(0, state_vector_size, size_of_block):
            for offset in range(size_of_half_block):
                i1: Int = (
                    block_start | offset
                )  # faster than, but equivalent to, block_start + offset

                if (i1 & inclusion_mask) != desired_value_mask:
                    continue  # skip this iteration if the control bits do not match

                i2: Int = (
                    i1 | size_of_half_block
                )  # equivalent to i1 + size_of_half_block

                quantum_state_out_re[i1] = (
                    (gate_set_re[gate_index, 0, 0] * quantum_state_re[i1])
                    - (gate_set_im[gate_index, 0, 0] * quantum_state_im[i1])
                    + (gate_set_re[gate_index, 0, 1] * quantum_state_re[i2])
                    - (gate_set_im[gate_index, 0, 1] * quantum_state_im[i2])
                )

                quantum_state_out_im[i1] = (
                    (gate_set_re[gate_index, 0, 0] * quantum_state_im[i1])
                    + (gate_set_im[gate_index, 0, 0] * quantum_state_re[i1])
                    + (gate_set_re[gate_index, 0, 1] * quantum_state_im[i2])
                    + (gate_set_im[gate_index, 0, 1] * quantum_state_re[i2])
                )

                quantum_state_out_re[i2] = (
                    (gate_set_re[gate_index, 1, 0] * quantum_state_re[i1])
                    - (gate_set_im[gate_index, 1, 0] * quantum_state_im[i1])
                    + (gate_set_re[gate_index, 1, 1] * quantum_state_re[i2])
                    - (gate_set_im[gate_index, 1, 1] * quantum_state_im[i2])
                )

                quantum_state_out_im[i2] = (
                    (gate_set_re[gate_index, 1, 0] * quantum_state_im[i1])
                    + (gate_set_im[gate_index, 1, 0] * quantum_state_re[i1])
                    + (gate_set_re[gate_index, 1, 1] * quantum_state_im[i2])
                    + (gate_set_im[gate_index, 1, 1] * quantum_state_re[i2])
                )
    else:
        # GPU implementation
        # Parallel copy of the entire state vector
        if global_i < state_vector_size:
            quantum_state_out_re[global_i] = quantum_state_re[global_i]
            quantum_state_out_im[global_i] = quantum_state_im[global_i]

        # Synchronize all threads to ensure the copy is complete before proceeding.
        barrier()

        # Each thread works on one index `global_i`.
        # We only need to proceed if the thread is within the state vector bounds.
        if global_i < state_vector_size:
            # The core parallelization pattern:
            # Only threads whose index has a '0' at the target_qubit position will do the work.
            # We already know that these are the 'i1' indices.
            # This thread is responsible for an `i1` index.
            i1: Int = global_i

            # Check if the control bit condition is met for this pair.
            if (i1 & inclusion_mask) == desired_value_mask:
                # The condition is met, so we apply the gate.
                # First, find the partner index `i2`.
                i2: Int = i1 | size_of_half_block

                # Fetch state vector values for the pair (ψ1, ψ2)
                psi1_re = quantum_state_re[i1]
                psi1_im = quantum_state_im[i1]
                psi2_re = quantum_state_re[i2]
                psi2_im = quantum_state_im[i2]

                # Fetch gate matrix elements (g00, g01, g10, g11)
                g00_re = gate_set_re[gate_index, 0, 0]
                g00_im = gate_set_im[gate_index, 0, 0]
                g01_re = gate_set_re[gate_index, 0, 1]
                g01_im = gate_set_im[gate_index, 0, 1]
                g10_re = gate_set_re[gate_index, 1, 0]
                g10_im = gate_set_im[gate_index, 1, 0]
                g11_re = gate_set_re[gate_index, 1, 1]
                g11_im = gate_set_im[gate_index, 1, 1]

                # Perform the 2x2 matrix-vector multiplication:
                # [ out1 ] = [ g00 g01 ] [ psi1 ]
                # [ out2 ]   [ g10 g11 ] [ psi2 ]

                # Calculate out1 = g00 * psi1 + g01 * psi2
                # Real part: (g00_re*psi1_re - g00_im*psi1_im) + (g01_re*psi2_re - g01_im*psi2_im)
                quantum_state_out_re[i1] = (
                    g00_re * psi1_re - g00_im * psi1_im
                ) + (g01_re * psi2_re - g01_im * psi2_im)

                # Imaginary part: (g00_re*psi1_im + g00_im*psi1_re) + (g01_re*psi2_im + g01_im*psi2_re)
                # NOTE: This uses the standard complex multiplication rule (ad+bc).
                quantum_state_out_im[i1] = (
                    g00_re * psi1_im + g00_im * psi1_re
                ) + (g01_re * psi2_im + g01_im * psi2_re)

                # Calculate out2 = g10 * psi1 + g11 * psi2
                # Real part: (g10_re*psi1_re - g10_im*psi1_im) + (g11_re*psi2_re - g11_im*psi2_im)
                quantum_state_out_re[i2] = (
                    g10_re * psi1_re - g10_im * psi1_im
                ) + (g11_re * psi2_re - g11_im * psi2_im)

                # Imaginary part: (g10_re*psi1_im + g10_im*psi1_re) + (g11_re*psi2_im + g11_im*psi2_re)
                quantum_state_out_im[i2] = (
                    g10_re * psi1_im + g10_im * psi1_re
                ) + (g11_re * psi2_im + g11_im * psi2_re)
            # If control bits do not match, we do nothing. The values already
            # copied to quantum_state_out are correct.

    # Using more threads per elements, one for real, one for imaginary part
    # target_qubits_count: Int = count_trailing_zeros(gate_size)
    # size_of_half_block: Int = 1 << target_qubit

    # # Total number of pairs to calculate
    # num_pairs = quantum_state_size // 2

    # if global_i < quantum_state_size:
    #     # 1. Identify group and role
    #     group_id = global_i // 4
    #     local_id_in_group = global_i % 4

    #     # Only proceed if we are part of a valid group for a pair
    #     if group_id < num_pairs:
    #         # 2. Map group_id to state vector indices i1, i2
    #         block_id = group_id // size_of_half_block
    #         offset_in_block = group_id % size_of_half_block
    #         i1: Int = (block_id * size_of_block) + offset_in_block

    #         if (i1 & inclusion_mask) == desired_value_mask:
    #             i2: Int = i1 | size_of_half_block

    #             # Fetch state vector values for the pair (ψ1, ψ2)
    #             psi1_re = quantum_state_re[i1]
    #             psi1_im = quantum_state_im[i1]
    #             psi2_re = quantum_state_re[i2]
    #             psi2_im = quantum_state_im[i2]

    #             # 4. Divide the calculation
    #             if local_id_in_group == 0:  # quantum_state_out_re[i1]
    #                 g00_re = gate_set_re[gate_index, 0, 0]
    #                 g00_im = gate_set_im[gate_index, 0, 0]
    #                 g01_re = gate_set_re[gate_index, 0, 1]
    #                 g01_im = gate_set_im[gate_index, 0, 1]
    #                 quantum_state_out_re[i1] = (
    #                     g00_re * psi1_re - g00_im * psi1_im
    #                 ) + (g01_re * psi2_re - g01_im * psi2_im)
    #             elif local_id_in_group == 1:  # quantum_state_out_im[i1]
    #                 g00_re = gate_set_re[gate_index, 0, 0]
    #                 g00_im = gate_set_im[gate_index, 0, 0]
    #                 g01_re = gate_set_re[gate_index, 0, 1]
    #                 g01_im = gate_set_im[gate_index, 0, 1]
    #                 quantum_state_out_im[i1] = (
    #                     g00_re * psi1_im + g00_im * psi1_re
    #                 ) + (g01_re * psi2_im + g01_im * psi2_re)
    #             elif local_id_in_group == 2:  # quantum_state_out_re[i2]
    #                 g10_re = gate_set_re[gate_index, 1, 0]
    #                 g10_im = gate_set_im[gate_index, 1, 0]
    #                 g11_re = gate_set_re[gate_index, 1, 1]
    #                 g11_im = gate_set_im[gate_index, 1, 1]
    #                 quantum_state_out_re[i2] = (
    #                     g10_re * psi1_re - g10_im * psi1_im
    #                 ) + (g11_re * psi2_re - g11_im * psi2_im)
    #             elif local_id_in_group == 3:  # quantum_state_out_im[i2]
    #                 g10_re = gate_set_re[gate_index, 1, 0]
    #                 g10_im = gate_set_im[gate_index, 1, 0]
    #                 g11_re = gate_set_re[gate_index, 1, 1]
    #                 g11_im = gate_set_im[gate_index, 1, 1]
    #                 quantum_state_out_im[i2] = (
    #                     g10_re * psi1_im + g10_im * psi1_re
    #                 ) + (g11_re * psi2_im + g11_im * psi2_re)


# # TODO one day, but maybe it will become memory bound if we do that since we have to create
# # intermediary values for complex multiplications
# fn qubit_wise_multiply_gpu_3[
#     number_control_bits: Int
# ](
#     gate_set: LayoutTensor[mut=False, dtype, gate_set_1qubit_vectorized_layout],
#     gate_index: Int,
#     gate_size: Int,
#     target_qubit: Int,
#     quantum_state_re: LayoutTensor[
#         mut=True, dtype, state_vector_3qubits_layout
#     ],
#     quantum_state_im: LayoutTensor[
#         mut=True, dtype, state_vector_3qubits_layout
#     ],
#     number_qubits: Int,
#     quantum_state_size: Int,
#     quantum_state_out_re: LayoutTensor[
#         mut=True, dtype, state_vector_3qubits_layout
#     ],
#     quantum_state_out_im: LayoutTensor[
#         mut=True, dtype, state_vector_3qubits_layout
#     ],
#     # control_bits: LayoutTensor[mut=True, DType.int32, control_bits_layout],
#     control_bits_circuit: LayoutTensor[
#         mut=False, DType.int32, circuit_control_bits_layout
#     ],
#     current_control_gate_circuit: LayoutTensor[
#         mut=True, DType.int32, Layout.row_major(1)
#     ],
# ) -> None:
#     """Applies a quantum gate to specific qubits in the quantum state.

#     It will apply the gate starting from the target qubit assuming that the other
#     qubits that the gate acts on are following the target qubit.

#     Parameters:
#         number_control_bits: Number of control bits.

#     Args:
#         gate_set: All unique gates applied in the circuit, real and imaginary parts
#                   to be treated as a SIMD vector.
#         gate_index: Index of the gate in the gate set to apply.
#         gate_size: Size of the gate (2^n, where n is the number of qubits the gate acts on).
#         target_qubit: The index of the target qubit to apply the gate to.
#         quantum_state_re: Real part of the quantum state vector.
#         quantum_state_im: Imaginary part of the quantum state vector.
#         number_qubits: Total number of qubits in the quantum state.
#         quantum_state_size: Size of the quantum state vector (2^number_qubits).
#         quantum_state_out_re: Output real part of the quantum state vector after applying the gate.
#         quantum_state_out_im: Output imaginary part of the quantum state vector after applying the gate.
#         control_bits_circuit: Control bits, where each control bit contains
#                             [wire_index, flag] (1 for control, 0 for anti-control).
#         current_control_gate_circuit: Current control gate circuit index,
#                                 used to track the position in the control_bits_circuit.
#     """
#     print("Inside qubit_wise_multiply_gpu")
#     target_qubits_count: Int = count_trailing_zeros(gate_size)
#     if (target_qubit < 0) or (target_qubit >= number_qubits):
#         print(
#             "Error: target_qubit index out of bounds. Must be between 0 and",
#             number_qubits - 1,
#         )
#         print("Skipping gate application.")
#         return

#     print("AAAAA")
#     inclusion_mask: Int = 0
#     desired_value_mask: Int = 0

#     @parameter
#     for i in range(number_control_bits):
#         print("before")
#         wire_index, flag = (
#             control_bits_circuit[Int(current_control_gate_circuit[0]), i, 0],
#             control_bits_circuit[Int(current_control_gate_circuit[0]), i, 1],
#         )
#         current_control_gate_circuit[0] += 1
#         print("after")
#         bit: Int = 1 << Int(
#             wire_index
#         )  # efficient way of computing 2^wire_index
#         inclusion_mask |= bit  # turn on the bit
#         if flag == 1:
#             desired_value_mask |= bit  # turn on the bit

#     print("BBBBB")
#     size_of_state_vector: Int = quantum_state_size
#     size_of_half_block: Int = 1 << target_qubit  # 2^target_qubit
#     size_of_block: Int = size_of_half_block << target_qubits_count

#     print("CCCC")
#     # copies all amplitudes from quantum_state to quantum_state_out
#     for i in range(size_of_state_vector):
#         quantum_state_out_re[i] = quantum_state_re[i]
#         quantum_state_out_im[i] = quantum_state_im[i]

#     print("before loop")
#     for block_start in range(0, size_of_state_vector, size_of_block):
#         # print("block_start:", block_start)
#         for offset in range(size_of_half_block):
#             # print("offset:", offset)
#             i1: Int = (
#                 block_start | offset
#             )  # faster than, but equivalent to, block_start + offset

#             if (i1 & inclusion_mask) != desired_value_mask:
#                 continue  # skip this iteration if the control bits do not match

#             i2: Int = (
#                 i1 | size_of_half_block
#             )  # equivalent to i1 + size_of_half_block

#             print("i1:", i1, "i2:", i2)

#             # new_state_vector[i1] = (
#             #     gate[0, 0] * quantum_state[i1] + gate[0, 1] * quantum_state[i2]
#             # )
#             # new_state_vector[i2] = (
#             #     gate[1, 0] * quantum_state[i1] + gate[1, 1] * quantum_state[i2]
#             # )

#             right_part = gate_set[gate_index, 0, 0] * quantum_state_im[i1]
#             right_part_re = right_part
#             right_part_re[1] = 0
#             right_part_im = right_part
#             right_part_im[0] = 0

#             quantum_state_out_re[i1] = (
#                 (gate_set[gate_index, 0, 0] * quantum_state_re[i1])
#                 - right_part_re
#                 + right_part_im
#             )
