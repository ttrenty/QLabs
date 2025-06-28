from bit import count_trailing_zeros

from gpu import thread_idx, block_dim, block_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

alias SIZE = 2
alias BLOCKS_PER_GRID = 1
alias THREADS_PER_BLOCK = (3, 3)
alias dtype = DType.float32
alias layout = Layout.row_major(SIZE, SIZE)

alias gate_1qubit_layout = Layout.row_major(2, 2)
alias state_vector_3qubits_layout = Layout.row_major(8, 1)


fn qubit_wise_multiply_gpu(
    # Use SIMD instead
    gate_re: LayoutTensor[mut=True, dtype, gate_1qubit_layout],
    gate_im: LayoutTensor[mut=True, dtype, gate_1qubit_layout],
    gate_size: Int,
    target_qubit: Int,
    quantum_state_re: LayoutTensor[
        mut=True, dtype, state_vector_3qubits_layout
    ],
    quantum_state_im: LayoutTensor[
        mut=True, dtype, state_vector_3qubits_layout
    ],
    number_qubits: Int,
    quantum_state_size: Int,
    quantum_state_out_re: LayoutTensor[
        mut=True, dtype, state_vector_3qubits_layout
    ],
    quantum_state_out_im: LayoutTensor[
        mut=True, dtype, state_vector_3qubits_layout
    ],
    control_bits: List[List[Int]] = [],
) -> None:
    """Applies a quantum gate to specific qubits in the quantum state.

    It will apply the gate starting from the target qubit assuming that the other
    qubits that the gate acts on are following the target qubit.

    Args:
        gate_re: Real part of the gate matrix.
        gate_im: Imaginary part of the gate matrix.
        gate_size: Size of the gate (2^n, where n is the number of qubits the gate acts on).
        target_qubit: The index of the target qubit to apply the gate to.
        quantum_state_re: Real part of the quantum state vector.
        quantum_state_im: Imaginary part of the quantum state vector.
        number_qubits: Total number of qubits in the quantum state.
        quantum_state_size: Size of the quantum state vector (2^number_qubits).
        quantum_state_out_re: Output real part of the quantum state vector after applying the gate.
        quantum_state_out_im: Output imaginary part of the quantum state vector after applying the gate.
        control_bits: List of control bits, where each control bit is a list containing
                      [wire_index, flag] (1 for control, 0 for anti-control).
    """
    target_qubits_count: Int = count_trailing_zeros(gate_size)
    if (target_qubit < 0) or (target_qubit >= number_qubits):
        print(
            "Error: target_qubit index out of bounds. Must be between 0 and",
            number_qubits - 1,
        )
        print("Skipping gate application.")
        return

    inclusion_mask: Int = 0
    desired_value_mask: Int = 0
    for control in control_bits:
        wire_index, flag = control[0], control[1]
        bit: Int = 1 << wire_index  # efficient way of computing 2^wire_index
        inclusion_mask |= bit  # turn on the bit
        if flag == 1:
            desired_value_mask |= bit  # turn on the bit

    size_of_state_vector: Int = quantum_state_size
    size_of_half_block: Int = 1 << target_qubit  # 2^target_qubit
    size_of_block: Int = size_of_half_block << target_qubits_count

    # copies all amplitudes from quantum_state to quantum_state_out
    for i in range(size_of_state_vector):
        quantum_state_out_re[i] = quantum_state_re[i]
        quantum_state_out_im[i] = quantum_state_im[i]

    for block_start in range(0, size_of_state_vector, size_of_block):
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
                (gate_re[0, 0] * quantum_state_re[i1])
                - (gate_im[0, 0] * quantum_state_im[i1])
                + (gate_re[0, 1] * quantum_state_re[i2])
                - (gate_im[0, 1] * quantum_state_im[i2])
            )

            quantum_state_out_im[i1] = (
                (gate_re[0, 0] * quantum_state_im[i1])
                - (gate_im[0, 0] * quantum_state_re[i1])
                + (gate_re[0, 1] * quantum_state_im[i2])
                - (gate_im[0, 1] * quantum_state_re[i2])
            )

            quantum_state_out_re[i2] = (
                (gate_re[1, 0] * quantum_state_re[i1])
                - (gate_im[1, 0] * quantum_state_im[i1])
                + (gate_re[1, 1] * quantum_state_re[i2])
                - (gate_im[1, 1] * quantum_state_im[i2])
            )

            quantum_state_out_im[i2] = (
                (gate_re[1, 0] * quantum_state_im[i1])
                - (gate_im[1, 0] * quantum_state_re[i1])
                + (gate_re[1, 1] * quantum_state_im[i2])
                - (gate_im[1, 1] * quantum_state_re[i2])
            )
