from bit import count_trailing_zeros
from sys import has_accelerator

from gpu import thread_idx, block_dim, block_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, IntTuple, print_layout

from qlabs.base import (
    StateVector,
    Gate,
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
    NOT,
    H,
    X,
    Y,
    Z,
    SWAP,
    iSWAP,
)

from qlabs.base.gpu import qubit_wise_multiply_gpu, qubit_wise_multiply_gpu_2

alias BLOCKS_PER_GRID = 1
alias THREADS_PER_BLOCK = (1, 1)
alias dtype = DType.float32

alias GATE_SIZE = 2
alias STATE_VECTOR_SIZE = 8
alias NUMBER_CONTROL_BITS = 1
# crashes the code on GitHub (tag Austin Doolittle)
# another crash due to IntTuple
# alias control_bits_list: List[List[List[Int]]] = [
#     [[1, 1]],  # Control on qubit 1 and is control because flag=1
#     [[1, 1]],  # Control on qubit 1 and is control because flag=1
# ]
alias CIRCUIT_NUMBER_CONTROL_GATES = 2


alias gate_1qubit_layout = Layout.row_major(GATE_SIZE, GATE_SIZE)
alias state_vector_3qubits_layout = Layout.row_major(STATE_VECTOR_SIZE, 1)
alias control_bits_layout = Layout.row_major(NUMBER_CONTROL_BITS, 2)

alias gate_set = [Hadamard, PauliX, PauliZ]
alias gate_set_dic: Dict[String, Int] = {
    Hadamard.symbol: 0,
    PauliX.symbol: 1,
    PauliZ.symbol: 2,
}
alias GATE_SET_SIZE = 3
alias gate_set_1qubit_layout = Layout.row_major(
    GATE_SET_SIZE, GATE_SIZE, GATE_SIZE
)


def gpu_debug_something():
    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()
        print("Found GPU:", ctx.name())

        gate_re = ctx.enqueue_create_buffer[dtype](
            GATE_SIZE * GATE_SIZE
        ).enqueue_fill(0)
        gate_im = ctx.enqueue_create_buffer[dtype](
            GATE_SIZE * GATE_SIZE
        ).enqueue_fill(0)
        quantum_state_re = ctx.enqueue_create_buffer[dtype](
            STATE_VECTOR_SIZE
        ).enqueue_fill(0)
        quantum_state_im = ctx.enqueue_create_buffer[dtype](
            STATE_VECTOR_SIZE
        ).enqueue_fill(0)
        quantum_state_out_re = ctx.enqueue_create_buffer[dtype](
            STATE_VECTOR_SIZE
        ).enqueue_fill(0)
        quantum_state_out_im = ctx.enqueue_create_buffer[dtype](
            STATE_VECTOR_SIZE
        ).enqueue_fill(0)

        control_bits = ctx.enqueue_create_buffer[DType.int32](
            NUMBER_CONTROL_BITS * 2
        ).enqueue_fill(0)

        gate_re_tensor = LayoutTensor[mut=True, dtype, gate_1qubit_layout](
            gate_re.unsafe_ptr()
        )
        gate_im_tensor = LayoutTensor[mut=True, dtype, gate_1qubit_layout](
            gate_im.unsafe_ptr()
        )
        quantum_state_re_tensor = LayoutTensor[
            mut=True, dtype, state_vector_3qubits_layout
        ](quantum_state_re.unsafe_ptr())
        quantum_state_im_tensor = LayoutTensor[
            mut=True, dtype, state_vector_3qubits_layout
        ](quantum_state_im.unsafe_ptr())
        quantum_state_out_re_tensor = LayoutTensor[
            mut=True, dtype, state_vector_3qubits_layout
        ](quantum_state_out_re.unsafe_ptr())
        quantum_state_out_im_tensor = LayoutTensor[
            mut=True, dtype, state_vector_3qubits_layout
        ](quantum_state_out_im.unsafe_ptr())
        control_bits_tensor = LayoutTensor[
            mut=True, DType.int32, control_bits_layout
        ](control_bits.unsafe_ptr())

        print("Before")

        ctx.enqueue_function[qubit_wise_multiply_gpu](
            gate_re_tensor,
            gate_im_tensor,
            GATE_SIZE,
            0,  # target_qubit
            quantum_state_re_tensor,
            quantum_state_im_tensor,
            3,  # number_qubits
            STATE_VECTOR_SIZE,  # quantum_state_size
            quantum_state_out_re_tensor,
            quantum_state_out_im_tensor,
            control_bits_tensor,
            NUMBER_CONTROL_BITS,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        print("After")

        ctx.synchronize()

        with quantum_state_out_re.map_to_host() as host_re:
            print("Output real part:", host_re)
        with quantum_state_out_im.map_to_host() as host_im:
            print("Output imaginary part:", host_im)
        with gate_re.map_to_host() as host_gate_re:
            print("Gate real part:", host_gate_re)
        with gate_im.map_to_host() as host_gate_im:
            print("Gate imaginary part:", host_gate_im)
        with quantum_state_re.map_to_host() as host_quantum_state_re:
            print("Quantum state real part:", host_quantum_state_re)
        with quantum_state_im.map_to_host() as host_quantum_state_im:
            print("Quantum state imaginary part:", host_quantum_state_im)


def run_gpu_not_abstract():
    """Simulates the circuit from Figure 1 in the paper."""

    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        print("Simulating Figure 1 circuit.\nCircuit design:")
        print(
            """
    |0> -------|X|--|Z|--
                |
    |0> --|H|---*----*---
                    |
    |0> --|X|-------|X|--
        """
        )
        var control_bits_list: List[List[List[Int]]] = [
            [[1, 1]],  # Control on qubit 1 and is control because flag=1
            [[1, 1]],  # Control on qubit 1 and is control because flag=1
        ]

        ctx = DeviceContext()
        print("Using GPU:", ctx.name())

        # -- Create GPU variables -- #
        ctx = DeviceContext()

        # Initialize the quantum circuit to the |000⟩ state
        quantum_state: StateVector = StateVector.from_bitstring("000")
        print("Initial quantum state:\n", quantum_state)

        quantum_state_re = ctx.enqueue_create_buffer[dtype](
            STATE_VECTOR_SIZE
        ).enqueue_fill(0)
        quantum_state_im = ctx.enqueue_create_buffer[dtype](
            STATE_VECTOR_SIZE
        ).enqueue_fill(0)

        with quantum_state_re.map_to_host() as device_re, quantum_state_im.map_to_host() as device_im:
            for i in range(STATE_VECTOR_SIZE):
                device_re[i] = quantum_state[i].re
                device_im[i] = quantum_state[i].im

        with quantum_state_re.map_to_host() as host_re:
            print("Quantum state real part:", host_re)
        with quantum_state_im.map_to_host() as host_im:
            print("Quantum state imaginary part:", host_im)

        gate_set_re = ctx.enqueue_create_buffer[dtype](
            GATE_SET_SIZE * GATE_SIZE * GATE_SIZE
        ).enqueue_fill(0)
        gate_set_im = ctx.enqueue_create_buffer[dtype](
            GATE_SET_SIZE * GATE_SIZE * GATE_SIZE
        ).enqueue_fill(0)

        with gate_set_re.map_to_host() as device_gate_re, gate_set_im.map_to_host() as device_gate_im:
            for i in range(GATE_SET_SIZE):
                gate = gate_set[i]
                for j in range(GATE_SIZE):
                    for k in range(GATE_SIZE):
                        index = gate_set_1qubit_layout(
                            IntTuple(i, j, k)
                        )  # Get the index in the 1D buffer
                        device_gate_re[index] = gate[j, k].re
                        device_gate_im[index] = gate[j, k].im

        quantum_state_out_re = ctx.enqueue_create_buffer[dtype](
            STATE_VECTOR_SIZE
        ).enqueue_fill(0)
        quantum_state_out_im = ctx.enqueue_create_buffer[dtype](
            STATE_VECTOR_SIZE
        ).enqueue_fill(0)

        # TODO have one big variable for all control bits
        # and have NUMBER_CONTROL_BITS be a list defining each gates specific control bits count
        # maybe need a variable to keep track of the current control gate index
        control_bits_0 = ctx.enqueue_create_buffer[DType.int32](
            NUMBER_CONTROL_BITS * 2
        ).enqueue_fill(0)

        control_bits_1 = ctx.enqueue_create_buffer[DType.int32](
            NUMBER_CONTROL_BITS * 2
        ).enqueue_fill(0)

        with control_bits_0.map_to_host() as device_control_0, control_bits_1.map_to_host() as device_control_1:
            # Set control bits for the first controlled gate
            device_control_0[0] = 1
            device_control_0[1] = 1

            # Set control bits for the second controlled gate
            device_control_1[0] = 1
            device_control_1[1] = 1

        control_bits_empty = ctx.enqueue_create_buffer[DType.int32](
            NUMBER_CONTROL_BITS * 2
        ).enqueue_fill(0)

        # Create control bits for the circuit
        control_bits_circuit = ctx.enqueue_create_buffer[DType.int32](
            CIRCUIT_NUMBER_CONTROL_GATES * NUMBER_CONTROL_BITS * 2
        ).enqueue_fill(0)
        current_control_gate_circuit = ctx.enqueue_create_buffer[DType.int32](
            1
        ).enqueue_fill(0)

        print("FIRST LOOP")
        for i in range(len(control_bits_list)):
            for j in range(len(control_bits_list[i])):
                for k in range(len(control_bits_list[i][j])):
                    print("i:", i, "j:", j, "k:", k)
                    print(
                        "control_bits_list[i][j][k]:",
                        control_bits_list[i][j][k],
                    )

        print("before printing index")

        var coords_2 = IntTuple(0, 0, 0)
        print("coords_2:", coords_2)

        alias circuit_control_bits_layout = Layout.row_major(
            CIRCUIT_NUMBER_CONTROL_GATES, NUMBER_CONTROL_BITS, 2
        )
        var index = circuit_control_bits_layout(coords_2)
        print(
            "gate_set_1qubit_layout(coords_2):",
            index,
        )

        # print(
        #     "circuit_control_bits_layout(coords_2):",
        #     circuit_control_bits_layout(IntTuple(0, 0, 0)),
        # )

        # _ = gate_set_1qubit_layout
        # _ = circuit_control_bits_layout

        print("\nSECOND LOOP")
        # with control_bits_circuit.map_to_host() as device_control_bits_circuit:
        #     for i in range(CIRCUIT_NUMBER_CONTROL_GATES):
        #         for j in range(NUMBER_CONTROL_BITS):
        #             for k in range(2):
        #                 # print("i:", i, "j:", j, "k:", k, "index:", index)
        #                 print("i:", i, "j:", j, "k:", k)
        #                 print(
        #                     "control_bits_list[i][j][k]:",
        #                     control_bits_list[i][j][k],
        #                 )
        #                 index = circuit_control_bits_layout(IntTuple(i, j, k))
        #                 print("index:", index)
        #                 # device_control_bits_circuit[index] = control_bits_list[
        #                 #     i
        #                 # ][j][k]

        #     print(
        #         "Control bits for the circuit:\n",
        #         device_control_bits_circuit,
        #     )

        # -- Create layout tensors for GPU operations -- #
        gate_set_re_tensor = LayoutTensor[
            mut=True, dtype, gate_set_1qubit_layout
        ](gate_set_re.unsafe_ptr())
        gate_set_im_tensor = LayoutTensor[
            mut=True, dtype, gate_set_1qubit_layout
        ](gate_set_im.unsafe_ptr())

        quantum_state_re_tensor = LayoutTensor[
            mut=True, dtype, state_vector_3qubits_layout
        ](quantum_state_re.unsafe_ptr())
        quantum_state_im_tensor = LayoutTensor[
            mut=True, dtype, state_vector_3qubits_layout
        ](quantum_state_im.unsafe_ptr())

        quantum_state_out_re_tensor = LayoutTensor[
            mut=True, dtype, state_vector_3qubits_layout
        ](quantum_state_out_re.unsafe_ptr())
        quantum_state_out_im_tensor = LayoutTensor[
            mut=True, dtype, state_vector_3qubits_layout
        ](quantum_state_out_im.unsafe_ptr())

        control_bits_empty_tensor = LayoutTensor[
            mut=True, DType.int32, control_bits_layout
        ](control_bits_empty.unsafe_ptr())
        control_bits_0_tensor = LayoutTensor[
            mut=True, DType.int32, control_bits_layout
        ](control_bits_0.unsafe_ptr())
        control_bits_1_tensor = LayoutTensor[
            mut=True, DType.int32, control_bits_layout
        ](control_bits_1.unsafe_ptr())

        control_bits_circuit_tensor = LayoutTensor[
            mut=True, DType.int32, circuit_control_bits_layout
        ](control_bits_circuit.unsafe_ptr())
        current_control_gate_circuit_tensor = LayoutTensor[
            mut=True, DType.int32, Layout.row_major(1)
        ](current_control_gate_circuit.unsafe_ptr())

        # Enqueue create_initial_state

        # Enqueue applying gates

        # Gate 0
        # quantum_state = qubit_wise_multiply_gpu(
        #     Hadamard.matrix, 1, quantum_state
        # )
        ctx.enqueue_function[qubit_wise_multiply_gpu_2[number_control_bits=0]](
            gate_set_re_tensor,
            gate_set_im_tensor,
            gate_set_dic[Hadamard.symbol],
            GATE_SIZE,
            1,  # target_qubit
            quantum_state_re_tensor,
            quantum_state_im_tensor,
            3,  # number_qubits
            STATE_VECTOR_SIZE,  # quantum_state_size
            quantum_state_out_re_tensor,
            quantum_state_out_im_tensor,
            control_bits_empty,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        # # It works
        # with quantum_state_out_re.map_to_host() as host_re, quantum_state_out_im.map_to_host() as host_im:
        #     print(
        #         "After Hadamard gate on qubit 1\nreal part:\n",
        #         host_re,
        #         "\nimaginary part:\n",
        #         host_im,
        #     )

        # Gate 1 (reverse the states input <-> output)
        # quantum_state = qubit_wise_multiply(PauliX.matrix, 2, quantum_state)
        ctx.enqueue_function[qubit_wise_multiply_gpu_2[number_control_bits=0]](
            gate_set_re_tensor,
            gate_set_im_tensor,
            gate_set_dic[PauliX.symbol],
            GATE_SIZE,
            2,  # target_qubit
            quantum_state_out_re_tensor,
            quantum_state_out_im_tensor,
            3,  # number_qubits
            STATE_VECTOR_SIZE,  # quantum_state_size
            quantum_state_re_tensor,
            quantum_state_im_tensor,
            control_bits_empty,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        # with quantum_state_re.map_to_host() as host_re, quantum_state_im.map_to_host() as host_im:
        #     print(
        #         "After Pauli-X gate on qubit 2:",
        #         "\nreal part:\n",
        #         host_re,
        #         "\nimaginary part:\n",
        #         host_im,
        #     )

        # # Gate 2
        # quantum_state = qubit_wise_multiply(
        #     PauliX.matrix, 0, quantum_state, [[1, 1]]
        # )
        ctx.enqueue_function[qubit_wise_multiply_gpu_2[number_control_bits=1]](
            gate_set_re_tensor,
            gate_set_im_tensor,
            gate_set_dic[PauliX.symbol],
            GATE_SIZE,
            0,  # target_qubit
            quantum_state_re_tensor,
            quantum_state_im_tensor,
            3,  # number_qubits
            STATE_VECTOR_SIZE,  # quantum_state_size
            quantum_state_out_re_tensor,
            quantum_state_out_im_tensor,
            control_bits_0_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        # with quantum_state_out_re.map_to_host() as host_re, quantum_state_out_im.map_to_host() as host_im:
        #     print(
        #         "After Pauli-X gate on qubit 0 with control on qubit 1:",
        #         "\nreal part:\n",
        #         host_re,
        #         "\nimaginary part:\n",
        #         host_im,
        #     )

        # Gate 3
        # quantum_state = qubit_wise_multiply(PauliZ.matrix, 0, quantum_state)
        ctx.enqueue_function[qubit_wise_multiply_gpu_2[number_control_bits=0]](
            gate_set_re_tensor,
            gate_set_im_tensor,
            gate_set_dic[PauliZ.symbol],
            GATE_SIZE,
            0,  # target_qubit
            quantum_state_out_re_tensor,
            quantum_state_out_im_tensor,
            3,  # number_qubits
            STATE_VECTOR_SIZE,  # quantum_state_size
            quantum_state_re_tensor,
            quantum_state_im_tensor,
            control_bits_empty_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        # with quantum_state_re.map_to_host() as host_re, quantum_state_im.map_to_host() as host_im:
        #     print(
        #         "After Pauli-Z gate on qubit 0:\nreal part:\n",
        #         host_re,
        #         "\nimaginary part:\n",
        #         host_im,
        #     )

        # Gate 4
        # quantum_state = qubit_wise_multiply(
        #     PauliX.matrix, 2, quantum_state, [[1, 1]]
        # )
        ctx.enqueue_function[qubit_wise_multiply_gpu_2[number_control_bits=1]](
            gate_set_re_tensor,
            gate_set_im_tensor,
            gate_set_dic[PauliX.symbol],
            GATE_SIZE,
            2,  # target_qubit
            quantum_state_re_tensor,
            quantum_state_im_tensor,
            3,  # number_qubits
            STATE_VECTOR_SIZE,  # quantum_state_size
            quantum_state_out_re_tensor,
            quantum_state_out_im_tensor,
            control_bits_1_tensor,
            grid_dim=BLOCKS_PER_GRID,
            block_dim=THREADS_PER_BLOCK,
        )

        with quantum_state_out_re.map_to_host() as host_re, quantum_state_out_im.map_to_host() as host_im:
            print(
                (
                    "After Pauli-X gate on qubit 2 with control on qubit 1"
                    " (Final State):\nreal part:\n"
                ),
                host_re,
                "\nimaginary part:\n",
                host_im,
            )
