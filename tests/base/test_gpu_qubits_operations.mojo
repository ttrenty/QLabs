from testing import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
    assert_almost_equal,
)
from testing_matrix import assert_matrix_almost_equal
from testing_state_vector import (
    assert_state_vector_almost_equal,
    test_qubit_wise_multiply_0_reference,
    test_qubit_wise_multiply_figure1_reference,
)

from sys import has_accelerator
from gpu import thread_idx, block_dim, block_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, IntTuple, print_layout

from bit import count_trailing_zeros

from qlabs.base import (
    StateVector,
    Gate,
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
)

from qlabs.base.gpu import qubit_wise_multiply_inplace_gpu

from qlabs.local_stdlib.complex import ComplexFloat32

alias dtype = DType.float32

alias GATE_SIZE = 2
alias NUMBER_CONTROL_BITS = 1


def test_qubit_wise_multiply_0():
    """Simulate a small circuit"""

    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
        return

    gate_set: List[Gate] = [Hadamard, PauliX]
    alias gate_set_dic: Dict[String, Int] = {
        Hadamard.symbol: 0,
        PauliX.symbol: 1,
    }
    alias gate_set_size = 2
    alias gate_set_1qubit_layout = Layout.row_major(
        gate_set_size, GATE_SIZE, GATE_SIZE
    )

    alias CIRCUIT_NUMBER_CONTROL_GATES = 1
    alias circuit_control_bits_layout = Layout.row_major(
        CIRCUIT_NUMBER_CONTROL_GATES, NUMBER_CONTROL_BITS, 2
    )

    alias num_qubits = 3
    alias state_vector_size = 1 << num_qubits
    alias state_vector_layout = Layout.row_major(state_vector_size)

    alias total_threads = state_vector_size

    alias blocks_per_grid = 1
    alias threads_per_block = (
        total_threads,
        1,
        1,
    )

    ctx = DeviceContext()

    host_quantum_state_re = ctx.enqueue_create_host_buffer[dtype](
        state_vector_size
    )
    host_quantum_state_im = ctx.enqueue_create_host_buffer[dtype](
        state_vector_size
    )

    host_gate_set_re = ctx.enqueue_create_host_buffer[dtype](
        gate_set_size * GATE_SIZE * GATE_SIZE
    )
    host_gate_set_im = ctx.enqueue_create_host_buffer[dtype](
        gate_set_size * GATE_SIZE * GATE_SIZE
    )

    host_control_bits_circuit = ctx.enqueue_create_host_buffer[DType.int32](
        CIRCUIT_NUMBER_CONTROL_GATES * NUMBER_CONTROL_BITS * 2
    )

    quantum_state: StateVector = StateVector.from_bitstring("000")
    var control_bits_list: List[List[List[Int]]] = [
        [[0, 1]],
    ]

    ctx.synchronize()

    # -- Fill host buffers -- #

    for i in range(state_vector_size):
        host_quantum_state_re[i] = quantum_state[i].re
        host_quantum_state_im[i] = quantum_state[i].im

    for i in range(gate_set_size):
        gate = gate_set[i]
        for j in range(GATE_SIZE):
            for k in range(GATE_SIZE):
                index = gate_set_1qubit_layout(
                    IntTuple(i, j, k)
                )  # Get the index in the 1D buffer
                host_gate_set_re[index] = gate[j, k].re
                host_gate_set_im[index] = gate[j, k].im

    for i in range(CIRCUIT_NUMBER_CONTROL_GATES):
        for j in range(NUMBER_CONTROL_BITS):
            for k in range(2):
                index = circuit_control_bits_layout(IntTuple(i, j, k))
                host_control_bits_circuit[index] = control_bits_list[i][j][k]

    # -- Copy host buffers to device buffers -- #
    quantum_state_re = ctx.enqueue_create_buffer[dtype](state_vector_size)
    quantum_state_im = ctx.enqueue_create_buffer[dtype](state_vector_size)

    gate_set_re = ctx.enqueue_create_buffer[dtype](
        gate_set_size * GATE_SIZE * GATE_SIZE
    )
    gate_set_im = ctx.enqueue_create_buffer[dtype](
        gate_set_size * GATE_SIZE * GATE_SIZE
    )

    control_bits_circuit = ctx.enqueue_create_buffer[DType.int32](
        CIRCUIT_NUMBER_CONTROL_GATES * NUMBER_CONTROL_BITS * 2
    )
    current_control_gate_circuit = ctx.enqueue_create_buffer[DType.int32](1)

    # Create other buffers for functions

    quantum_state_out_re = ctx.enqueue_create_buffer[dtype](state_vector_size)
    quantum_state_out_im = ctx.enqueue_create_buffer[dtype](state_vector_size)

    quantum_state_re.enqueue_copy_from(host_quantum_state_re)
    quantum_state_im.enqueue_copy_from(host_quantum_state_im)

    gate_set_re.enqueue_copy_from(host_gate_set_re)
    gate_set_im.enqueue_copy_from(host_gate_set_im)

    control_bits_circuit.enqueue_copy_from(host_control_bits_circuit)

    ctx.enqueue_memset(current_control_gate_circuit, 0)
    ctx.enqueue_memset(quantum_state_out_re, 0.0)
    ctx.enqueue_memset(quantum_state_out_im, 0.0)

    # -- Create layout tensors for GPU operations -- #
    gate_set_re_tensor = LayoutTensor[mut=False, dtype, gate_set_1qubit_layout](
        gate_set_re.unsafe_ptr()
    )
    gate_set_im_tensor = LayoutTensor[mut=False, dtype, gate_set_1qubit_layout](
        gate_set_im.unsafe_ptr()
    )

    quantum_state_re_tensor = LayoutTensor[
        mut=True, dtype, state_vector_layout
    ](quantum_state_re.unsafe_ptr())
    quantum_state_im_tensor = LayoutTensor[
        mut=True, dtype, state_vector_layout
    ](quantum_state_im.unsafe_ptr())

    quantum_state_out_re_tensor = LayoutTensor[
        mut=True, dtype, state_vector_layout
    ](quantum_state_out_re.unsafe_ptr())
    quantum_state_out_im_tensor = LayoutTensor[
        mut=True, dtype, state_vector_layout
    ](quantum_state_out_im.unsafe_ptr())

    control_bits_circuit_tensor = LayoutTensor[
        mut=False, DType.int32, circuit_control_bits_layout
    ](control_bits_circuit.unsafe_ptr())
    current_control_gate_circuit_tensor = LayoutTensor[
        mut=True, DType.int32, Layout.row_major(1)
    ](current_control_gate_circuit.unsafe_ptr())

    # quantum_state = qubit_wise_multiply(Hadamard.matrix, 0, quantum_state)
    ctx.enqueue_function[
        qubit_wise_multiply_inplace_gpu[number_control_bits=0]
    ](
        gate_set_re_tensor,
        gate_set_im_tensor,
        gate_set_dic[Hadamard.symbol],
        GATE_SIZE,
        0,  # target_qubit
        quantum_state_re_tensor,
        quantum_state_im_tensor,
        num_qubits,  # number_qubits
        state_vector_size,  # quantum_state_size
        quantum_state_out_re_tensor,
        quantum_state_out_im_tensor,
        control_bits_circuit_tensor,
        current_control_gate_circuit_tensor,
        grid_dim=blocks_per_grid,
        block_dim=threads_per_block,
    )

    # quantum_state = qubit_wise_multiply(
    #     PauliX.matrix, 1, quantum_state, [[0, 1]]
    # )
    ctx.enqueue_function[
        qubit_wise_multiply_inplace_gpu[number_control_bits=1]
    ](
        gate_set_re_tensor,
        gate_set_im_tensor,
        gate_set_dic[PauliX.symbol],
        GATE_SIZE,
        1,  # target_qubit
        quantum_state_out_re_tensor,
        quantum_state_out_im_tensor,
        num_qubits,  # number_qubits
        state_vector_size,  # quantum_state_size
        quantum_state_re_tensor,
        quantum_state_im_tensor,
        control_bits_circuit_tensor,
        current_control_gate_circuit_tensor,
        grid_dim=blocks_per_grid,
        block_dim=threads_per_block,
    )

    # quantum_state = qubit_wise_multiply(Hadamard.matrix, 2, quantum_state)
    ctx.enqueue_function[
        qubit_wise_multiply_inplace_gpu[number_control_bits=0]
    ](
        gate_set_re_tensor,
        gate_set_im_tensor,
        gate_set_dic[Hadamard.symbol],
        GATE_SIZE,
        2,  # target_qubit
        quantum_state_re_tensor,
        quantum_state_im_tensor,
        num_qubits,  # number_qubits
        state_vector_size,  # quantum_state_size
        quantum_state_out_re_tensor,
        quantum_state_out_im_tensor,
        control_bits_circuit_tensor,
        current_control_gate_circuit_tensor,
        grid_dim=blocks_per_grid,
        block_dim=threads_per_block,
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
        for i in range(state_vector_size):
            quantum_state[i] = ComplexFloat32(host_re[i], host_im[i])

    assert_state_vector_almost_equal(
        quantum_state, test_qubit_wise_multiply_0_reference
    )


def test_qubit_wise_multiply_figure1():
    """Simulates a circuit of arbitrary number of qubits"""

    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
        return

    alias num_qubits = 3

    alias circuit_number_control_gates = 2
    alias circuit_control_bits_layout = Layout.row_major(
        circuit_number_control_gates, NUMBER_CONTROL_BITS, 2
    )

    gate_set: List[Gate] = [Hadamard, PauliX, PauliZ]
    gate_set_dic: Dict[String, Int] = {
        Hadamard.symbol: 0,
        PauliX.symbol: 1,
        PauliZ.symbol: 2,
    }
    alias gate_set_size = 3
    alias gate_set_1qubit_layout = Layout.row_major(
        gate_set_size, GATE_SIZE, GATE_SIZE
    )

    alias state_vector_size = 1 << num_qubits
    alias state_vector_layout = Layout.row_major(state_vector_size)

    alias total_threads = state_vector_size

    alias max_threads_per_block = 1024  # Maximum threads per block in CUDA
    alias blocks_per_grid = (
        total_threads + max_threads_per_block - 1
    ) // max_threads_per_block

    alias threads_per_block = (
        max_threads_per_block,
        1,
        1,
    )

    @parameter
    if total_threads < max_threads_per_block:
        alias threads_per_block = (
            total_threads,
            1,
            1,
        )  # 1D block of threads

    print("state_vector_size:", state_vector_size)
    print("blocks_per_grid:", blocks_per_grid)
    print("threads_per_block[0]:", threads_per_block[0])

    var control_bits_list: List[List[List[Int]]] = [
        [[1, 1]],  # Control on qubit 1 and is control because flag=1
        [[1, 1]],  # Control on qubit 1 and is control because flag=1
    ]

    ctx = DeviceContext()
    print("Using GPU:", ctx.name())
    print("ctx.device_info:", ctx.device_info)
    print(
        "ctx.device_info.max_thread_block_size:",
        ctx.device_info.max_thread_block_size,
    )
    print(
        "ctx.device_info.max_blocks_per_multiprocessor:",
        ctx.device_info.max_blocks_per_multiprocessor,
    )
    try:
        (free, total) = ctx.get_memory_info()
        print("Free memory:", free / (1024 * 1024), "MB")
        print("Total memory:", total / (1024 * 1024), "MB")
    except:
        print("Failed to get memory information")

    # -- Create GPU variables -- #
    # These don't need to be initialized to zero, they will be filled later

    host_quantum_state_re = ctx.enqueue_create_host_buffer[dtype](
        state_vector_size
    )
    host_quantum_state_im = ctx.enqueue_create_host_buffer[dtype](
        state_vector_size
    )

    host_gate_set_re = ctx.enqueue_create_host_buffer[dtype](
        gate_set_size * GATE_SIZE * GATE_SIZE
    )
    host_gate_set_im = ctx.enqueue_create_host_buffer[dtype](
        gate_set_size * GATE_SIZE * GATE_SIZE
    )

    host_control_bits_circuit = ctx.enqueue_create_host_buffer[DType.int32](
        circuit_number_control_gates * NUMBER_CONTROL_BITS * 2
    )

    # -- Initialize the quantum circuit to the |000⟩ state -- #
    quantum_state: StateVector = StateVector.from_bitstring("0" * num_qubits)
    # print("Initial quantum state:\n", quantum_state)

    # Wait for host buffers to be ready
    ctx.synchronize()

    # -- Fill host buffers -- #

    for i in range(state_vector_size):
        host_quantum_state_re[i] = quantum_state[i].re
        host_quantum_state_im[i] = quantum_state[i].im

    print("Initial state real part:", host_quantum_state_re)
    print("Initial state imaginary part:", host_quantum_state_im)

    for i in range(gate_set_size):
        gate = gate_set[i]
        for j in range(GATE_SIZE):
            for k in range(GATE_SIZE):
                index = gate_set_1qubit_layout(
                    IntTuple(i, j, k)
                )  # Get the index in the 1D buffer
                host_gate_set_re[index] = gate[j, k].re
                host_gate_set_im[index] = gate[j, k].im

    for i in range(circuit_number_control_gates):
        for j in range(NUMBER_CONTROL_BITS):
            for k in range(2):
                index = circuit_control_bits_layout(IntTuple(i, j, k))
                host_control_bits_circuit[index] = control_bits_list[i][j][k]

    # -- Copy host buffers to device buffers -- #
    quantum_state_re = ctx.enqueue_create_buffer[dtype](state_vector_size)
    quantum_state_im = ctx.enqueue_create_buffer[dtype](state_vector_size)

    gate_set_re = ctx.enqueue_create_buffer[dtype](
        gate_set_size * GATE_SIZE * GATE_SIZE
    )
    gate_set_im = ctx.enqueue_create_buffer[dtype](
        gate_set_size * GATE_SIZE * GATE_SIZE
    )

    control_bits_circuit = ctx.enqueue_create_buffer[DType.int32](
        circuit_number_control_gates * NUMBER_CONTROL_BITS * 2
    )
    current_control_gate_circuit = ctx.enqueue_create_buffer[DType.int32](1)

    # Create other buffers for functions

    quantum_state_out_re = ctx.enqueue_create_buffer[dtype](state_vector_size)
    quantum_state_out_im = ctx.enqueue_create_buffer[dtype](state_vector_size)

    quantum_state_re.enqueue_copy_from(host_quantum_state_re)
    quantum_state_im.enqueue_copy_from(host_quantum_state_im)

    gate_set_re.enqueue_copy_from(host_gate_set_re)
    gate_set_im.enqueue_copy_from(host_gate_set_im)

    control_bits_circuit.enqueue_copy_from(host_control_bits_circuit)

    ctx.enqueue_memset(current_control_gate_circuit, 0)
    ctx.enqueue_memset(quantum_state_out_re, 0.0)
    ctx.enqueue_memset(quantum_state_out_im, 0.0)

    # -- Create layout tensors for GPU operations -- #
    gate_set_re_tensor = LayoutTensor[mut=False, dtype, gate_set_1qubit_layout](
        gate_set_re.unsafe_ptr()
    )
    gate_set_im_tensor = LayoutTensor[mut=False, dtype, gate_set_1qubit_layout](
        gate_set_im.unsafe_ptr()
    )

    quantum_state_re_tensor = LayoutTensor[
        mut=True, dtype, state_vector_layout
    ](quantum_state_re.unsafe_ptr())
    quantum_state_im_tensor = LayoutTensor[
        mut=True, dtype, state_vector_layout
    ](quantum_state_im.unsafe_ptr())

    quantum_state_out_re_tensor = LayoutTensor[
        mut=True, dtype, state_vector_layout
    ](quantum_state_out_re.unsafe_ptr())
    quantum_state_out_im_tensor = LayoutTensor[
        mut=True, dtype, state_vector_layout
    ](quantum_state_out_im.unsafe_ptr())

    control_bits_circuit_tensor = LayoutTensor[
        mut=False, DType.int32, circuit_control_bits_layout
    ](control_bits_circuit.unsafe_ptr())
    current_control_gate_circuit_tensor = LayoutTensor[
        mut=True, DType.int32, Layout.row_major(1)
    ](current_control_gate_circuit.unsafe_ptr())

    # -- Apply circuit operations -- #

    # Gate 0
    # quantum_state = qubit_wise_multiply_gpu(
    #     Hadamard.matrix, 1, quantum_state
    # )
    ctx.enqueue_function[
        qubit_wise_multiply_inplace_gpu[number_control_bits=0]
    ](
        gate_set_re_tensor,
        gate_set_im_tensor,
        gate_set_dic[Hadamard.symbol],
        GATE_SIZE,
        1,  # target_qubit
        quantum_state_re_tensor,
        quantum_state_im_tensor,
        num_qubits,  # number_qubits
        state_vector_size,  # quantum_state_size
        quantum_state_out_re_tensor,
        quantum_state_out_im_tensor,
        control_bits_circuit_tensor,
        current_control_gate_circuit_tensor,
        grid_dim=blocks_per_grid,
        block_dim=threads_per_block,
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
    ctx.enqueue_function[
        qubit_wise_multiply_inplace_gpu[number_control_bits=0]
    ](
        gate_set_re_tensor,
        gate_set_im_tensor,
        gate_set_dic[PauliX.symbol],
        GATE_SIZE,
        2,  # target_qubit
        quantum_state_out_re_tensor,
        quantum_state_out_im_tensor,
        num_qubits,  # number_qubits
        state_vector_size,  # quantum_state_size
        quantum_state_re_tensor,
        quantum_state_im_tensor,
        control_bits_circuit_tensor,
        current_control_gate_circuit_tensor,
        grid_dim=blocks_per_grid,
        block_dim=threads_per_block,
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
    ctx.enqueue_function[
        qubit_wise_multiply_inplace_gpu[number_control_bits=1]
    ](
        gate_set_re_tensor,
        gate_set_im_tensor,
        gate_set_dic[PauliX.symbol],
        GATE_SIZE,
        0,  # target_qubit
        quantum_state_re_tensor,
        quantum_state_im_tensor,
        num_qubits,  # number_qubits
        state_vector_size,  # quantum_state_size
        quantum_state_out_re_tensor,
        quantum_state_out_im_tensor,
        control_bits_circuit_tensor,
        current_control_gate_circuit_tensor,
        grid_dim=blocks_per_grid,
        block_dim=threads_per_block,
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
    ctx.enqueue_function[
        qubit_wise_multiply_inplace_gpu[number_control_bits=0]
    ](
        gate_set_re_tensor,
        gate_set_im_tensor,
        gate_set_dic[PauliZ.symbol],
        GATE_SIZE,
        0,  # target_qubit
        quantum_state_out_re_tensor,
        quantum_state_out_im_tensor,
        num_qubits,  # number_qubits
        state_vector_size,  # quantum_state_size
        quantum_state_re_tensor,
        quantum_state_im_tensor,
        control_bits_circuit_tensor,
        current_control_gate_circuit_tensor,
        grid_dim=blocks_per_grid,
        block_dim=threads_per_block,
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
    ctx.enqueue_function[
        qubit_wise_multiply_inplace_gpu[number_control_bits=1]
    ](
        gate_set_re_tensor,
        gate_set_im_tensor,
        gate_set_dic[PauliX.symbol],
        GATE_SIZE,
        2,  # target_qubit
        quantum_state_re_tensor,
        quantum_state_im_tensor,
        num_qubits,  # number_qubits
        state_vector_size,  # quantum_state_size
        quantum_state_out_re_tensor,
        quantum_state_out_im_tensor,
        control_bits_circuit_tensor,
        current_control_gate_circuit_tensor,
        grid_dim=blocks_per_grid,
        block_dim=threads_per_block,
    )

    with quantum_state_out_re.map_to_host() as host_re, quantum_state_out_im.map_to_host() as host_im:
        print(
            (
                "After Pauli-X gate on qubit 2 with control on qubit 1"
                " (Final State):\nreal part:\n"
            ),
            host_re,
            "\nhost_rere[3]:",
            host_re[4],
            "\nhost_rere[4]:",
            host_re[5],
            "\nimaginary part:\n",
            host_im,
        )
        for i in range(state_vector_size):
            quantum_state[i] = ComplexFloat32(host_re[i], host_im[i])

    assert_state_vector_almost_equal(
        quantum_state, test_qubit_wise_multiply_figure1_reference
    )
