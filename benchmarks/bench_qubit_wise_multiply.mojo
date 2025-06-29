from gpu.host import DeviceContext

from layout import Layout, LayoutTensor, IntTuple

from benchmark import Bench, BenchConfig, Bencher, BenchId, keep

from pathlib import Path

import random

from qlabs.base import (
    StateVector,
    ComplexMatrix,
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
    qubit_wise_multiply,
    qubit_wise_multiply_inplace,
    qubit_wise_multiply_extended,
    apply_swap,
    partial_trace,
)

from qlabs.base.gpu import qubit_wise_multiply_inplace_gpu

from qlabs.abstractions import (
    GateCircuit,
    StateVectorSimulator,
    ShowAfterEachGate,
    ShowAfterEachLayer,
    ShowOnlyEnd,
)


# alias BLOCKS_PER_GRID = 1
# alias THREADS_PER_BLOCK = (1, 1)
alias dtype = DType.float32

alias GATE_SIZE = 2
alias NUMBER_CONTROL_BITS = 1
# TODO have NUMBER_CONTROL_BITS be a list defining each gates specific control bits count
alias CIRCUIT_NUMBER_CONTROL_GATES = 1
alias circuit_control_bits_layout = Layout.row_major(
    CIRCUIT_NUMBER_CONTROL_GATES, NUMBER_CONTROL_BITS, 2
)

alias gate_1qubit_layout = Layout.row_major(GATE_SIZE, GATE_SIZE)
alias STATE_VECTOR_SIZE = 8
alias state_vector_3qubits_layout = Layout.row_major(STATE_VECTOR_SIZE)
alias control_bits_layout = Layout.row_major(NUMBER_CONTROL_BITS, 2)

alias gate_set_dic: Dict[String, Int] = {
    Hadamard.symbol: 0,
    PauliX.symbol: 1,
    PauliY.symbol: 2,
    PauliZ.symbol: 3,
}
alias GATE_SET_SIZE = 4
alias gate_set_1qubit_layout = Layout.row_major(
    GATE_SET_SIZE, GATE_SIZE, GATE_SIZE
)
alias gate_set_1qubit_vectorized_layout = Layout.row_major(
    GATE_SET_SIZE, GATE_SIZE, GATE_SIZE, 2
)


@parameter
@always_inline
fn benchmark_qubit_wise_multiply[
    num_qubits: Int, number_layers: Int
](mut b: Bencher) raises:
    gates_list: List[Gate] = [Hadamard, PauliX, PauliY, PauliZ]

    indexes: UnsafePointer[Int8] = UnsafePointer[Int8].alloc(
        number_layers * 2 * num_qubits
    )
    random.seed()  # Seed on current time
    random.randint(
        indexes, number_layers * 2 * num_qubits, 0, len(gates_list) - 1
    )

    @parameter
    @always_inline
    fn qubit_wise_multiply_workflow(ctx: DeviceContext) raises:
        """Simulates a random quantum circuit with the specified number of qubits and layers.
        """

        # Initialize the quantum circuit to the |0⟩ state
        quantum_state: StateVector = StateVector.from_bitstring(
            "0" * num_qubits
        )

        for layer in range(number_layers):
            for i in range(num_qubits):
                quantum_state = qubit_wise_multiply(
                    gates_list[Int(indexes[layer * num_qubits + i])].matrix,
                    i,
                    quantum_state,
                )
            for i in range(num_qubits - 1):
                quantum_state = qubit_wise_multiply(
                    gates_list[
                        Int(indexes[layer * num_qubits + num_qubits + i])
                    ].matrix,
                    i,
                    quantum_state,
                    [[(i + 1) % num_qubits, 1]],
                )

    bench_ctx = DeviceContext()
    b.iter_custom[qubit_wise_multiply_workflow](bench_ctx)


@parameter
@always_inline
fn benchmark_qubit_wise_multiply_inplace[
    num_qubits: Int, number_layers: Int
](mut b: Bencher) raises:
    gates_list: List[Gate] = [Hadamard, PauliX, PauliY, PauliZ]

    indexes: UnsafePointer[Int8] = UnsafePointer[Int8].alloc(
        number_layers * 2 * num_qubits
    )
    random.seed()  # Seed on current time
    random.randint(
        indexes, number_layers * 2 * num_qubits, 0, len(gates_list) - 1
    )

    @parameter
    @always_inline
    fn qubit_wise_multiply_inplace_workflow(ctx: DeviceContext) raises:
        """Simulates a random quantum circuit with the specified number of qubits and layers.
        """

        # TODO report github that this is inconvenient because it won't compile
        # error: argument of 'qubit_wise_multiply_inplace' call allows writing a memory location previously writable through another aliased argument
        # quantum_states = List[StateVector](
        #     StateVector.from_bitstring("0" * num_qubits),
        #     StateVector.from_bitstring("0" * num_qubits),
        # )

        # Why would this work while the above doesn't?
        # quantum_states: Dict[Int, StateVector] = {
        #     0: StateVector.from_bitstring("0" * num_qubits),
        #     1: StateVector.from_bitstring("0" * num_qubits),
        # }

        quantum_state_0 = StateVector.from_bitstring("0" * num_qubits)
        quantum_state_1 = StateVector.from_bitstring("0" * num_qubits)

        current_state = 0
        for layer in range(number_layers):
            for i in range(num_qubits):
                # NOTE Works but is slow with the dictionary
                # qubit_wise_multiply_inplace(
                #     gates_list[Int(indexes[layer * num_qubits + i])].matrix,
                #     i,
                #     quantum_states[current_state],
                #     quantum_states[1 - current_state],
                # )
                # NOTE: Fast buty doesn't actually use the next state for new operations
                # qubit_wise_multiply_inplace(
                #     gates_list[Int(indexes[layer * num_qubits + i])].matrix,
                #     i,
                #     quantum_state_0,
                #     quantum_state_1,
                # )
                if current_state == 0:
                    qubit_wise_multiply_inplace(
                        gates_list[Int(indexes[layer * num_qubits + i])].matrix,
                        i,
                        quantum_state_0,
                        quantum_state_1,
                    )
                    current_state = 1
                else:
                    qubit_wise_multiply_inplace(
                        gates_list[Int(indexes[layer * num_qubits + i])].matrix,
                        i,
                        quantum_state_1,
                        quantum_state_0,
                    )
                    current_state = 0
            for i in range(num_qubits - 1):
                # qubit_wise_multiply_inplace(
                #     gates_list[
                #         Int(indexes[layer * num_qubits + num_qubits + i])
                #     ].matrix,
                #     i,
                #     quantum_states[current_state],
                #     quantum_states[1 - current_state],
                #     [[(i + 1) % num_qubits, 1]],
                # )
                # current_state = 1 - current_state
                # qubit_wise_multiply_inplace(
                #     gates_list[
                #         Int(indexes[layer * num_qubits + num_qubits + i])
                #     ].matrix,
                #     i,
                #     quantum_state_0,
                #     quantum_state_1,
                #     [[(i + 1) % num_qubits, 1]],
                # )
                if current_state == 0:
                    qubit_wise_multiply_inplace(
                        gates_list[
                            Int(indexes[layer * num_qubits + num_qubits + i])
                        ].matrix,
                        i,
                        quantum_state_0,
                        quantum_state_1,
                        [[(i + 1) % num_qubits, 1]],
                    )
                    current_state = 1
                else:
                    qubit_wise_multiply_inplace(
                        gates_list[
                            Int(indexes[layer * num_qubits + num_qubits + i])
                        ].matrix,
                        i,
                        quantum_state_1,
                        quantum_state_0,
                        [[(i + 1) % num_qubits, 1]],
                    )
                    current_state = 0

    bench_ctx = DeviceContext()
    b.iter_custom[qubit_wise_multiply_inplace_workflow](bench_ctx)


@parameter
@always_inline
fn benchmark_qubit_wise_multiply_inplace_gpu[
    num_qubits: Int, number_layers: Int
](mut b: Bencher) raises:
    # gates_list: List[Gate] = [Hadamard, PauliX, PauliY, PauliZ]

    # indexes: UnsafePointer[Int8] = UnsafePointer[Int8].alloc(
    #     number_layers * 1 * num_qubits
    # )
    # random.seed()  # Seed on current time
    # random.randint(
    #     indexes, number_layers * 2 * num_qubits, 0, len(gates_list) - 1
    # )

    bench_ctx = DeviceContext()

    alias state_vector_size = 1 << num_qubits
    alias state_vector_layout = Layout.row_major(state_vector_size)

    alias total_threads = 2 * state_vector_size

    # alias max_threads_per_block = 1024
    alias sm_count = bench_ctx.device_info.sm_count
    alias max_blocks_per_multiprocessor = bench_ctx.device_info.max_blocks_per_multiprocessor

    # alias max_number_blocks = sm_count * max_blocks_per_multiprocessor
    alias max_number_blocks = 128
    alias max_threads_per_block = bench_ctx.device_info.max_thread_block_size

    print("state_vector_size:", state_vector_size)

    # try:
    #     print("BEFORE:")
    #     (free, total) = bench_ctx.get_memory_info()
    #     print("Free memory:", free / (1024 * 1024), "MB")
    #     print("Total memory:", total / (1024 * 1024), "MB")
    # except:
    #     print("Failed to get memory information")

    @parameter
    @always_inline
    fn qubit_wise_multiply_inplace_gpu_workflow(ctx: DeviceContext) raises:
        """Simulates on GPU a random quantum circuit with the specified number of qubits and layers.
        """
        # gate_set: List[Gate] = [Hadamard, PauliX, PauliY, PauliZ]

        try:
            print("1.BEFORE ALLOCATING:")
            (free, total) = ctx.get_memory_info()
            print("Free memory:", free / (1024 * 1024), "MB")
            print("Total memory:", total / (1024 * 1024), "MB")
        except:
            print("Failed to get memory information")

        # blocks_per_grid = (
        #     total_threads + max_threads_per_block - 1
        # ) // max_threads_per_block

        # if blocks_per_grid >= max_number_blocks:
        #     blocks_per_grid = max_number_blocks - 1

        blocks_per_grid = 1

        threads_per_block = (
            1,
            1,
            1,
        )

        # @parameter
        # if total_threads < max_threads_per_block:
        #     threads_per_block = (
        #         total_threads,
        #         1,
        #         1,
        #     )  # 1D block of threads

        # print(
        #     "blocks_per_grid:",
        #     blocks_per_grid,
        #     "max_number_blocks:",
        #     max_number_blocks,
        # )
        # print("threads_per_block[0]:", threads_per_block[0])

        # var control_bits_list: List[List[List[Int]]] = [
        #     [[1, 1]],
        #     # [[1, 1]],
        # ]

        # control_bits_list: List[List[List[Int]]] = []

        # -- Create GPU variables -- #

        host_quantum_state_re = ctx.enqueue_create_host_buffer[dtype](
            state_vector_size
        )
        host_quantum_state_im = ctx.enqueue_create_host_buffer[dtype](
            state_vector_size
        )

        host_gate_set_re = ctx.enqueue_create_host_buffer[dtype](
            GATE_SET_SIZE * GATE_SIZE * GATE_SIZE
        )
        host_gate_set_im = ctx.enqueue_create_host_buffer[dtype](
            GATE_SET_SIZE * GATE_SIZE * GATE_SIZE
        )

        # host_control_bits_circuit = ctx.enqueue_create_host_buffer[DType.int32](
        #     CIRCUIT_NUMBER_CONTROL_GATES * NUMBER_CONTROL_BITS * 2
        # )

        quantum_state: StateVector = StateVector.from_bitstring("000")

        # Wait for host buffers to be ready
        ctx.synchronize()

        # -- Fill host buffers -- #

        for i in range(state_vector_size):
            host_quantum_state_re[i] = quantum_state[i].re
            host_quantum_state_im[i] = quantum_state[i].im

        # for i in range(GATE_SET_SIZE):
        #     gate = gate_set[i]
        #     for j in range(GATE_SIZE):
        #         for k in range(GATE_SIZE):
        #             index = gate_set_1qubit_layout(
        #                 IntTuple(i, j, k)
        #             )  # Get the index in the 1D buffer
        #             host_gate_set_re[index] = gate[j, k].re
        #             host_gate_set_im[index] = gate[j, k].im

        # for i in range(CIRCUIT_NUMBER_CONTROL_GATES):
        #     for j in range(NUMBER_CONTROL_BITS):
        #         for k in range(2):
        #             index = circuit_control_bits_layout(IntTuple(i, j, k))
        #             host_control_bits_circuit[index] = control_bits_list[i][j][
        #                 k
        #             ]

        # -- Copy host buffers to device buffers -- #
        quantum_state_re = ctx.enqueue_create_buffer[dtype](state_vector_size)
        quantum_state_im = ctx.enqueue_create_buffer[dtype](state_vector_size)

        gate_set_re = ctx.enqueue_create_buffer[dtype](
            GATE_SET_SIZE * GATE_SIZE * GATE_SIZE
        )
        gate_set_im = ctx.enqueue_create_buffer[dtype](
            GATE_SET_SIZE * GATE_SIZE * GATE_SIZE
        )

        control_bits_circuit = ctx.enqueue_create_buffer[DType.int32](
            CIRCUIT_NUMBER_CONTROL_GATES * NUMBER_CONTROL_BITS * 2
        )
        current_control_gate_circuit = ctx.enqueue_create_buffer[DType.int32](1)

        # Create other buffers for functions

        quantum_state_out_re = ctx.enqueue_create_buffer[dtype](
            state_vector_size
        )
        quantum_state_out_im = ctx.enqueue_create_buffer[dtype](
            state_vector_size
        )

        quantum_state_re.enqueue_copy_from(host_quantum_state_re)
        quantum_state_im.enqueue_copy_from(host_quantum_state_im)

        # gate_set_re.enqueue_copy_from(host_gate_set_re)
        # gate_set_im.enqueue_copy_from(host_gate_set_im)
        ctx.enqueue_memset(gate_set_re, 0.0)
        ctx.enqueue_memset(gate_set_im, 0.0)

        # control_bits_circuit.enqueue_copy_from(host_control_bits_circuit)
        ctx.enqueue_memset(control_bits_circuit, 0)

        # TODO report that this create a runtime error only in this context not when
        # running the same code in a standalone script
        # ctx.enqueue_memset(current_control_gate_circuit, 0.0)
        ctx.enqueue_memset(current_control_gate_circuit, 0)
        ctx.enqueue_memset(quantum_state_out_re, 0.0)
        ctx.enqueue_memset(quantum_state_out_im, 0.0)

        # -- Create layout tensors for GPU operations -- #
        gate_set_re_tensor = LayoutTensor[
            mut=False, dtype, gate_set_1qubit_layout
        ](gate_set_re.unsafe_ptr())
        gate_set_im_tensor = LayoutTensor[
            mut=False, dtype, gate_set_1qubit_layout
        ](gate_set_im.unsafe_ptr())

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

        ctx.synchronize()
        try:
            print("2.AFTER ALLOCATING:")
            (free, total) = ctx.get_memory_info()
            print("Free memory:", free / (1024 * 1024), "MB")
            print("Total memory:", total / (1024 * 1024), "MB")
        except:
            print("Failed to get memory information")

        # print("HERE")
        current_state = 0
        for layer in range(number_layers):
            # print("Layer:", layer, "out of", number_layers)
            for qubit in range(num_qubits):
                # print("Applying gate on qubit:", i, "of", num_qubits)
                # print(
                #     "gate symbol: ",
                #     gates_list[Int(indexes[layer * num_qubits + i])].symbol,
                # )
                # print(
                #     "gate index: ",
                #     gate_set_dic[
                #         gates_list[Int(indexes[layer * num_qubits + i])].symbol
                #     ],
                # )
                if current_state == 0:
                    ctx.enqueue_function[
                        qubit_wise_multiply_inplace_gpu[number_control_bits=0]
                    ](
                        gate_set_re_tensor,
                        gate_set_im_tensor,
                        gate_set_dic[Hadamard.symbol],
                        # gate_set_dic[
                        #     gates_list[
                        #         Int(indexes[layer * num_qubits + qubit])
                        #     ].symbol
                        # ],
                        GATE_SIZE,
                        qubit,  # target_qubit
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
                    current_state = 1
                else:
                    ctx.enqueue_function[
                        qubit_wise_multiply_inplace_gpu[number_control_bits=0]
                    ](
                        gate_set_re_tensor,
                        gate_set_im_tensor,
                        gate_set_dic[Hadamard.symbol],
                        # gate_set_dic[
                        #     gates_list[
                        #         Int(indexes[layer * num_qubits + qubit])
                        #     ].symbol
                        # ],
                        GATE_SIZE,
                        qubit,  # target_qubit
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
                    current_state = 0

        keep(quantum_state_re.unsafe_ptr())
        keep(quantum_state_im.unsafe_ptr())
        keep(quantum_state_out_re.unsafe_ptr())
        keep(quantum_state_out_im.unsafe_ptr())
        keep(gate_set_re.unsafe_ptr())
        keep(gate_set_im.unsafe_ptr())
        keep(control_bits_circuit.unsafe_ptr())
        keep(current_control_gate_circuit.unsafe_ptr())

        ctx.synchronize()
        try:
            print("3. AFTER AUTOMATIC FREE:")
            (free, total) = ctx.get_memory_info()
            print("Free memory:", free / (1024 * 1024), "MB")
            print("Total memory:", total / (1024 * 1024), "MB")
        except:
            print("Failed to get memory information")

    b.iter_custom[qubit_wise_multiply_inplace_gpu_workflow](bench_ctx)


@parameter
@always_inline
fn benchmark_qubit_wise_multiply_extended[
    num_qubits: Int, number_layers: Int
](mut b: Bencher) raises:
    gates_list: List[Gate] = [Hadamard, PauliX, PauliY, PauliZ]

    indexes: UnsafePointer[Int8] = UnsafePointer[Int8].alloc(
        number_layers * 2 * num_qubits
    )
    random.seed()  # Seed on current time
    random.randint(
        indexes, number_layers * 2 * num_qubits, 0, len(gates_list) - 1
    )

    @parameter
    @always_inline
    fn qubit_wise_multiply_extended_workflow(ctx: DeviceContext) raises:
        """Simulates a random quantum circuit with the specified number of qubits and layers.
        """

        # Initialize the quantum circuit to the |0⟩ state
        quantum_state: StateVector = StateVector.from_bitstring(
            "0" * num_qubits
        )

        for layer in range(number_layers):
            for i in range(num_qubits):
                quantum_state = qubit_wise_multiply_extended(
                    1,
                    gates_list[Int(indexes[layer * num_qubits + i])].matrix,
                    [i],
                    quantum_state,
                )
            for i in range(num_qubits - 1):
                quantum_state = qubit_wise_multiply_extended(
                    1,
                    gates_list[
                        Int(indexes[layer * num_qubits + num_qubits + i])
                    ].matrix,
                    [i],
                    quantum_state,
                    [[(i + 1) % num_qubits, 1]],
                )

    bench_ctx = DeviceContext()
    b.iter_custom[qubit_wise_multiply_extended_workflow](bench_ctx)


# def run_benchmark[
#     max_number_qubits: Int = 10,
#     max_number_layers: Int = 20,
#     fixed_number_qubits: Int = 5,
#     fixed_number_layers: Int = 10,
#     # TODO how to do this without errors?
#     benchmark_function: fn[Int, Int] (
#         mut b: Bencher
#     ) raises capturing -> None = benchmark_qubit_wise_multiply_extended,
# ]():
#     print("Running aaa() Benchmarks...")
#     print("-" * 80)
#     bench_config = BenchConfig(max_iters=10, min_warmuptime_secs=0.2)
#     bench = Bench(bench_config)

#     @parameter
#     for number_qubits in range(1, max_number_qubits + 1):
#         bench.bench_function[
#             benchmark_function[number_qubits, fixed_number_layers]
#         ](
#             BenchId(
#                 "aaa_"
#                 + String(number_qubits)
#                 + "q_"
#                 + String(fixed_number_layers)
#                 + "l"
#             )
#         )

#     @parameter
#     for number_layers in range(1, max_number_layers + 1):
#         bench.bench_function[
#             benchmark_function[fixed_number_qubits, number_layers]
#         ](
#             BenchId(
#                 "aaa_"
#                 + String(fixed_number_qubits)
#                 + "q_"
#                 + String(number_layers)
#                 + "l"
#             )
#         )

#     print(bench)

#     # bench.config.out_file = Path("out.csv")
#     # bench.dump_report()

#     print("aaa() Benchmarks completed!")
#     print("-" * 80)


def bench_qubit_wise_multiply[
    max_number_qubits: Int = 10,
    max_number_layers: Int = 20,
    fixed_number_qubits: Int = 5,
    fixed_number_layers: Int = 10,
]():
    # run_benchmark[
    #     max_number_qubits,
    #     max_number_layers,
    #     fixed_number_qubits,
    #     fixed_number_layers,
    #     benchmark_function=benchmark_qubit_wise_multiply,
    # ]()
    print("Running qubit_wise_multiply() Benchmarks...")
    print("-" * 80)
    bench_config = BenchConfig(max_iters=10, min_warmuptime_secs=0.2)
    bench = Bench(bench_config)

    @parameter
    for number_qubits in range(1, max_number_qubits + 1):
        bench.bench_function[
            benchmark_qubit_wise_multiply[number_qubits, fixed_number_layers]
        ](
            BenchId(
                "qubit_wise_multiply_"
                + String(number_qubits)
                + "q_"
                + String(fixed_number_layers)
                + "l"
            )
        )

    @parameter
    for number_layers in range(1, max_number_layers + 1):
        bench.bench_function[
            benchmark_qubit_wise_multiply[fixed_number_qubits, number_layers]
        ](
            BenchId(
                "qubit_wise_multiply_"
                + String(fixed_number_qubits)
                + "q_"
                + String(number_layers)
                + "l"
            )
        )

    print(bench)

    # bench.config.out_file = Path("out.csv")
    # bench.dump_report()

    print("qubit_wise_multiply() Benchmarks completed!")
    print("-" * 80)


def bench_qubit_wise_multiply_inplace[
    # max_number_qubits: Int = 10,
    # max_number_layers: Int = 20,
    # fixed_number_qubits: Int = 5,
    # fixed_number_layers: Int = 10,
    max_number_qubits: Int = 16,
    max_number_layers: Int = 2000,
    fixed_number_qubits: Int = 5,
    fixed_number_layers: Int = 200,
]():
    print("Running qubit_wise_multiply_inplace() Benchmarks...")
    print("-" * 80)
    bench_config = BenchConfig(max_iters=10, min_warmuptime_secs=0.2)
    bench = Bench(bench_config)

    @parameter
    for number_qubits in range(1, max_number_qubits + 1, 5):
        bench.bench_function[
            benchmark_qubit_wise_multiply_inplace[
                number_qubits, fixed_number_layers
            ]
        ](
            BenchId(
                "qubit_wise_multiply_inplace_"
                + String(number_qubits)
                + "q_"
                + String(fixed_number_layers)
                + "l"
            )
        )

    @parameter
    for number_layers in range(1, max_number_layers + 1, 200):
        bench.bench_function[
            benchmark_qubit_wise_multiply_inplace[
                fixed_number_qubits, number_layers
            ]
        ](
            BenchId(
                "qubit_wise_multiply_inplace_"
                + String(fixed_number_qubits)
                + "q_"
                + String(number_layers)
                + "l"
            )
        )

    print(bench)

    # bench.config.out_file = Path("out.csv")
    # bench.dump_report()

    print("qubit_wise_multiply_inplace() Benchmarks completed!")
    print("-" * 80)


def bench_qubit_wise_multiply_inplace_gpu[
    max_number_qubits: Int = 25,
    max_number_layers: Int = 2000,
    fixed_number_qubits: Int = 5,
    fixed_number_layers: Int = 2,
]():
    print("Running qubit_wise_multiply_inplace() Benchmarks...")
    print("-" * 80)
    bench_config = BenchConfig(max_iters=10, min_warmuptime_secs=0.2)
    bench = Bench(bench_config)

    @parameter
    for number_qubits in range(15, max_number_qubits + 1, 1):
        bench.bench_function[
            benchmark_qubit_wise_multiply_inplace_gpu[
                number_qubits, fixed_number_layers
            ]
        ](
            BenchId(
                "qubit_wise_multiply_inplace_"
                + String(number_qubits)
                + "q_"
                + String(fixed_number_layers)
                + "l"
            )
        )

    @parameter
    for number_layers in range(1, max_number_layers + 1, 200):
        bench.bench_function[
            benchmark_qubit_wise_multiply_inplace_gpu[
                fixed_number_qubits, number_layers
            ]
        ](
            BenchId(
                "qubit_wise_multiply_inplace_"
                + String(fixed_number_qubits)
                + "q_"
                + String(number_layers)
                + "l"
            )
        )

    print(bench)

    # bench.config.out_file = Path("out.csv")
    # bench.dump_report()

    print("qubit_wise_multiply_inplace_gpu() Benchmarks completed!")
    print("-" * 80)


def bench_qubit_wise_multiply_extended[
    max_number_qubits: Int = 10,
    max_number_layers: Int = 20,
    fixed_number_qubits: Int = 5,
    fixed_number_layers: Int = 10,
]():
    print("Running qubit_wise_multiply() Benchmarks...")
    print("-" * 80)
    bench_config = BenchConfig(max_iters=10, min_warmuptime_secs=0.2)
    bench = Bench(bench_config)

    @parameter
    for number_qubits in range(1, max_number_qubits + 1):
        bench.bench_function[
            benchmark_qubit_wise_multiply_extended[
                number_qubits, fixed_number_layers
            ]
        ](
            BenchId(
                "qubit_wise_multiply_extended_"
                + String(number_qubits)
                + "q_"
                + String(fixed_number_layers)
                + "l"
            )
        )

    @parameter
    for number_layers in range(1, max_number_layers + 1):
        bench.bench_function[
            benchmark_qubit_wise_multiply_extended[
                fixed_number_qubits, number_layers
            ]
        ](
            BenchId(
                "qubit_wise_multiply_extended_"
                + String(fixed_number_qubits)
                + "q_"
                + String(number_layers)
                + "l"
            )
        )

    print(bench)

    # bench.config.out_file = Path("out.csv")
    # bench.dump_report()

    print("qubit_wise_multiply_extended() Benchmarks completed!")
    print("-" * 80)
