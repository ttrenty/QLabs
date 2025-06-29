from gpu.host import DeviceContext

from layout import Layout, LayoutTensor, IntTuple

from benchmark import Bench, BenchConfig, Bencher, BenchId, keep

from pathlib import Path
from os import makedirs

import random

from qlabs.base import (
    StateVector,
    Gate,
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
)

from qlabs.base.gpu import qubit_wise_multiply_inplace_gpu

from qlabs.abstractions import (
    GateCircuit,
    StateVectorSimulator,
    ShowAfterEachGate,
    ShowAfterEachLayer,
    ShowOnlyEnd,
)


alias dtype = DType.float32

alias GATE_SIZE = 2
alias NUMBER_CONTROL_BITS = 1


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

    @parameter
    @always_inline
    fn qubit_wise_multiply_inplace_gpu_workflow(ctx: DeviceContext) raises:
        """Simulates on GPU a random quantum circuit with the specified number of qubits and layers.
        """
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

        alias max_threads_per_block = ctx.device_info.max_thread_block_size
        # alias max_threads_per_block = 1024  # Maximum threads per block in CUDA

        # alias sm_count = ctx.device_info.sm_count
        # alias max_blocks_per_multiprocessor = ctx.device_info.max_blocks_per_multiprocessor
        # alias max_number_blocks = sm_count * max_blocks_per_multiprocessor

        alias blocks_per_grid = (
            total_threads + max_threads_per_block - 1
        ) // max_threads_per_block

        threads_per_block = (
            max_threads_per_block,
            1,
            1,
        )

        if total_threads < max_threads_per_block:
            threads_per_block = (
                total_threads,
                1,
                1,
            )

        # alias blocks_per_grid = (1)

        # threads_per_block = (
        #     1,
        #     1,
        #     1,
        # )

        # print("vector size:", state_vector_size)
        # print("blocks per grid:", blocks_per_grid)
        # print("threads per block[0]:", threads_per_block[0])

        var control_bits_list: List[List[List[Int]]] = [
            [[1, 1]],  # Control on qubit 1 and is control because flag=1
            [[1, 1]],  # Control on qubit 1 and is control because flag=1
        ]

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
        quantum_state: StateVector = StateVector.from_bitstring(
            "0" * num_qubits
        )
        # print("Initial quantum state:\n", quantum_state)

        # Wait for host buffers to be ready
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

        for i in range(circuit_number_control_gates):
            for j in range(NUMBER_CONTROL_BITS):
                for k in range(2):
                    index = circuit_control_bits_layout(IntTuple(i, j, k))
                    host_control_bits_circuit[index] = control_bits_list[i][j][
                        k
                    ]

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

        quantum_state_out_re = ctx.enqueue_create_buffer[dtype](
            state_vector_size
        )
        quantum_state_out_im = ctx.enqueue_create_buffer[dtype](
            state_vector_size
        )

        quantum_state_re.enqueue_copy_from(host_quantum_state_re)
        quantum_state_im.enqueue_copy_from(host_quantum_state_im)

        gate_set_re.enqueue_copy_from(host_gate_set_re)
        gate_set_im.enqueue_copy_from(host_gate_set_im)

        control_bits_circuit.enqueue_copy_from(host_control_bits_circuit)

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

        current_state = 0
        for layer in range(number_layers):
            for qubit in range(num_qubits):
                if current_state == 0:
                    ctx.enqueue_function[
                        qubit_wise_multiply_inplace_gpu[
                            state_vector_size=state_vector_size,
                            gate_set_size=gate_set_size,
                            circuit_number_control_gates=circuit_number_control_gates,
                            number_control_bits=0,
                        ]
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
                        qubit_wise_multiply_inplace_gpu[
                            state_vector_size=state_vector_size,
                            gate_set_size=gate_set_size,
                            circuit_number_control_gates=circuit_number_control_gates,
                            number_control_bits=0,
                        ]
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

    bench_ctx = DeviceContext()
    b.iter_custom[qubit_wise_multiply_inplace_gpu_workflow](bench_ctx)


def bench_qubit_wise_multiply_inplace_gpu[
    min_number_qubits: Int = 15,
    max_number_qubits: Int = 25,
    number_qubits_step_size: Int = 1,
    min_number_layers: Int = 1,
    max_number_layers: Int = 2000,
    number_layers_step_size: Int = 200,
    fixed_number_qubits: Int = 5,
    fixed_number_layers: Int = 2,
]():
    print("Running qubit_wise_multiply_inplace_gpu() Benchmarks...")
    print("-" * 80)
    bench_config = BenchConfig(max_iters=10, min_warmuptime_secs=0.2)
    bench = Bench(bench_config)

    makedirs("data", exist_ok=True)

    @parameter
    for number_qubits in range(
        min_number_qubits, max_number_qubits + 1, number_qubits_step_size
    ):
        bench.bench_function[
            benchmark_qubit_wise_multiply_inplace_gpu[
                number_qubits, fixed_number_layers
            ]
        ](
            BenchId(
                "qubit_wise_multiply_inplace_gpu_"
                + String(number_qubits)
                + "q_"
                + String(fixed_number_layers)
                + "l"
            )
        )

    bench.config.out_file = Path(
        "data/qubit_wise_multiply_inplace_gpu_qubits.csv"
    )
    bench.dump_report()
    bench = Bench(bench_config)

    @parameter
    for number_layers in range(
        min_number_layers, max_number_layers + 1, number_layers_step_size
    ):
        bench.bench_function[
            benchmark_qubit_wise_multiply_inplace_gpu[
                fixed_number_qubits, number_layers
            ]
        ](
            BenchId(
                "qubit_wise_multiply_inplace_gpu_"
                + String(fixed_number_qubits)
                + "q_"
                + String(number_layers)
                + "l"
            )
        )

    bench.config.out_file = Path(
        "data/qubit_wise_multiply_inplace_gpu_layers.csv"
    )
    bench.dump_report()

    print("qubit_wise_multiply_inplace_gpu() Benchmarks completed!")
    print("-" * 80)
