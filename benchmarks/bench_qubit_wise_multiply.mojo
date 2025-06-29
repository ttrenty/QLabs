from gpu.host import DeviceContext

from layout import Layout, LayoutTensor, IntTuple

from benchmark import Bench, BenchConfig, Bencher, BenchId, keep

from pathlib import Path
from os import makedirs

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

from qlabs.abstractions import (
    GateCircuit,
    StateVectorSimulator,
    ShowAfterEachGate,
    ShowAfterEachLayer,
    ShowOnlyEnd,
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
    min_number_qubits: Int = 15,
    max_number_qubits: Int = 25,
    number_qubits_step_size: Int = 1,
    min_number_layers: Int = 1,
    max_number_layers: Int = 2000,
    number_layers_step_size: Int = 200,
    fixed_number_qubits: Int = 5,
    fixed_number_layers: Int = 2,
]():
    print("Running qubit_wise_multiply_inplace() Benchmarks...")
    print("-" * 80)
    bench_config = BenchConfig(max_iters=10, min_warmuptime_secs=0.2)
    bench = Bench(bench_config)
    makedirs("data", exist_ok=True)

    @parameter
    for number_qubits in range(
        min_number_qubits, max_number_qubits + 1, number_qubits_step_size
    ):
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

    # print(bench)
    bench.config.out_file = Path("data/qubit_wise_multiply_inplace_qubits.csv")
    bench.dump_report()

    bench = Bench(bench_config)

    @parameter
    for number_layers in range(
        min_number_layers, max_number_layers + 1, number_layers_step_size
    ):
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

    # print(bench)
    bench.config.out_file = Path("data/qubit_wise_multiply_inplace_layers.csv")
    bench.dump_report()

    print("qubit_wise_multiply_inplace() Benchmarks completed!")
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
