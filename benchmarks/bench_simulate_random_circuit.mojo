from gpu.host import DeviceContext

from benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
)

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


fn simulate_random_circuit[num_qubits: Int, number_layers: Int]() -> None:
    """Simulates a random quantum circuit with the specified number of qubits and layers.

    Parameters:
        num_qubits: The number of qubits in the circuit.
        number_layers: The number of layers in the circuit.
    """

    qc: GateCircuit = GateCircuit(num_qubits)

    gates_list: List[Gate] = [Hadamard, PauliX, PauliY, PauliZ]

    # index: UnsafePointer[Int8] = UnsafePointer[Int8].alloc(2*num_qubits)
    # print("Creating random circuit...")
    # random.seed()  # Seed on current time
    # for _ in range(400):
    #     random.randint(index, 2*num_qubits, 0, len(gates_list) - 1)
    #     for i in range(num_qubits):
    #         qc = qc.apply(gates_list[Int(index[i])], i)
    #     qc = qc.barrier()
    #     for i in range(num_qubits - 1):
    #         qc = qc.apply(
    #             gates_list[Int(index[num_qubits + i])],
    #             i,
    #             controls=[(i + 1) % num_qubits],
    #             is_anti_control=[False],
    #         )
    #     qc = qc.barrier()

    index: UnsafePointer[Int8] = UnsafePointer[Int8].alloc(
        number_layers * 2 * num_qubits
    )
    random.seed()  # Seed on current time
    random.randint(
        index, number_layers * 2 * num_qubits, 0, len(gates_list) - 1
    )

    for iter in range(number_layers):
        for i in range(num_qubits):
            qc.apply(gates_list[Int(index[iter * num_qubits + i])](i))
        qc.barrier()
        for i in range(num_qubits - 1):
            qc.apply(
                gates_list[Int(index[iter * num_qubits + num_qubits + i])](
                    i, controls=[(i + 1) % num_qubits]
                ),
            )
        qc.barrier()

    initial_state_bitstring: String = (
        "0" * num_qubits
    )  # Initial state |000...0⟩
    initial_state: StateVector = StateVector.from_bitstring(
        initial_state_bitstring
    )

    qsimu = StateVectorSimulator(
        qc,
        initial_state=initial_state,
        optimisation_level=0,  # No optimisations for now
        verbose=False,
        # verbose_step_size=ShowAfterEachLayer,  # ShowAfterEachGate, ShowOnlyEnd
        verbose_step_size=ShowAfterEachGate,  # ShowAfterEachGate, ShowOnlyEnd
        # stop_at=ShowAfterEachGate,  # ShowAfterEachGate, ShowOnlyEnd # TODO implement that instead of having access to manual methods
    )

    for _ in range(100):
        _ = qsimu.run()


@parameter
@always_inline
fn benchmark_simulate_random_circuit[
    num_qubits: Int, number_layers: Int
](mut b: Bencher) raises:
    @parameter
    @always_inline
    fn simulate_random_circuit_workflow(ctx: DeviceContext) raises:
        simulate_random_circuit[num_qubits, number_layers]()

    bench_ctx = DeviceContext()
    b.iter_custom[simulate_random_circuit_workflow](bench_ctx)


def bench_simulate_random_circuit[
    max_number_qubits: Int = 10,
    max_number_layers: Int = 20,
    fixed_number_qubits: Int = 5,
    fixed_number_layers: Int = 10,
]():
    print("Running qubit_wise_multiply() CPU Benchmarks...")
    # print("SIMD width:", SIMD_WIDTH)
    print("-" * 80)
    bench_config = BenchConfig(max_iters=10, min_warmuptime_secs=0.2)
    bench = Bench(bench_config)

    @parameter
    for number_qubits in range(1, max_number_qubits + 1):
        bench.bench_function[
            benchmark_simulate_random_circuit[
                number_qubits, fixed_number_layers
            ]
        ](
            BenchId(
                "simulate_random_circuit_"
                + String(number_qubits)
                + "q_"
                + String(fixed_number_layers)
                + "l"
            )
        )

    @parameter
    for number_layers in range(1, max_number_layers + 1):
        bench.bench_function[
            benchmark_simulate_random_circuit[
                fixed_number_qubits, number_layers
            ]
        ](
            BenchId(
                "simulate_random_circuit_"
                + String(fixed_number_qubits)
                + "q_"
                + String(number_layers)
                + "l"
            )
        )

    print(bench)

    # bench.config.out_file = Path("out.csv")
    # bench.dump_report()

    print("simulate_random_circuit() CPU Benchmarks completed!")
    print("-" * 80)
