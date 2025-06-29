import random

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
    partial_trace,
)

from qlabs.abstractions import (
    GateCircuit,
    StateVectorSimulator,
    ShowAfterEachGate,
    ShowAfterEachLayer,
    ShowOnlyEnd,
)


fn simulate_figure1_circuit_abstract() -> None:
    """
    Simulates the circuit from Figure 1 in the paper.
    """
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

    num_qubits: Int = 3

    qc: GateCircuit = GateCircuit(num_qubits)

    qc.apply_gates(
        Hadamard(1),
        NOT(2),
        NOT(0, controls=[1]),
        PauliZ(0),
        NOT(2, controls=[1]),
    )

    # Create the initial state |000⟩
    initial_state: StateVector = StateVector.from_bitstring("000")

    qsimu = StateVectorSimulator(
        qc,
        initial_state=initial_state,
        verbose=True,
        verbose_step_size=ShowAfterEachGate,  # ShowAfterEachGate, ShowOnlyEnd
    )

    state = qsimu.run()

    print("Final quantum state:\n", state)


fn simulate_figure4_circuit_abstract() -> None:
    """
    Simulates the circuit from Figure 4 in the paper.
    """
    print("Simulating Figure 4 circuit.\nCircuit design:")
    print(
        """
|0> --|H|---x--------|X|--|Y|---*-------
            |         |         |
|0> --------|---|X|---*---------x--|Z|--
            |    |              |
|0> --------x----o--------------x-------
    """
    )
    num_qubits: Int = 3
    qc: GateCircuit = GateCircuit(num_qubits)

    qc.apply_gates(
        Hadamard(0),
        SWAP(0, 2),
        NOT(1, controls=[], anti_controls=[2]),
        NOT(0, controls=[1], anti_controls=[]),
        PauliY(0),
        SWAP(1, 2, controls=[0]),
        PauliZ(1),
    )

    print("Quantum circuit created:\n", qc)

    qsimu = StateVectorSimulator(
        qc,
        initial_state=StateVector.from_bitstring("000"),
        verbose=True,
        verbose_step_size=ShowAfterEachGate,  # ShowAfterEachGate, ShowOnlyEnd
    )
    final_state = qsimu.run()
    print("Final quantum state:\n", final_state)


fn simulate_random_circuit(number_qubits: Int, number_layers: Int) -> None:
    """Simulates a random quantum circuit with the specified number of qubits and layers.

    Args:
        number_qubits: The number of qubits in the circuit.
        number_layers: The number of layers in the circuit.
    """

    qc: GateCircuit = GateCircuit(number_qubits)

    gates_list: List[Gate] = [Hadamard, PauliX, PauliY, PauliZ]

    # index: UnsafePointer[Int8] = UnsafePointer[Int8].alloc(2*number_qubits)
    # print("Creating random circuit...")
    # random.seed()  # Seed on current time
    # for _ in range(400):
    #     random.randint(index, 2*number_qubits, 0, len(gates_list) - 1)
    #     for i in range(number_qubits):
    #         qc = qc.apply(gates_list[Int(index[i])], i)
    #     qc = qc.barrier()
    #     for i in range(number_qubits - 1):
    #         qc = qc.apply(
    #             gates_list[Int(index[number_qubits + i])],
    #             i,
    #             controls=[(i + 1) % number_qubits],
    #             is_anti_control=[False],
    #         )
    #     qc = qc.barrier()

    print(
        "Creating random circuit of",
        number_qubits,
        "qubits and",
        number_layers,
        "layers...",
    )
    index: UnsafePointer[Int8] = UnsafePointer[Int8].alloc(
        number_layers * 2 * number_qubits
    )
    random.seed()  # Seed on current time
    random.randint(
        index, number_layers * 2 * number_qubits, 0, len(gates_list) - 1
    )
    print("Random gates choices generated, applying them to the circuit...")
    for iter in range(number_layers):
        for i in range(number_qubits):
            qc.apply(gates_list[Int(index[iter * number_qubits + i])](i))
        qc.barrier()
        for i in range(number_qubits - 1):
            qc.apply(
                gates_list[
                    Int(index[iter * number_qubits + number_qubits + i])
                ](i, controls=[(i + 1) % number_qubits]),
            )
        qc.barrier()

    print("Random circuit created has", qc.num_gates(), "total gates.")

    # print(qc)

    print("Running Simulations...")
    initial_state_bitstring: String = (
        "0" * number_qubits
    )  # Initial state |000...0⟩
    initial_state: StateVector = StateVector.from_bitstring(
        initial_state_bitstring
    )

    qsimu = StateVectorSimulator(
        qc,
        initial_state=initial_state,
        verbose=False,
        # verbose_step_size=ShowAfterEachLayer,  # ShowAfterEachGate, ShowOnlyEnd
        verbose_step_size=ShowAfterEachGate,  # ShowAfterEachGate, ShowOnlyEnd
        # stop_at=ShowAfterEachGate,  # ShowAfterEachGate, ShowOnlyEnd # TODO implement that instead of having access to manual methods
    )

    for i in range(1000):
        print("Iteration:", i, end="\r")
        _ = qsimu.run()

    print("")


fn try_density_matrix() -> None:
    """
    Returns the density matrix of the given quantum state.
    If qubits is empty, returns the full density matrix.
    """
    num_qubits: Int = 2
    qc: GateCircuit = GateCircuit(num_qubits)

    qc.apply_gates(
        Hadamard(0),
        Hadamard(1, controls=[0]),
        Z(0),
        X(1),
    )

    print("Quantum circuit created:\n", qc)

    qsimu = StateVectorSimulator(
        qc,
        initial_state=StateVector.from_bitstring("00"),
        verbose=True,
        verbose_step_size=ShowAfterEachGate,  # ShowAfterEachGate, ShowOnlyEnd
    )
    final_state = qsimu.run()
    print("Final quantum state:\n", final_state)

    matrix = final_state.to_density_matrix()
    print("Density matrix:\n", matrix)
    other_matrix = partial_trace(final_state, [])  # Empty list means full trace
    print("Partial trace matrix:\n", other_matrix)
    other_matrix_0 = partial_trace(
        final_state, [0]
    )  # Empty list means full trace
    print("Partial trace matrix qubit 0:\n", other_matrix_0)
    other_matrix_1 = partial_trace(
        final_state, [1]
    )  # Empty list means full trace
    print("Partial trace matrix qubit 1:\n", other_matrix_1)


fn try_get_purity() -> None:
    """
    Returns the density matrix of the given quantum state.
    If qubits is empty, returns the full density matrix.
    """
    num_qubits: Int = 2
    qc: GateCircuit = GateCircuit(num_qubits)

    qc.apply_gates(
        Hadamard(0),
        Hadamard(1, controls=[0]),
        Z(0),
        X(1),
    )

    print("Quantum circuit created:\n", qc)

    qsimu = StateVectorSimulator(
        qc,
        initial_state=StateVector.from_bitstring("00"),
        verbose=True,
        verbose_step_size=ShowAfterEachGate,  # ShowAfterEachGate, ShowOnlyEnd
    )
    final_state = qsimu.run()
    print("Final quantum state:\n", final_state)

    purity = final_state.purity()
    print("Purity of the quantum state:", purity)

    purity0 = final_state.purity([0, 1])
    print("Purity of the quantum state:", purity0)

    purity1 = final_state.purity([0])
    print("Purity of qubit 0:", purity1)

    # for QOL
    # list_purity = final_state.purity(0, 1)
    # print("Purity of qubit 0:", list_purity[0])
    # print("Purity of qubit 1:", list_purity[1])

    normalised_purity = final_state.normalised_purity()
    print("Normalised purity of the quantum state:", normalised_purity)


fn try_measument() -> None:
    """
    Returns the density matrix of the given quantum state.
    If qubits is empty, returns the full density matrix.
    """
    num_qubits: Int = 2
    qc: GateCircuit = GateCircuit(num_qubits)

    qc.apply_gates(
        Hadamard(0),
        Hadamard(1, controls=[0]),
        Z(0),
        X(1),
    )

    print("Quantum circuit created:\n", qc)

    qsimu = StateVectorSimulator(
        qc,
        initial_state=StateVector.from_bitstring("00"),
        verbose=True,
        verbose_step_size=ShowAfterEachGate,  # ShowAfterEachGate, ShowOnlyEnd
    )
    final_state = qsimu.run()
    print("Final quantum state:\n", final_state)

    purity = final_state.purity()
    print("Purity of the quantum state:", purity)

    purity0 = final_state.purity([0, 1])
    print("Purity of the quantum state:", purity0)

    purity1 = final_state.purity([0])
    print("Purity of qubit 0:", purity1)

    # for QOL
    # list_purity = final_state.purity(0, 1)
    # print("Purity of qubit 0:", list_purity[0])
    # print("Purity of qubit 1:", list_purity[1])

    normalised_purity = final_state.normalised_purity()
    print("Normalised purity of the quantum state:", normalised_purity)
