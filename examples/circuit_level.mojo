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

    alias num_qubits: Int = 3

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

    # Simulating using CPU
    qsimu = StateVectorSimulator(
        qc,
        initial_state=initial_state,
        use_gpu_if_available=False,
        verbose=True,
        verbose_step_size=ShowOnlyEnd,  # or ShowAfterEachGate, ShowAfterEachLayer
    )

    # # Simulating using GPU
    # qsimu = StateVectorSimulator[
    #     gpu_num_qubits=num_qubits,
    #     gpu_gate_ordered_set= [Hadamard, PauliZ, NOT],
    #     gpu_control_gate_count=2,
    #     gpu_control_bits_list= [[[1, 1]], [[1, 1]]],
    # ](
    #     qc,
    #     initial_state=initial_state,
    #     use_gpu_if_available=True,
    #     verbose=True,
    #     verbose_step_size=ShowOnlyEnd,
    # )

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


fn simulate_random_circuit[
    number_qubits: Int, use_gpu: Bool = True
](number_layers: Int) -> None:
    """Simulates a random quantum circuit with the specified number of qubits and layers.

    Parameters:
        number_qubits: The number of qubits in the circuit.

    Args:
        number_layers: The number of layers in the circuit.
    """

    alias gates_list: List[Gate] = [Hadamard, PauliX, PauliY, PauliZ]

    qc: GateCircuit = GateCircuit(number_qubits)

    print(
        "Creating random circuit with",
        number_qubits,
        "qubits and",
        number_layers,
        "layers.",
    )
    index: UnsafePointer[Int8] = UnsafePointer[Int8].alloc(
        number_layers * 2 * number_qubits
    )
    random.seed(42)
    random.randint(
        index, number_layers * 1 * number_qubits, 0, len(gates_list) - 1
    )
    for layer in range(number_layers):
        for qubit in range(number_qubits):
            gate = gates_list[Int(index[layer * number_qubits + qubit])]
            qc.apply(gate(qubit))

    print("Random circuit created has", qc.num_gates(), "total gates.")

    # Initial state |000...0⟩
    quantum_state: StateVector = StateVector.from_bitstring("0" * number_qubits)

    print("Running Simulations...")
    qsimu = StateVectorSimulator[
        gpu_num_qubits=number_qubits,
        gpu_gate_ordered_set=gates_list,
    ](
        qc,
        initial_state=quantum_state,
        use_gpu_if_available=use_gpu,
        verbose=False,
    )

    for i in range(20):
        print("Simulating Circuit:", i, end="\r")
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
