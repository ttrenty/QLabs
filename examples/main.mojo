# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Imports              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

from sys import argv
import random
from collections.linked_list import LinkedList

# from complex import ComplexFloat64
from qlabs.local_stdlib.complex import ComplexFloat64
from qlabs.local_stdlib import CustomList

from qlabs.base import (
    PureBasisState,
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

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Examples             #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


fn simulate_figure1_circuit() -> None:
    """Simulates the circuit from Figure 1 in the paper."""
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
    # Initialize the quantum circuit to the |000⟩ state
    quantum_state: PureBasisState = PureBasisState.from_bitstring("000")

    print("Initial quantum state:\n", quantum_state)

    # Gate 0
    quantum_state = qubit_wise_multiply(Hadamard.matrix, 1, quantum_state)

    print("After Hadamard gate on qubit 1:\n", quantum_state)

    # Gate 1
    quantum_state = qubit_wise_multiply(PauliX.matrix, 2, quantum_state)

    print("After Pauli-X gate on qubit 2:\n", quantum_state)

    # Gate 2
    quantum_state = qubit_wise_multiply(
        PauliX.matrix, 0, quantum_state, [[1, 1]]
    )

    print(
        "After Pauli-X gate on qubit 0 with control on qubit 1:\n",
        quantum_state,
    )

    # Gate 3
    quantum_state = qubit_wise_multiply(PauliZ.matrix, 0, quantum_state)

    print("After Pauli-Z gate on qubit 0:\n", quantum_state)

    # Gate 4
    quantum_state = qubit_wise_multiply(
        PauliX.matrix, 2, quantum_state, [[1, 1]]
    )

    print(
        "After Pauli-X gate on qubit 2 with control on qubit 1:\n",
        quantum_state,
    )

    final_matrix = partial_trace(quantum_state, [])  # Trace out qubits

    print("Final quantum state after tracing out qubits:\n", final_matrix)


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
    initial_state: PureBasisState = PureBasisState.from_bitstring("000")

    qsimu = StateVectorSimulator(
        qc,
        initial_state=initial_state,
        optimisation_level=0,  # No optimisations for now
        verbose=True,
        verbose_step_size=ShowAfterEachGate,  # ShowAfterEachGate, ShowOnlyEnd
    )

    state = qsimu.run()

    print("Final quantum state:\n", state)


fn simulate_random_circuit(num_qubits: Int, number_layers: Int) -> None:
    """Simulates a random quantum circuit with the specified number of qubits and layers.

    Args:
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

    print(
        "Creating random circuit of",
        num_qubits,
        "qubits and",
        number_layers,
        "layers...",
    )
    index: UnsafePointer[Int8] = UnsafePointer[Int8].alloc(
        number_layers * 2 * num_qubits
    )
    random.seed()  # Seed on current time
    random.randint(
        index, number_layers * 2 * num_qubits, 0, len(gates_list) - 1
    )
    print("Random gates choices generated, applying them to the circuit...")
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

    print("Random circuit created has", qc.num_gates(), "total gates.")

    # print(qc)

    print("Running Simulations...")
    initial_state_bitstring: String = (
        "0" * num_qubits
    )  # Initial state |000...0⟩
    initial_state: PureBasisState = PureBasisState.from_bitstring(
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

    for i in range(1000):
        print("Iteration:", i, end="\r")
        _ = qsimu.run()

    print("")


fn simulate_figure4_circuit() -> None:
    """Simulates the circuit from Figure 4 in the paper."""
    print("Simulating Figure 1 circuit.\nCircuit design:")
    print(
        """
|0> --|H|---x--------|X|--|Y|---*-------
            |         |         |
|0> --------|---|X|---*---------x--|Z|--
            |    |              |
|0> --------x----o--------------x-------
    """
    )
    # Number of qubits
    num_qubits: Int = 3

    # Initialize the quantum circuit to the |000⟩ state
    quantum_state: PureBasisState = PureBasisState.from_bitstring("000")

    print("Intial quantum state:\n", quantum_state)

    # Apply Hadamard gate to qubit 0
    quantum_state = qubit_wise_multiply(Hadamard.matrix, 0, quantum_state)

    print("After Hadamard gate on qubit 1:\n", quantum_state)

    # Apply SWAP gate to qubits 0 and 2
    quantum_state = apply_swap(num_qubits, 0, 2, quantum_state)

    print("After SWAP gate on qubits 0 and 1:\n", quantum_state)

    # Apply anti-control Pauli-X gate to qubit 1 with control on qubit 2
    quantum_state = qubit_wise_multiply(
        PauliX.matrix, 1, quantum_state, [[2, 0]]
    )

    print(
        "After anti-control Pauli-X gate on qubit 1 with control on qubit 2:\n",
        quantum_state,
    )

    # Apply control Pauli-X gate to qubit 0 with control on qubit 1
    quantum_state = qubit_wise_multiply(
        PauliX.matrix, 0, quantum_state, [[1, 1]]
    )

    print(
        "After control Pauli-X gate on qubit 0 with control on qubit 1:\n",
        quantum_state,
    )

    # Apply control Pauli-Y gate to qubit 0
    quantum_state = qubit_wise_multiply(PauliY.matrix, 0, quantum_state)

    print(
        "After control Pauli-Y gate on qubit 0 with control on qubit 1:\n",
        quantum_state,
    )

    # Apply SWAP gate to qubits 1 and 2 with control on qubit 0
    quantum_state = apply_swap(num_qubits, 1, 2, quantum_state, [[0, 1]])

    print(
        "After SWAP gate on qubits 1 and 2 with control on qubit 0:\n",
        quantum_state,
    )

    # Apply control Pauli-Z gate to qubit 1
    quantum_state = qubit_wise_multiply(
        PauliZ.matrix, 1, quantum_state, [[0, 1]]
    )

    print(
        "After control Pauli-Z gate on qubit 1 with control on qubit 0:\n",
        quantum_state,
    )


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
        initial_state=PureBasisState.from_bitstring("000"),
        optimisation_level=0,  # No optimisations for now
        verbose=True,
        verbose_step_size=ShowAfterEachGate,  # ShowAfterEachGate, ShowOnlyEnd
    )
    final_state = qsimu.run()
    print("Final quantum state:\n", final_state)


fn presentation() -> None:
    """
    Simulates the circuit from Figure 4 in the paper.
    """
    print("Simulating Figure 4 circuit.\nCircuit design:")
    print(
        """
|0> --|H|---*---|Y|--
            |       
|0> -------|H|-------
    """
    )
    num_qubits: Int = 2
    qc: GateCircuit = GateCircuit(num_qubits)

    qc.apply_gates(
        Hadamard(0),
        Hadamard(1, controls=[0]),
        Z(0),
    )

    print("Quantum circuit created:\n", qc)

    qsimu = StateVectorSimulator(
        qc,
        initial_state=PureBasisState.from_bitstring("000"),
        optimisation_level=0,  # No optimisations for now
        verbose=True,
        verbose_step_size=ShowAfterEachGate,  # ShowAfterEachGate, ShowOnlyEnd
    )
    final_state = qsimu.run()
    print("Final quantum state:\n", final_state)


fn test_density_matrix() -> None:
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
        initial_state=PureBasisState.from_bitstring("00"),
        optimisation_level=0,  # No optimisations for now
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


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Tests                #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# fn test_all() -> None:
#     """
#     Runs all tests and examples.
#     """
# qc = qc.apply_layer([
#     Hadamard([1]),
#     NOT([2], controls=[1], anti_controls=[])
#     ])
# qc = qc.apply_layer([
#     NOT([0], controls=[1], anti_controls=[])
# ])
# qc = qc.apply_layer([
#     PauliZ([0]),
#     NOT([2], controls=[1], anti_controls=[])
#     ])

# # Create the initial state |000⟩
# initial_state: PureBasisState = PureBasisState.from_bitstring("000")

# qsimu = StateVectorSimulator(
#     qc,
#     initial_state=initial_state,
#     optimisation_level=0,  # No optimisations for now
#     verbose=True,
#     # verbose_step_size=ShowAfterEachLayer,  # ShowAfterEachGate, ShowOnlyEnd
#     verbose_step_size=ShowAfterEachGate,  # ShowAfterEachGate, ShowOnlyEnd
# )

# while (qsimu.circuit.num_gates() != 0):
#     qsimu, state = qsimu.next_gate(state)
#     print("New quantum state after gate:\n", state)
#     # qsimu, state = qsimu.next_layer(state)
#     # or
#     # qsimu, state = qsimu.next_block(state)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Main                 #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def main():
    args = argv()
    number_qubits: Int = 10
    number_layers: Int = 20
    if len(args) == 3:
        try:
            number_qubits = Int(args[1])
            number_layers = Int(args[2])
        except ValueError:
            print(
                "Invalid arguments. Using default values: 3 qubits and 10"
                " layers."
            )
    else:
        print("Usage: ./main [number_of_qubits] [number_of_layers]")

    # simulate_figure1_circuit()

    simulate_figure1_circuit_abstract()

    simulate_random_circuit(number_qubits, number_layers)

    # simulate_figure4_circuit()

    # simulate_figure4_circuit_abstract()

    # presentation()

    # test_density_matrix()
