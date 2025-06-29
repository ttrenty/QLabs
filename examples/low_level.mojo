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
    quantum_state: StateVector = StateVector.from_bitstring("000")

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


fn simulate_figure1_circuit_inplace() -> None:
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
    quantum_state_0: StateVector = StateVector.from_bitstring("000")
    quantum_state_1: StateVector = StateVector.from_bitstring("000")

    print("Initial quantum state:\n", quantum_state_0)

    # Gate 0
    qubit_wise_multiply_inplace(
        Hadamard.matrix, 1, quantum_state_0, quantum_state_1
    )

    print("After Hadamard gate on qubit 1:\n", quantum_state_1)

    # Gate 1
    qubit_wise_multiply_inplace(
        PauliX.matrix, 2, quantum_state_1, quantum_state_0
    )

    print("After Pauli-X gate on qubit 2:\n", quantum_state_0)

    # Gate 2
    qubit_wise_multiply_inplace(
        PauliX.matrix, 0, quantum_state_0, quantum_state_1, [[1, 1]]
    )

    print(
        "After Pauli-X gate on qubit 0 with control on qubit 1:\n",
        quantum_state_1,
    )

    # Gate 3
    qubit_wise_multiply_inplace(
        PauliZ.matrix, 0, quantum_state_1, quantum_state_0
    )

    print("After Pauli-Z gate on qubit 0:\n", quantum_state_0)

    # Gate 4
    qubit_wise_multiply_inplace(
        PauliX.matrix, 2, quantum_state_0, quantum_state_1, [[1, 1]]
    )

    print(
        "After Pauli-X gate on qubit 2 with control on qubit 1:\n",
        quantum_state_1,
    )

    final_matrix = partial_trace(quantum_state_1, [])  # Trace out qubits

    print("Final quantum state after tracing out qubits:\n", final_matrix)


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
    quantum_state: StateVector = StateVector.from_bitstring("000")

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
