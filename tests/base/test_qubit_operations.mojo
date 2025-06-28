from testing import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
    assert_almost_equal,
)

from testing_matrix import assert_matrix_almost_equal

from testing_state_vector import assert_state_vector_almost_equal

from math import sqrt

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

from qlabs.local_stdlib import CustomList
from qlabs.local_stdlib.complex import ComplexFloat32


def test_qubit_wise_multiply_0():
    """Simulate a small circuit"""

    quantum_state: StateVector = StateVector.from_bitstring("000")

    quantum_state = qubit_wise_multiply(Hadamard.matrix, 0, quantum_state)
    quantum_state = qubit_wise_multiply(
        PauliX.matrix, 1, quantum_state, [[0, 1]]
    )
    quantum_state = qubit_wise_multiply(Hadamard.matrix, 2, quantum_state)

    assert_state_vector_almost_equal(
        quantum_state,
        StateVector(
            3,
            CustomList[ComplexFloat32, hint_trivial_type=True](
                ComplexFloat32(0.5, 0),
                ComplexFloat32(0, 0),
                ComplexFloat32(0, 0),
                ComplexFloat32(0.5, 0),
                ComplexFloat32(0.5, 0),
                ComplexFloat32(0, 0),
                ComplexFloat32(0, 0),
                ComplexFloat32(0.5, 0),
            ),
        ),
    )


def test_qubit_wise_multiply_figure1():
    """Simulates the circuit from Figure 1 in the paper.

    |0> -------|X|--|Z|--
                |
    |0> --|H|---*----*---
                     |
    |0> --|X|-------|X|--

    """

    # Initialize the quantum circuit to the |000⟩ state
    quantum_state: StateVector = StateVector.from_bitstring("000")

    # Gate 0
    quantum_state = qubit_wise_multiply(Hadamard.matrix, 1, quantum_state)

    # Gate 1
    quantum_state = qubit_wise_multiply(PauliX.matrix, 2, quantum_state)

    # Gate 2
    quantum_state = qubit_wise_multiply(
        PauliX.matrix, 0, quantum_state, [[1, 1]]
    )

    # Gate 3
    quantum_state = qubit_wise_multiply(PauliZ.matrix, 0, quantum_state)

    # Gate 4
    quantum_state = qubit_wise_multiply(
        PauliX.matrix, 2, quantum_state, [[1, 1]]
    )

    assert_state_vector_almost_equal(
        quantum_state,
        StateVector(
            3,
            CustomList[ComplexFloat32, hint_trivial_type=True](
                ComplexFloat32(0, 0),
                ComplexFloat32(0, 0),
                ComplexFloat32(0, 0),
                ComplexFloat32(-1.0 / Float32(sqrt(2.0)), 0),
                ComplexFloat32(1.0 / Float32(sqrt(2.0)), 0),
                ComplexFloat32(0, 0),
                ComplexFloat32(0, 0),
                ComplexFloat32(0, 0),
            ),
        ),
    )


def test_partial_trace_all():
    """Test the partial trace operation on a 2-qubit state. Keep all qubits."""
    state: StateVector = StateVector(
        2,
        CustomList[ComplexFloat32, hint_trivial_type=True](
            ComplexFloat32(0, 0),
            ComplexFloat32(-0.5, 0),
            ComplexFloat32(0.7071067811863477, 0),
            ComplexFloat32(-0.5, 0),
        ),
    )
    matrix: ComplexMatrix = partial_trace(state, [])
    assert_matrix_almost_equal(
        matrix,
        ComplexMatrix(
            List[List[ComplexFloat32]](
                [
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0, 0),
                ],
                [
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0.25, 0),
                    ComplexFloat32(-0.3535533905929741, 0),
                    ComplexFloat32(0.25, 0),
                ],
                [
                    ComplexFloat32(0, 0),
                    ComplexFloat32(-0.3535533905929741, 0),
                    ComplexFloat32(0.5, 0),
                    ComplexFloat32(-0.3535533905929741, 0),
                ],
                [
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0.25, 0),
                    ComplexFloat32(-0.3535533905929741, 0),
                    ComplexFloat32(0.25, 0),
                ],
            )
        ),
        "full trace",
    )


def test_partial_trace_sec67():
    """Test the partial trace operation on a 2-qubit state. Trace out qubit 1, keep qubit 0.
    """
    state: StateVector = StateVector(
        3,
        CustomList[ComplexFloat32, hint_trivial_type=True](
            ComplexFloat32(0.5, 0),
            ComplexFloat32(0, 0),
            ComplexFloat32(0, 0),
            ComplexFloat32(0.5, 0),
            ComplexFloat32(0.5, 0),
            ComplexFloat32(0, 0),
            ComplexFloat32(0, 0),
            ComplexFloat32(0.5, 0),
        ),
    )

    matrix = partial_trace(state, [0, 1])
    assert_matrix_almost_equal(
        matrix,
        ComplexMatrix(
            List[List[ComplexFloat32]](
                [ComplexFloat32(0.5, 0), ComplexFloat32(0.5, 0)],
                [ComplexFloat32(0.5, 0), ComplexFloat32(0.5, 0)],
            )
        ),
        "partial trace qubit 0 and 1",
    )

    matrix = partial_trace(state, [2])
    assert_matrix_almost_equal(
        matrix,
        ComplexMatrix(
            List[List[ComplexFloat32]](
                [
                    ComplexFloat32(0.5, 0),
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0.5, 0),
                ],
                [
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0, 0),
                ],
                [
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0, 0),
                ],
                [
                    ComplexFloat32(0.5, 0),
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0.5, 0),
                ],
            )
        ),
        "partial trace qubits 2",
    )

    matrix = partial_trace(state, [1, 2])
    assert_matrix_almost_equal(
        matrix,
        ComplexMatrix(
            List[List[ComplexFloat32]](
                [ComplexFloat32(0.5, 0), ComplexFloat32(0, 0)],
                [ComplexFloat32(0, 0), ComplexFloat32(0.5, 0)],
            )
        ),
        "partial trace qubits 1 and 2",
    )

    matrix = partial_trace(state, [0])
    assert_matrix_almost_equal(
        matrix,
        ComplexMatrix(
            List[List[ComplexFloat32]](
                [
                    ComplexFloat32(0.25, 0),
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0.25, 0),
                    ComplexFloat32(0, 0),
                ],
                [
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0.25, 0),
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0.25, 0),
                ],
                [
                    ComplexFloat32(0.25, 0),
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0.25, 0),
                    ComplexFloat32(0, 0),
                ],
                [
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0.25, 0),
                    ComplexFloat32(0, 0),
                    ComplexFloat32(0.25, 0),
                ],
            )
        ),
        "partial trace qubit 1",
    )
