from testing import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
    assert_almost_equal,
)

from testing_matrix import assert_matrix_almost_equal

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

from qlabs.local_stdlib import CustomList
from qlabs.local_stdlib.complex import ComplexFloat64


def test_partial_trace_all():
    """Test the partial trace operation on a 2-qubit state. Keep all qubits."""
    state: PureBasisState = PureBasisState(
        2,
        CustomList[ComplexFloat64, hint_trivial_type=True](
            ComplexFloat64(0, 0),
            ComplexFloat64(-0.5, 0),
            ComplexFloat64(0.7071067811863477, 0),
            ComplexFloat64(-0.5, 0),
        ),
    )
    matrix: ComplexMatrix = partial_trace(state, [])
    assert_matrix_almost_equal(
        matrix,
        ComplexMatrix(
            List[List[ComplexFloat64]](
                [
                    ComplexFloat64(0, 0),
                    ComplexFloat64(0, 0),
                    ComplexFloat64(0, 0),
                    ComplexFloat64(0, 0),
                ],
                [
                    ComplexFloat64(0, 0),
                    ComplexFloat64(0.25, 0),
                    ComplexFloat64(-0.3535533905929741, 0),
                    ComplexFloat64(0.25, 0),
                ],
                [
                    ComplexFloat64(0, 0),
                    ComplexFloat64(-0.3535533905929741, 0),
                    ComplexFloat64(0.5, 0),
                    ComplexFloat64(-0.3535533905929741, 0),
                ],
                [
                    ComplexFloat64(0, 0),
                    ComplexFloat64(0.25, 0),
                    ComplexFloat64(-0.3535533905929741, 0),
                    ComplexFloat64(0.25, 0),
                ],
            )
        ),
        "full trace",
    )


def test_partial_trace_qubit0():
    """Test the partial trace operation on a 2-qubit state. Trace out qubit 0, keep qubit 1.
    """
    state: PureBasisState = PureBasisState(
        2,
        CustomList[ComplexFloat64, hint_trivial_type=True](
            ComplexFloat64(0, 0),
            ComplexFloat64(-0.5, 0),
            ComplexFloat64(0.7071067811863477, 0),
            ComplexFloat64(-0.5, 0),
        ),
    )
    matrix = partial_trace(state, [0])
    assert_matrix_almost_equal(
        matrix,
        ComplexMatrix(
            List[List[ComplexFloat64]](
                [
                    ComplexFloat64(0, 0),
                    ComplexFloat64(0, 0),
                ],
                [
                    ComplexFloat64(0, 0),
                    ComplexFloat64(1, 0),
                ],
            )
        ),
    )


def test_partial_trace_qubit1():
    """Test the partial trace operation on a 2-qubit state. Trace out qubit 1, keep qubit 0.
    """
    state: PureBasisState = PureBasisState(
        2,
        CustomList[ComplexFloat64, hint_trivial_type=True](
            ComplexFloat64(0, 0),
            ComplexFloat64(-0.5, 0),
            ComplexFloat64(0.7071067811863477, 0),
            ComplexFloat64(-0.5, 0),
        ),
    )
    matrix = partial_trace(state, [1])
    assert_matrix_almost_equal(
        matrix,
        ComplexMatrix(
            List[List[ComplexFloat64]](
                [
                    ComplexFloat64(0, 0),
                    ComplexFloat64(0, 0),
                ],
                [
                    ComplexFloat64(0, 0),
                    ComplexFloat64(0.5, 0),
                ],
            )
        ),
    )
