from math import sqrt

from testing import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
    assert_almost_equal,
)

from qlabs.base import StateVector

from qlabs.local_stdlib.complex import ComplexFloat32
from qlabs.local_stdlib import CustomList

alias test_qubit_wise_multiply_0_reference = StateVector(
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

alias test_qubit_wise_multiply_figure1_reference = StateVector(
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
)


def assert_state_vector_almost_equal(
    reference_state: StateVector, state: StateVector, message: String = ""
) -> None:
    """Asserts that two state vectors are almost equal.

    Args:
        reference_state: The reference state to compare against.
        state: The state to check for equality.
        message: An optional message to include in the assertion error.
    """
    assert_equal(
        reference_state.size(),
        state.size(),
        String("State vectors must have the same size.") + message,
    )
    assert_equal(
        reference_state.number_qubits(),
        state.number_qubits(),
        String("State vectors must have the same number of qubits.") + message,
    )
    for i in range(reference_state.size()):
        assert_almost_equal(
            reference_state[i].re,
            state[i].re,
            String(
                "Real parts of state vectors are not equal at index ",
                i,
                ". ",
            )
            + message,
        )
        assert_almost_equal(
            reference_state[i].im,
            state[i].im,
            String(
                "Imaginary parts of state vectors are not equal at index ",
                i,
                ". ",
            )
            + message,
        )
