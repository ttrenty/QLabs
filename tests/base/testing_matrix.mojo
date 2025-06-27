from testing import (
    assert_true,
    assert_false,
    assert_equal,
    assert_not_equal,
    assert_almost_equal,
)

from qlabs.base import ComplexMatrix


def assert_matrix_almost_equal(
    reference_matrix: ComplexMatrix, matrix: ComplexMatrix, message: String = ""
) -> None:
    """Asserts that two complex matrices are almost equal.

    Args:
        reference_matrix: The reference matrix to compare against.
        matrix: The matrix to check for equality.
    """
    assert_equal(
        reference_matrix.size(),
        matrix.size(),
        String("Matrices must have the same size.") + message,
    )
    for i in range(reference_matrix.size()):
        for j in range(reference_matrix.size()):
            assert_almost_equal(
                reference_matrix[i, j].re,
                matrix[i, j].re,
                String(
                    "Real parts of matrices are not equal at indices (",
                    i,
                    ", ",
                    j,
                    "). ",
                )
                + message,
            )
            assert_almost_equal(
                reference_matrix[i, j].im,
                matrix[i, j].im,
                String(
                    "Imaginary parts of matrices are not equal at indices (",
                    i,
                    ", ",
                    j,
                    "). ",
                )
                + message,
            )
