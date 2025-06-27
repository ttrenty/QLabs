# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Imports              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

from ..local_stdlib import CustomList
from ..local_stdlib.complex import ComplexFloat64


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Structs              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@fieldwise_init
# struct PureBasisState[check_bounds:Bool = False](Copyable, Movable, Stringable, Writable):
struct PureBasisState(Copyable, Movable, Stringable, Writable):
    """Represents a pure quantum state as a basis state in the computational basis.

    Uses a vector of complex numbers to represent the amplitudes of the basis states.
    The squared magnitudes of the amplitudes sum to 1, representing the probabilities
    of measuring the state in each basis state.
    """

    var num_qubits: Int
    """The number of qubits in the state, which determines the size of the state vector."""
    var state_vector: CustomList[ComplexFloat64, hint_trivial_type=True]
    """The state vector representing the amplitudes of the basis states."""

    fn __init__(out self, size: Int):
        """Initializes a PureBasisState with the given size.

        Args:
            size: The size of the vector, which is 2^n for n qubits.
        """
        self.num_qubits = 0
        self.state_vector = CustomList[ComplexFloat64, hint_trivial_type=True](
            length=size, fill=ComplexFloat64(0.0, 0.0)
        )
        # self.state_vector.memset_zero()

    @always_inline
    fn __getitem__(self, index: Int) -> ComplexFloat64:
        # @parameter
        # if check_bounds:
        #     if index < 0 or index >= self.size():
        #         print("ERROR: Index", index, "is out of bounds for state vector of size", self.size())
        #         return ComplexFloat64(0.0, 0.0)
        # else:
        return self.state_vector[index]

    @always_inline
    fn __setitem__(mut self, index: Int, value: ComplexFloat64) -> None:
        self.state_vector[index] = value

    fn __str__(self) -> String:
        """Returns a beautifully formatted string representation of the PureBasisState.
        """
        string: String = "PureBasisState:\n"
        for i in range(self.size()):
            amplitude = self.state_vector[i]
            # amplitude_str: String = String(amplitude)
            amplitude_re: Float64 = amplitude.re
            amplitude_im: Float64 = amplitude.im
            if amplitude_im == 0.0 and amplitude_re == 0.0:
                amplitude_str: String = String(Int(amplitude_re))
            elif amplitude_im == 0.0:
                # amplitude_str = String(round(amplitude_re, 2))
                amplitude_str = String(amplitude_re)
            elif amplitude_re == 0.0:
                # amplitude_str = String(round(amplitude_im, 2)) + "i"
                amplitude_str = String(amplitude_im) + "i"
            else:
                amplitude_str = (
                    String(round(amplitude_re, 2))
                    + " + "
                    + String(round(amplitude_im, 2))
                    + "i"
                )
            bitstring: String = ""  # base 2 with leading zeros
            for j in range(self.num_qubits):
                if (i & (1 << j)) != 0:
                    bitstring = "1" + bitstring
                else:
                    bitstring = "0" + bitstring

            string += "  |" + bitstring + "⟩: " + amplitude_str + "\n"
        return string

    @staticmethod
    fn from_bitstring(bitstring: String) -> Self:
        """Returns a PureBasisState corresponding to the given bitstring.

        Params:
            bitstring: A string of '0's and '1's representing the state, with
                        the least significant qubit (top one) (LSB) at the start.

        Examples:
        ```mojo
        state = PureBasisState.from_bitstring("110")
        ```

        Returns:
            A PureBasisState object with the appropriate data initialized.
        """
        num_qubits: Int = len(bitstring)

        # state_vector: List[ComplexFloat64] = [ComplexFloat64(0.0, 0.0)] * (
        #     1 << num_qubits
        # )  # 2^num_qubits

        state_vector = CustomList[ComplexFloat64, hint_trivial_type=True](
            length=1 << num_qubits, fill=ComplexFloat64(0.0, 0.0)
        )  # 2^num_qubits
        state_vector.memset_zero()  # Initialize the state vector with zeros

        # Put coefficent correspondin to the bitstring to 1
        index: Int = 0
        i: Int = 0
        for bit in bitstring.codepoints():
            if bit == Codepoint.ord("1"):
                index |= 1 << i  # Set the bit at position i
            i += 1

        state_vector[index] = ComplexFloat64(
            1.0, 0.0
        )  # Set the amplitude for the state to 1
        return Self(num_qubits, state_vector)

    fn write_to[W: Writer](self, mut writer: W) -> None:
        writer.write(String(self))

    @always_inline
    fn size(self) -> Int:
        """Returns the size of the state vector, which is 2^num_qubits."""
        return len(self.state_vector)

    @always_inline
    fn number_qubits(self) -> Int:
        """Returns the number of qubits in the state."""
        return self.num_qubits

    fn fill_zeros(mut self) -> None:
        """Fills the state vector with zeros."""
        self.state_vector.memset_zero()  # Set all elements to zero
        # for i in range(self.size()):
        #     self.state_vector[i] = ComplexFloat64(0.0, 0.0)  # Set each amplitude to zero

    fn is_valid_state(self) -> Bool:
        """Checks if the state vector is a valid quantum state.

        A valid quantum state must have the squared magnitudes of the amplitudes
        summing to 1.

        Returns:
            True if the state is valid, False otherwise.
        """
        if self.number_qubits() == 0:
            return False  # No qubits means no valid state

        # Use the buil-in methods of the Complex type
        squared_norm: Float64 = 0.0
        for amplitude in self.state_vector:
            squared_norm += amplitude.squared_norm()
        # Check if the squared norm is approximately equal to 1
        return (
            abs(squared_norm - 1.0) < 1e-6
        )  # Allow a small tolerance for floating-point errors

    fn conjugate(self) -> Self:
        """Returns the conjugate of the pure state.

        The conjugate is computed by taking the complex conjugate of each amplitude
        in the state vector.

        Returns:
            A new PureBasisState with the conjugated amplitudes.
        """
        conjugated_state: Self = Self(self.num_qubits, self.state_vector.copy())

        for i in range(self.size()):
            conjugated_state.state_vector[i] = self.state_vector[i].conjugate()
        return conjugated_state

    fn to_density_matrix(self) -> ComplexMatrix:
        """Returns the density matrix of the pure state.

        The density matrix is computed as the outer product of the state vector with itself.

        Returns:
            A ComplexMatrix representing the density matrix of the pure state.
        """
        size: Int = self.size()
        density_matrix: ComplexMatrix = ComplexMatrix(size, size)

        for i in range(size):
            for j in range(size):
                density_matrix[i, j] = (
                    self.state_vector[i] * self.state_vector[j].conjugate()
                )
        return density_matrix


@fieldwise_init
struct ComplexMatrix(Copyable, Movable, Stringable, Writable):
    """Represents a 2D matrix of complex numbers.

    This is used to represent quantum gates in the form of matrices.
    """

    var matrix: List[List[ComplexFloat64]]
    """The 2D matrix representation of the quantum gate."""

    fn __init__(out self, rows: Int, cols: Int):
        """Initializes a ComplexMatrix with the given number of rows and columns.

        Args:
            rows: The number of rows in the matrix.
            cols: The number of columns in the matrix.
        """
        self.matrix = List[List[ComplexFloat64]](
            length=rows,
            fill=List[ComplexFloat64](
                length=cols, fill=ComplexFloat64(0.0, 0.0)
            ),
        )
        # # Initialize the matrix with zeros
        # for i in range(rows):
        #     for j in range(cols):
        #         self.matrix[i][j] = ComplexFloat64(0.0, 0.0)

    @always_inline
    fn __getitem__(self, row: Int, col: Int) -> ComplexFloat64:
        return self.matrix[row][col]

    @always_inline
    fn __setitem__(mut self, row: Int, col: Int, value: ComplexFloat64) -> None:
        """Sets the value at the specified row and column in the matrix.
        Args:
            row: The row index of the matrix.
            col: The column index of the matrix.
            value: The complex number to set at the specified position.
        """
        # if row < 0 or row >= len(self.matrix) or col < 0 or col >= len(self.matrix[row]):
        #     print("Error: Index out of bounds for matrix.")
        #     return
        self.matrix[row][col] = value

    fn __str__(self) -> String:
        """Return a beautifully formatted string representation of the ComplexMatrix.
        Values that are 0 are omitted, and the matrix is printed in a human-readable format.
        """
        size = len(self.matrix)
        string = (
            "ComplexMatrix: (size: " + String(size) + "x" + String(size) + ")\n"
        )
        for row in self.matrix:
            row_str: String = ""
            for value in row:
                row_str += String(value) + " "
            string += row_str.strip() + "\n"
        return String(string.strip())

    fn write_to[W: Writer](self, mut writer: W) -> None:
        writer.write(String(self))

    @always_inline
    fn mult(self, other: PureBasisState, mut buffer: PureBasisState) -> None:
        """Multiplies the matrix by a complex vector and stores the result in a buffer.

        Args:
            other: The complex vector to multiply with the matrix.
            buffer: The buffer to store the result of the multiplication.
        """
        if len(self.matrix) != other.size():
            print(
                "Error: Matrix and vector sizes do not match for"
                " multiplication."
            )
            return

        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                buffer[i] += self.matrix[i][j] * other[j]

    fn size(self) -> Int:
        """Returns the size of the matrix, which is the number of rows (or columns).
        """
        return len(self.matrix)
