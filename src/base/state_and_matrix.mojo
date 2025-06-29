# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Imports              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

from collections.linked_list import LinkedList

from ..local_stdlib import CustomList
from ..local_stdlib.complex import ComplexFloat32

from .qubits_operations import partial_trace

# GPU imports

from layout import Layout, LayoutTensor


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Structs              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@fieldwise_init
# struct StateVector[check_bounds:Bool = False](Copyable, Movable, Stringable, Writable):
struct StateVector(Copyable, Movable, Stringable, Writable):
    """Represents a pure quantum state as a basis state in the computational basis.

    Uses a vector of complex numbers to represent the amplitudes of the basis states.
    The squared magnitudes of the amplitudes sum to 1, representing the probabilities
    of measuring the state in each basis state.
    """

    var num_qubits: Int
    """The number of qubits in the state, which determines the size of the state vector."""
    var state_vector: CustomList[ComplexFloat32, hint_trivial_type=True]
    """The state vector representing the amplitudes of the basis states."""

    fn __init__(out self, size: Int):
        """Initializes a StateVector with the given size.

        Args:
            size: The size of the vector, which is 2^n for n qubits.
        """
        self.num_qubits = 0
        self.state_vector = CustomList[ComplexFloat32, hint_trivial_type=True](
            length=size, fill=ComplexFloat32(0.0, 0.0)
        )
        # self.state_vector.memset_zero()

    @always_inline
    fn __getitem__(self, index: Int) -> ComplexFloat32:
        # @parameter
        # if check_bounds:
        #     if index < 0 or index >= self.size():
        #         print("ERROR: Index", index, "is out of bounds for state vector of size", self.size())
        #         return ComplexFloat32(0.0, 0.0)
        # else:
        return self.state_vector[index]

    @always_inline
    fn __setitem__(mut self, index: Int, value: ComplexFloat32) -> None:
        self.state_vector[index] = value

    fn __str__(self) -> String:
        """Returns a beautifully formatted string representation of the StateVector.
        """
        string: String = "StateVector:\n"
        for i in range(self.size()):
            amplitude = self.state_vector[i]
            # amplitude_str: String = String(amplitude)
            amplitude_re: Float32 = amplitude.re
            amplitude_im: Float32 = amplitude.im
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
        """Returns a StateVector corresponding to the given bitstring.

        Params:
            bitstring: A string of '0's and '1's representing the state, with
                        the least significant qubit (top one) (LSB) at the start.

        Examples:
        ```mojo
        state = StateVector.from_bitstring("110")
        ```

        Returns:
            A StateVector object with the appropriate data initialized.
        """
        num_qubits: Int = len(bitstring)

        state_vector = CustomList[ComplexFloat32, hint_trivial_type=True](
            length=1 << num_qubits, fill=ComplexFloat32(0.0, 0.0)
        )  # 2^num_qubits
        state_vector.memset_zero()  # Initialize the state vector with zeros

        # Put coefficent correspondin to the bitstring to 1
        index: Int = 0
        i: Int = 0
        for bit in bitstring.codepoints():
            if bit == Codepoint.ord("1"):
                index |= 1 << i  # Set the bit at position i
            i += 1

        state_vector[index] = ComplexFloat32(
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
        #     self.state_vector[i] = ComplexFloat32(0.0, 0.0)  # Set each amplitude to zero

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
        squared_norm: Float32 = 0.0
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
            A new StateVector with the conjugated amplitudes.
        """
        conjugated_state: Self = Self(self.num_qubits, self.state_vector.copy())

        for i in range(self.size()):
            conjugated_state.state_vector[i] = self.state_vector[i].conjugate()
        return conjugated_state

    # @always_inline
    # fn tensor_product()

    @always_inline
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

    # fn partial_trace[
    #     use_lookup_table: Bool = True
    # ](quantum_state: StateVector, qubits_to_trace_out: List[Int],) -> ComplexMatrix:

    # qubits_to_trace_out: An array of indices of qubits to trace out, in ascending order
    #                         and without duplicates.

    @always_inline
    fn purity(self, arg_qubits_to_keep: LinkedList[Int] = []) -> Float32:
        """Calculates the purity of the pure state.

        The purity is defined as the trace of the density matrix squared.

        Args:
            arg_qubits_to_keep: A list of qubit indices to keep in the state in stricly ascending order.
                                If empty, all qubits are kept.

        Returns:
            The purity of the pure state, which should be 1 for a valid pure state.
        """
        qubits_to_keep = arg_qubits_to_keep.copy()

        # Sanity check for ascending order and in range
        for i in range(len(qubits_to_keep)):
            qubit: Int = qubits_to_keep[i]
            if qubit < 0 or qubit >= self.num_qubits:
                print(
                    "Error: Qubit index",
                    qubit,
                    "is out of range for the number of qubits",
                    self.num_qubits,
                )
                return 0.0
            if i > 0 and qubits_to_keep[i - 1] >= qubit:
                print(
                    (
                        "Error: Qubit indices must be in stricly ascending"
                        " order. Found "
                    ),
                    qubits_to_keep[i - 1],
                    "and",
                    qubit,
                )
                return 0.0

        # If no qubits to keep, keep all
        if len(qubits_to_keep) == 0:
            for i in range(self.num_qubits):
                qubits_to_keep.append(i)

        qubits_to_trace_out: List[Int] = []
        current_qubit: Int = 0
        for i in range(self.num_qubits):
            if (
                current_qubit < len(qubits_to_keep)
                and qubits_to_keep[current_qubit] == i
            ):
                current_qubit += 1  # This qubit is kept, so skip it
            else:
                qubits_to_trace_out.append(i)  # This qubit is traced out

        density_matrix: ComplexMatrix = partial_trace(self, qubits_to_trace_out)

        # density_matrix = self.to_density_matrix()
        trace_squared: Float32 = 0.0

        for i in range(density_matrix.size()):
            trace_squared += density_matrix[i, i].squared_norm()

        return trace_squared

    @always_inline
    fn normalised_purity(self, qubits_to_keep: LinkedList[Int] = []) -> Float32:
        """Calculates the normalised purity of the pure state.

        For a density matrix of size 2n×2n, purity ranges from 1/2^n to 1.
        The normalised purity is defined as the purity divided by the number of qubits.

        Returns:
            The normalised purity of the pure state.
        """
        return (self.size() * self.purity(qubits_to_keep) - 1) / (
            self.size() - 1
        )

    @always_inline
    fn linear_entropy(self, qubits_to_keep: LinkedList[Int] = []) -> Float32:
        """Calculates the linear entropy of the pure state.

        The linear entropy is defined as 1 - purity.

        Returns:
            The linear entropy of the pure state.
        """
        return 1.0 - self.purity(qubits_to_keep)


@fieldwise_init
struct ComplexMatrix(Copyable, Movable, Stringable, Writable):
    """Represents a 2D matrix of complex numbers.

    This is used to represent quantum gates in the form of matrices.
    """

    var matrix: List[List[ComplexFloat32]]
    """The 2D matrix representation of the quantum gate."""

    fn __init__(out self, rows: Int, cols: Int):
        """Initializes a ComplexMatrix with the given number of rows and columns.

        Args:
            rows: The number of rows in the matrix.
            cols: The number of columns in the matrix.
        """
        self.matrix = List[List[ComplexFloat32]](
            length=rows,
            fill=List[ComplexFloat32](
                length=cols, fill=ComplexFloat32(0.0, 0.0)
            ),
        )
        # # Initialize the matrix with zeros
        # for i in range(rows):
        #     for j in range(cols):
        #         self.matrix[i][j] = ComplexFloat32(0.0, 0.0)

    @always_inline
    fn __getitem__(self, row: Int, col: Int) -> ComplexFloat32:
        return self.matrix[row][col]

    @always_inline
    fn __setitem__(mut self, row: Int, col: Int, value: ComplexFloat32) -> None:
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
    fn mult(self, other: StateVector, mut buffer: StateVector) -> None:
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
