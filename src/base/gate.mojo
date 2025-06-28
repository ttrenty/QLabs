# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Imports              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

from math import sqrt

from ..local_stdlib import CustomList
from ..local_stdlib.complex import ComplexFloat32

from .state_and_matrix import (
    ComplexMatrix,
)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Aliases              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

alias Identity = Gate(
    size=2,
    matrix=ComplexMatrix(
        List[List[ComplexFloat32]](
            [ComplexFloat32(1, 0), ComplexFloat32(0, 0)],
            [ComplexFloat32(0, 0), ComplexFloat32(1, 0)],
        )
    ),
    symbol="I",
)

alias Hadamard = Gate(
    size=2,
    matrix=ComplexMatrix(
        List[List[ComplexFloat32]](
            [
                ComplexFloat32(1.0 / Float32(sqrt(2.0)), 0),
                ComplexFloat32(1 / Float32(sqrt(2.0)), 0),
            ],
            [
                ComplexFloat32(1 / Float32(sqrt(2.0)), 0),
                ComplexFloat32(-1 / Float32(sqrt(2.0)), 0),
            ],
        )
    ),
    symbol="H",
)

alias PauliX = Gate(
    size=2,
    matrix=ComplexMatrix(
        List[List[ComplexFloat32]](
            [ComplexFloat32(0, 0), ComplexFloat32(1, 0)],
            [ComplexFloat32(1, 0), ComplexFloat32(0, 0)],
        )
    ),
    symbol="X",
)

alias PauliZ = Gate(
    size=2,
    matrix=ComplexMatrix(
        List[List[ComplexFloat32]](
            [ComplexFloat32(1, 0), ComplexFloat32(0, 0)],
            [ComplexFloat32(0, 0), ComplexFloat32(-1, 0)],
        )
    ),
    symbol="Z",
)

alias PauliY = Gate(
    size=2,
    matrix=ComplexMatrix(
        List[List[ComplexFloat32]](
            [ComplexFloat32(0, 0), ComplexFloat32(0, -1)],
            [ComplexFloat32(0, 1), ComplexFloat32(0, 0)],
        )
    ),
    symbol="Y",
)

alias _SEPARATOR = Gate(
    size=2,
    matrix=ComplexMatrix(
        List[List[ComplexFloat32]](
            [ComplexFloat32(0, 0), ComplexFloat32(0, -1)],
            [ComplexFloat32(0, 1), ComplexFloat32(0, 0)],
        )
    ),
    symbol="_SEPARATOR",
)

alias _START = Gate(
    size=2,
    matrix=ComplexMatrix(
        List[List[ComplexFloat32]](
            [ComplexFloat32(0, 0), ComplexFloat32(0, -1)],
            [ComplexFloat32(0, 1), ComplexFloat32(0, 0)],
        )
    ),
    symbol="_START",
)

alias SWAP = Gate(
    size=2,
    matrix=ComplexMatrix(
        List[List[ComplexFloat32]](
            [ComplexFloat32(0, 0), ComplexFloat32(0, -1)],
            [ComplexFloat32(0, 1), ComplexFloat32(0, 0)],
        )
    ),
    symbol="SWAP",
)

alias iSWAP = Gate(
    size=4,
    matrix=ComplexMatrix(
        List[List[ComplexFloat32]](
            [
                ComplexFloat32(1, 0),
                ComplexFloat32(0, 0),
                ComplexFloat32(0, 0),
                ComplexFloat32(0, 0),
            ],
            [
                ComplexFloat32(0, 0),
                ComplexFloat32(0, 0),
                ComplexFloat32(0, 1),
                ComplexFloat32(0, 0),
            ],
            [
                ComplexFloat32(0, 0),
                ComplexFloat32(0, 1),
                ComplexFloat32(0, 0),
                ComplexFloat32(0, 0),
            ],
            [
                ComplexFloat32(0, 0),
                ComplexFloat32(0, 0),
                ComplexFloat32(0, 0),
                ComplexFloat32(1, 0),
            ],
        )
    ),
    symbol="iSWAP",
)

alias X = PauliX
alias Z = PauliZ
alias Y = PauliY
alias H = Hadamard

alias NOT = PauliX


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Structs              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


struct Gate(Copyable, Movable, Representable, Stringable, Writable):
    """Represents an abritrary quantum gate as a matrix of complex numbers.

    The gate is represented as a 2^n x 2^n matrix, where n is the number of qubits the
    gate acts on.
    """

    var size: Int
    """The size of the gate matrix, which is 2^n for n qubits."""

    var matrix: ComplexMatrix
    """The matrix representation of the quantum gate, a 2D matrix of complex numbers."""

    var symbol: String
    """A string symbol representing the gate, e.g., 'H' for Hadamard, 'X' for Pauli-X."""

    var target_qubits: List[Int]
    """A list of target qubits that the gate acts on. Each index corresponds to a qubit."""

    var control_qubits_with_flags: List[List[Int]]
    """A list of control qubits in the format [qubit_index, flag] where flag=1 for 
    control, flag=0 for anti-control.
    """

    @always_inline
    fn __init__(
        out self,
        size: Int,
        matrix: ComplexMatrix,
        symbol: String,
        target_qubits: List[Int] = [],
        control_qubits_with_flags: List[List[Int]] = [],
    ):
        """Initializes a Gate with the given parameters.

        Args:
            size: The size of the gate matrix, which is 2^n for n qubits.
            matrix: The 2D matrix representation of the quantum gate.
            symbol: A string symbol representing the gate, e.g., 'H' for Hadamard, 'X' for Pauli-X.
            target_qubits: A list of target qubits that the gate acts on.
            control_qubits_with_flags: A list of control qubits in the format [qubit_index, flag]
                where flag=1 for control, flag=0 for anti-control.
        """
        self.size = size
        self.matrix = matrix
        self.symbol = symbol
        self.target_qubits = target_qubits
        self.control_qubits_with_flags = control_qubits_with_flags

    @always_inline
    fn __call__(
        self,
        *target_qubits: Int,
        controls: List[Int] = [],
        anti_controls: List[Int] = [],
    ) -> Self:
        """Returns a new Gate instance with the same matrix but with specified target and control qubits.

        Args:
            target_qubits: A list of target qubits that the gate acts on.
            controls: A list of control qubit indices.
            anti_controls: A list of anti-control qubit indices.

        Returns:
            A new `Gate` instance with the specified parameters.
        """
        # Check if there is no conflict between control and anti-control qubits
        control_qubits_to_remove: List[Int] = []
        anti_control_qubits_to_remove: List[Int] = []
        for i in range(len(controls)):
            for j in range(len(anti_controls)):
                if controls[i] == anti_controls[j]:
                    print(
                        (
                            "Error: Control and anti-control qubits cannot be"
                            " the same. "
                        ),
                        "Qubit:",
                        controls[i],
                        ". Defaulting to no control qubits.",
                    )
                    control_qubits_to_remove.append(controls[i])
                    anti_control_qubits_to_remove.append(anti_controls[j])
        new_control_qubits: List[Int] = []
        new_anti_control_qubits: List[Int] = []
        for control in controls:
            if control not in control_qubits_to_remove:
                new_control_qubits.append(control)
        for anti_control in anti_controls:
            if anti_control not in anti_control_qubits_to_remove:
                new_anti_control_qubits.append(anti_control)
        target_qubits_list: List[Int] = []
        for target_qubit in target_qubits:
            target_qubits_list.append(target_qubit)
        return Self(
            self.size,
            self.matrix,
            self.symbol,
            target_qubits_list,
            [[qubit, 1] for qubit in new_control_qubits]
            + [[qubit, 0] for qubit in new_anti_control_qubits],
        )

    @always_inline
    fn __getitem__(self, row: Int, col: Int) -> ComplexFloat32:
        return self.matrix[row, col]

    fn __str__(self) -> String:
        """Returns the symbol of the gate as its string representation."""
        return self.symbol

    fn __repr__(self) -> String:
        """Convert the Gate to a string representation.

        Returns:
            A string representation of the Gate, including its symbol, and qubits it
            acts on.
        """
        target_qubits_str: String = "["
        for i in range(len(self.target_qubits)):
            target_qubits_str += String(self.target_qubits[i])
            if i < len(self.target_qubits) - 1:
                target_qubits_str += ", "
        target_qubits_str += "]"
        control_qubits_str: String = "["
        for i in range(len(self.control_qubits_with_flags)):
            control_qubits_str += (
                String(self.control_qubits_with_flags[i][0])
                + ":"
                + String(self.control_qubits_with_flags[i][1])
            )
            if i < len(self.control_qubits_with_flags) - 1:
                control_qubits_str += ", "
        control_qubits_str += "]"

        return String(
            "Gate(symbol="
            + self.symbol
            + ", target_qubits="
            + target_qubits_str
            + ", control_qubits_with_flags="
            + control_qubits_str
            + ")"
        )

    fn write_to[W: Writer](self, mut writer: W) -> None:
        writer.write(String(self))
