# """
#     Mojo Implementation of Reference:

#     title: How to Write a Simulator for Quantum Circuits from Scratch: A Tutorial
#     authors: Michael J. McGuffin, Jean-Marc Robert, and Kazuki Ikeda
#     date: 2025-06-09
#     abstract: This tutorial guides a competent programmer through the crafting of a
#     quantum circuit simulator from scratch, even for readers with almost no prior
#     experience in quantum computing. Open source simulators for quantum circuits already
#     exist, but a deeper understanding is gained by writing ones own. With roughly
#     1000-2000 lines of code, one can simulate Hadamard, Pauli X, Y, Z, SWAP, and other
#     quantum logic gates, with arbitrary combinations of control and anticontrol qubits,
#     on circuits of up to 20+ qubits, with no special libraries, on a personal computer.
#     for updating the state vector, and partial trace for finding a reduced density
#     matrix. We also discuss optimizations, and how to compute qubit phase, purity, and
#     other statistics. A complete example implementation in JavaScript is available at
#     https://github.com/MJMcGuffin/muqcs.js , which also demonstrates how to compute von
#     Neumann entropy, concurrence (to quantify entanglement), and magic, while remaining
#     much smaller and easier to study than other popular software packages.
#     url: https://arxiv.org/abs/2506.08142v1 (last accessed: 2025-06-12)
# """

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Conventions          #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# """
#     Qubits are indexed from 0 to n-1, top to bottom.
#     The top qubit is the least significant bit (LSB), and the bottom is the most significant bit (MSB),
#     This leads to states being written in the order |q(n-1) ... q(1) q(0)⟩.
#     This also means that the tensor products of gates applied to the circuit are 
#     computed from right to left, for example with a PauliX gate on q0, the layer of 
#     gates would be written as I ⊗ I ⊗ ... ⊗ I ⊗ X, where I is the identity gate of 
#     if size 2x2, and there are as many I gates as there are qubits-1 in the circuit.
#     This convention is often times called little-endian which is the convention of 
#     many quantum computing frameworks, but is the opposite of the convention used in 
#     quantum physics.
# """

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Imports              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# from complex import ComplexFloat64
from local_complex import ComplexFloat64

from collections.linked_list import LinkedList

import random

from sys import argv

from local_list import CustomList

from bit import count_trailing_zeros

# << for shift left
# | for bitwise OR
# ^ for bitwise XOR
# & for bitwise AND
# ~ for bitwise NOT


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Aliases              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

alias Hadamard = Gate(
    size=2,
    matrix=ComplexMatrix(List[List[ComplexFloat64]](
        [ComplexFloat64(1 / 2**0.5, 0), ComplexFloat64(1 / 2**0.5, 0)],
        [ComplexFloat64(1 / 2**0.5, 0), ComplexFloat64(-1 / 2**0.5, 0)],
    )),
    symbol="H",
)

alias PauliX = Gate(
    size=2,
    matrix=ComplexMatrix(List[List[ComplexFloat64]](
        [ComplexFloat64(0, 0), ComplexFloat64(1, 0)],
        [ComplexFloat64(1, 0), ComplexFloat64(0, 0)],
    )),
    symbol="X",
)

alias PauliZ = Gate(
    size=2,
    matrix=ComplexMatrix(List[List[ComplexFloat64]](
        [ComplexFloat64(1, 0), ComplexFloat64(0, 0)],
        [ComplexFloat64(0, 0), ComplexFloat64(-1, 0)],
    )),
    symbol="Z",
)

alias PauliY = Gate(
    size=2,
    matrix=ComplexMatrix(List[List[ComplexFloat64]](
        [ComplexFloat64(0, 0), ComplexFloat64(0, -1)],
        [ComplexFloat64(0, 1), ComplexFloat64(0, 0)],
    )),
    symbol="Y",
)

alias _SEPARATOR = Gate(
    size=2,
    matrix=ComplexMatrix(List[List[ComplexFloat64]](
        [ComplexFloat64(0, 0), ComplexFloat64(0, -1)],
        [ComplexFloat64(0, 1), ComplexFloat64(0, 0)],
    )),
    symbol="_SEPARATOR",
)

alias _START = Gate(
    size=2,
    matrix=ComplexMatrix(List[List[ComplexFloat64]](
        [ComplexFloat64(0, 0), ComplexFloat64(0, -1)],
        [ComplexFloat64(0, 1), ComplexFloat64(0, 0)],
    )),
    symbol="_START",
)

alias SWAP = Gate(
    size=2,
    matrix=ComplexMatrix(List[List[ComplexFloat64]](
        [ComplexFloat64(0, 0), ComplexFloat64(0, -1)],
        [ComplexFloat64(0, 1), ComplexFloat64(0, 0)],
    )),
    symbol="SWAP",
)

alias iSWAP = Gate(
    size=4,
    matrix=ComplexMatrix(List[List[ComplexFloat64]](
        [ComplexFloat64(1, 0), ComplexFloat64(0, 0), ComplexFloat64(0, 0), ComplexFloat64(0, 0)],
        [ComplexFloat64(0, 0), ComplexFloat64(0, 0), ComplexFloat64(0, 1), ComplexFloat64(0, 0)],
        [ComplexFloat64(0, 0), ComplexFloat64(0, 1), ComplexFloat64(0, 0), ComplexFloat64(0, 0)],
        [ComplexFloat64(0, 0), ComplexFloat64(0, 0), ComplexFloat64(0, 0), ComplexFloat64(1, 0)],
    )),
    symbol="iSWAP",
)

alias X = PauliX
alias Z = PauliZ
alias Y = PauliY
alias H = Hadamard

alias NOT = PauliX 


alias ShowOnlyEnd = "ShowOnlyEnd"
alias ShowAfterEachLayer = "ShowAfterEachLayer"
alias ShowAfterEachGate = "ShowAfterEachGate"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Structs              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

@fieldwise_init
struct PureBasisState[check_bounds:Bool = False](Copyable, Movable, Stringable, Writable):
# struct PureBasisState(Copyable, Movable, Stringable, Writable):
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
        self.state_vector = CustomList[ComplexFloat64, hint_trivial_type=True](length=size, fill=ComplexFloat64(0.0, 0.0))
        self.state_vector.memset_zero()  # Initialize the state vector with zeros

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
                amplitude_str = String(round(amplitude_re, 2))
            elif amplitude_re == 0.0:
                amplitude_str = String(round(amplitude_im, 2)) + "i"
            else:
                amplitude_str = String(round(amplitude_re, 2)) + " + " + String(round(amplitude_im, 2)) + "i"
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

        Example:
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

        state_vector = CustomList[ComplexFloat64, hint_trivial_type=True](length=1 << num_qubits, fill=ComplexFloat64(0.0, 0.0)) # 2^num_qubits
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
        self.state_vector.memset_zero() # Set all elements to zero
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
        return abs(squared_norm - 1.0) < 1e-6  # Allow a small tolerance for floating-point errors

@fieldwise_init
struct ComplexMatrix(Copyable, Movable, Stringable, Writable):
    """Represents a 2D matrix of complex numbers.

    This is used to represent quantum gates in the form of matrices.
    """

    var matrix: List[List[ComplexFloat64]]
    """The 2D matrix representation of the quantum gate."""

    @always_inline
    fn __getitem__(self, row: Int, col: Int) -> ComplexFloat64:
        return self.matrix[row][col]

    fn __str__(self) -> String:
        """Return a beautifully formatted string representation of the ComplexMatrix.
        Values that are 0 are omitted, and the matrix is printed in a human-readable format.
        """
        size = len(self.matrix)
        string = "ComplexMatrix: (size: " + String(size) + "x" + String(size) + ")\n"
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
            print("Error: Matrix and vector sizes do not match for multiplication.")
            return
        
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                buffer[i] += self.matrix[i][j] * other[j]

    fn size(self) -> Int:
        """Returns the size of the matrix, which is the number of rows (or columns)."""
        return len(self.matrix)
    

struct Gate(Copyable, Movable, Stringable, Writable, Representable):
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
    fn __call__(self, *target_qubits: Int, controls: List[Int] = [], anti_controls: List[Int] = []) -> Self:
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
                    print("Error: Control and anti-control qubits cannot be the same. ",
                      "Qubit:", controls[i],
                      ". Defaulting to no control qubits.")
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
            [[qubit, 1] for qubit in new_control_qubits] + [[qubit, 0] for qubit in new_anti_control_qubits],
        )

    @always_inline
    fn __getitem__(self, row: Int, col: Int) -> ComplexFloat64:
        return self.matrix[row, col]

    fn __str__(self) -> String:
        """Returns the symbol of the gate as its string representation.
        """
        return self.symbol

    fn __repr__(self) -> String:
        '''Convert the Gate to a string representation.
        
        Returns:
            A string representation of the Gate, including its symbol, and qubits it 
            acts on.
        '''
        target_qubits_str: String = "["
        for i in range(len(self.target_qubits)):
            target_qubits_str += String(self.target_qubits[i])
            if i < len(self.target_qubits) - 1:
                target_qubits_str += ", "
        target_qubits_str += "]"
        control_qubits_str: String = "["
        for i in range(len(self.control_qubits_with_flags)):
            control_qubits_str += String(self.control_qubits_with_flags[i][0]) + ":" + String(self.control_qubits_with_flags[i][1])
            if i < len(self.control_qubits_with_flags) - 1:
                control_qubits_str += ", "
        control_qubits_str += "]"

        return String(
            "Gate(symbol=" + self.symbol + ", target_qubits=" + target_qubits_str + 
            ", control_qubits_with_flags=" + control_qubits_str + ")")

    fn write_to[W: Writer](self, mut writer: W) -> None:
        writer.write(String(self))
   
# struct GateCircuit(Copyable, Movable, Stringable, Writable):
struct GateCircuit(Movable, Stringable, Writable):
    """Represents a quantum circuit consisting of gates applied to qubits.

    This struct allows for the construction, manipulation, and simulation
    of quantum circuits composed of various quantum gates.
    """

    var num_qubits: Int
    """The number of qubits in the circuit."""
    var gates: LinkedList[Gate]
    """The list of gates in the circuit. Each entry is a tuple: (gate, target_qubits, control_qubits_with_flags).
    control_qubits_with_flags is a list of [qubit_index, flag], where flag=1 for control, flag=0 for anti-control.
    """

    @always_inline
    fn __init__(
        out self,
        num_qubits: Int,
        gates: LinkedList[Gate] = [],
    ):
        """Initializes a GateCircuit with the given parameters.

        Args:
            num_qubits: The number of qubits in the circuit.
            gates: Optional list of initial gates in the circuit. Defaults to an empty list.
        """
        self.num_qubits = num_qubits
        self.gates = gates

    @always_inline
    fn __copyinit__(out self, other: Self):
        """Copy initialize the GateCircuit from another instance.

        Args:
            other: The GateCircuit instance to copy from.
        """
        self.num_qubits = other.num_qubits
        self.gates = other.gates.copy()  # Copy the linked list of gates

    fn __str__(self) -> String:
         """Returns a string representation of the GateCircuit.

        This representation includes the number of qubits, initial state,
        total number of gates, and a textual diagram of the circuit.

        Returns:
            A string detailing the circuit's configuration and structure.
        """
        # string: String = "GateCircuit:\n"
        # string += String("Number of qubits: {}\n").format(self.num_qubits)
        # string += String("Initial state: {}\n").format(self.initial_state)
        # string += "Gates:\n"
        # print("TODO")
        string: String = "GateCircuit:\n"
        # print the circuit in a human-readable format such as:
# -------|X|--|Z|--
#         |       
# --|H|---*----*---
#              |    
# --|X|-------|X|--
# # OR 

# --|H|---x--------|X|--|Y|---*-------
#             |         |         |
# --------|---|X|---o---------x--|Z|--
#             |    |              |
# --------x----o--------------x-------
        # seperate each gates layers with "--", than if there is a gate on that wire use
        # |symbol|, if not use "---", then add another "--", and repeat for all gates and wires,
        # add inbetween lines with the "|" symbol for control wires, use * for control wires
        # and o for anti-control wires. don't forget to include the state of qubits at the start.

        wires: List[String] = [String("--")] * self.num_qubits  # Initialize wires for each qubit
        in_between: List[String] = [String("  ")] * (self.num_qubits - 1)  # Initialize control wires in-between qubits
        wires_current_gate: List[Int] = [0] * self.num_qubits  # Track current gate on each wire
        for gate in self.gates:
            if gate.symbol == _START.symbol:  # If it's a start gate, initialize the wires
                continue
            if gate.symbol == _SEPARATOR.symbol:  # If it's a separator, add a new layer
                for i in range(self.num_qubits-1):
                    wires[i] += String("|")  
                    in_between[i] += String("|")
                wires[self.num_qubits - 1] += String("|")
            else:
                qubit_index: Int = gate.target_qubits[0]  # Assuming single target qubit for simplicity
                controls_flags: List[List[Int]] = gate.control_qubits_with_flags
                
                for control in controls_flags:
                    control_index, flag = control[0], control[1]
                    while wires_current_gate[qubit_index] < wires_current_gate[control_index]:
                        wires[qubit_index] += String("-----")  
                        # in_between[qubit_index] += String("     ")  
                        wires_current_gate[qubit_index] += 1  # Increment the current gate on this wire

                wires[qubit_index] += String("|") + gate.symbol + String("|")  # Add the gate symbol to the wire
                wires[qubit_index] += String("--")  # Add a separator after the gate  
                
                wires_current_gate[qubit_index] += 1  # Increment the current gate on this wire

                if len(controls_flags) > 0:
                    for control in controls_flags:
                        control_index, flag = control[0], control[1]
                        if flag == 1:  # Control qubit
                            wires[control_index] += String("-*-") 
                        else:  # Anti-control qubit
                            wires[control_index] += String("-o-") 
                        
                        while wires_current_gate[control_index] < wires_current_gate[qubit_index]:
                            wires[control_index] += String("--") 
                            wires_current_gate[control_index] += 1  # Increment the current gate on this wire


        for i in range(self.num_qubits):
            string += wires[i] + "\n"  # Print each wire
            if i < self.num_qubits - 1:
                string += in_between[i] + "\n"  # Print the in-between control wires
        
        return string


    fn write_to[W: Writer](self, mut writer: W) -> None:
        writer.write(String(self))

    @always_inline
    fn apply(
        mut self,
        gate: Gate,
    ) -> None:
        """Applies a quantum gate to the circuit and returns a new GateCircuit instance.

        The gate was defined with the target qubits and control qubits already set.
        """
        self.gates.append((gate))
    
    @always_inline
    fn apply_gates(
        mut self,
        *gates: Gate,
    ) -> None:
        """Applies a quantum gate to the circuit and returns a new GateCircuit instance.

        The gate was defined with the target qubits and control qubits already set.
        """
        for ref gate in gates:
            self.apply(gate)

    @always_inline
    fn barrier(mut self) -> None:
        """Adds a barrier to the circuit and returns a new GateCircuit instance.

        Barriers are typically used for visualization or to prevent gate reordering
        across them in optimization passes, though their exact function here is
        to mark a separation.
        """
        self.apply(_SEPARATOR)  # Use the _SEPARATOR gate to mark a barrier

    @always_inline
    fn apply_layer(mut self, *gates: Gate) -> None:
        """Adds a layer of gates to the circuit and returns a new GateCircuit instance.

        This method allows for adding multiple gates in a single layer, which can
        be useful for constructing complex circuits in a more structured way.

        Args:
            gates: A variable number of Gate instances to be added as a layer.
        """
        for ref gate in gates:
            self.apply(gate)
        self.barrier()

    @always_inline
    fn num_gates(self) -> Int:
        """Returns the total number of gates in the circuit.

        Returns:
            The total count of gate operations added to the circuit.
        """
        return len(self.gates)

struct StateVectorSimulator(Movable, Copyable):
    """A noiseless simulator for quantum circuits that uses State Vector Simulation.

    This simulator applies quantum gates to a state vector representing the quantum state.
    """

    var circuit: GateCircuit
    """The quantum circuit containing the gates to be applied."""
    var original_circuit: GateCircuit
    """The original circuit before any modifications, used for resetting the simulator."""
    var initial_state: PureBasisState
    """The initial state of the quantum system before any gates are applied."""
    var original_initial_state: PureBasisState
    """The original initial state before any modifications, used for resetting the simulator."""
    var optimisation_level: Int
    """The level of optimisation to apply during simulation, affecting performance and accuracy."""
    var verbose: Bool
    """Whether to print verbose output during simulation steps."""
    var verbose_step_size: String
    """The verbosity level for simulation output, controlling how often state updates are printed."""


    @always_inline
    fn __init__(
        out self,
        owned circuit: GateCircuit,
        initial_state: PureBasisState = PureBasisState.from_bitstring("0"),
        optimisation_level: Int = 0,
        verbose: Bool = False,
        verbose_step_size: String = "ShowOnlyEnd",
        ):
        """Initializes a StateVectorSimulator with the given parameters.

        Args:
            circuit: The quantum circuit containing the gates to be applied.
            initial_state: The initial state of the quantum system.
            optimisation_level: The level of optimisation to apply during simulation.
            verbose: Whether to print verbose output during simulation steps.
            verbose_step_size: The verbosity level for simulation output.
        """
        new_initial_state = initial_state
        if initial_state.size() != circuit.num_qubits:
            new_initial_state = PureBasisState.from_bitstring("0" * circuit.num_qubits) # Ensure initial state matches the number of qubits
        self.circuit = circuit
        self.original_circuit = circuit
        self.initial_state = new_initial_state
        self.original_initial_state = new_initial_state
        self.optimisation_level = optimisation_level
        self.verbose = verbose
        self.verbose_step_size = verbose_step_size

    @always_inline
    fn next_gate(
        mut self,
        owned quantum_state: PureBasisState,
    ) -> PureBasisState:
        """Applies the next gate in the circuit to the quantum state.

        Args:
            quantum_state: The current state of the quantum system.

        Returns:
            A tuple containing the updated StateVectorSimulator and the new quantum state.
        """

        gates = self.circuit.gates
        if len(gates) == 0:
            return quantum_state
        gate = _START         
        while gate.symbol in [_SEPARATOR.symbol, _START.symbol] and len(gates) > 0:
            try:
                gate = gates.pop(0)  # Get the next gate, skipping any layer separators
            except e:
                print("Error: No gates left to apply. Returning current state.")
                return quantum_state  # No gates left to apply

        new_quantum_state = qubitWiseMultiply(
            gate.matrix,
            gate.target_qubits[0],
            quantum_state,
            gate.control_qubits_with_flags,
        )

        if self.verbose and self.verbose_step_size == ShowAfterEachGate:
            print("Applied gate:", repr(gate))
            print("New quantum state:\n", new_quantum_state)

        self.circuit.gates = gates

        return new_quantum_state

    fn reset(mut self) -> None:
        """Resets the simulator to its initial state and circuit.
        """
        self.circuit = self.original_circuit  # Reset circuit to original
        self.initial_state = self.original_initial_state  # Reset initial state
        
    fn next_layer(
        self,
        quantum_state: PureBasisState,
    ) -> (Self, PureBasisState):
        """Applies the next layer of gates in the circuit to the quantum state.

        Args:
            quantum_state: The current state of the quantum system.

        Returns:
            A tuple containing the updated StateVectorSimulator and the new quantum state.
        """
        circuit = self.circuit

        gates = circuit.gates
        if len(gates) == 0:
            return (self, quantum_state)  # No gates left to apply

        new_quantum_state: PureBasisState = quantum_state  # Start with the current state
        
        gate = _START
        while len(gates) > 0 and gate.symbol != _SEPARATOR.symbol:
            try:
                gate = gates.pop(0)
            except e:
                print("Error: No gates left to apply. Returning current state.")
                return (self, quantum_state)  # No gates left to apply
            
            new_quantum_state = qubitWiseMultiply(
                gate.matrix,
                gate.target_qubits[0],  # Assuming single target qubit
                quantum_state,
                gate.control_qubits_with_flags,
            )

        if self.verbose and self.verbose_step_size == ShowAfterEachLayer:
            print("Applied layer of gates.")
            print("New quantum state:\n", new_quantum_state)
        
        circuit.gates = gates  # Update the circuit with remaining gates
        return (
            Self(
                circuit,
                new_quantum_state,
                self.optimisation_level,
                self.verbose,
                self.verbose_step_size,
            ),
            new_quantum_state,
        )

    fn next_block(
        self,
        quantum_state: PureBasisState,
    ) -> (Self, PureBasisState):
       return self.next_layer(quantum_state)  # For now, treat blocks as layers

    fn run(self) -> PureBasisState:
        """Runs the quantum circuit simulation.

        Applies all gates in sequence to the initial state and computes the
        final quantum state. Will print verbose output if enabled and at 
        the specified verbosity level.

        Returns:
            The final `PureBasisState` after all gates have been applied.
        """
        if self.verbose:
            print(
                "Running quantum circuit simulation with verbose step size:",
                self.verbose_step_size,
            )
            print("Initial state:\n", self.initial_state)

        # Start with the initial state
        quantum_state: PureBasisState = self.initial_state
        i: Int = 0
        layer_index: Int = 0
        for gate in self.circuit.gates:  # Iterate over the gates in the circuit

            if gate.symbol not in [_SEPARATOR.symbol, SWAP.symbol]:
                # Apply the next gate
                quantum_state = qubitWiseMultiplyExtended(
                    len(gate.target_qubits),  # Number of target qubits
                    gate.matrix,
                    gate.target_qubits,  # Assuming single target qubit
                    quantum_state,
                    gate.control_qubits_with_flags,
                )
            elif gate.symbol == SWAP.symbol:
                if len(gate.target_qubits) != 2:
                    print("Error: SWAP gate must have exactly 2 target qubits.")
                    continue  
                quantum_state = optimised_apply_swap(
                    self.circuit.num_qubits,
                    gate.target_qubits[0],
                    gate.target_qubits[1], 
                    quantum_state,
                    gate.control_qubits_with_flags,
                )  
            elif gate.symbol == _SEPARATOR.symbol:
                continue  
            else:
                print("Error: Unexpected gate symbol:", gate.symbol)
                continue  # Skip unexpected symbols
            
            i += 1
            if self.verbose:
                if self.verbose_step_size == ShowAfterEachGate:
                    print("New quantum state after gate " + String(i) + ":\n", quantum_state)
                elif self.verbose_step_size == ShowAfterEachLayer and gate.symbol == _SEPARATOR.symbol:
                    print("New quantum state after layer " + String(layer_index) + ":\n", quantum_state)
                if gate.symbol == _SEPARATOR.symbol:
                    layer_index += 1  # Increment layer index after a separator

        if self.verbose and self.verbose_step_size == ShowOnlyEnd:
            print("Final quantum state:\n", quantum_state)

        return quantum_state
        

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Functions            #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

@always_inline
fn qubitWiseMultiplyExtended(
    target_qubits_count: Int,
    gate: ComplexMatrix,
    target_qubits: List[Int],
    owned quantum_state: PureBasisState,
    control_bits: List[List[Int]] = [],
) -> PureBasisState:
    """Applies a quantum gate to multiple qubits in the quantum state.

    If the qubits are not adjacent, the function will apply SWAP gates before applying the gate.
    And reverse the SWAP gates after applying the gate.

    Args:
        target_qubits_count: The number of target qubits the gate acts on.
        gate: The 2^n x 2^n matrix representing the quantum gate, where n is the number of target qubits.
        target_qubits: A list of indices of the qubits on which the gate is applied.
        quantum_state: The current state of the quantum system.
        control_bits: A list of control bits, where each bit is represented as 
                    [wire_index, flag]. If flag is 1, it is a control bit; if 0, 
                    it is an anti-control bit.

    Returns:
        A new PureBasisState with the gate applied.
    """
    if target_qubits_count == 1:
        return qubitWiseMultiply(
            gate,
            target_qubits[0],
            quantum_state,
            control_bits,
        )  

    if target_qubits_count < 1:
        print("Error: target_qubits_count must be at least 1.")
        return quantum_state  # No gate to apply

    adjacent_qubits: Bool = True
    for i in range(len(target_qubits) - 1):
        if target_qubits[i] + 1 != target_qubits[i + 1]: # TODO add the instruction to the 
        # user that they have to provide qubits in increasing order.
            adjacent_qubits = False
            break
    if adjacent_qubits:
        return qubitWiseMultiply(
            gate,
            target_qubits[0],
            quantum_state,
            control_bits,
        )
    else:
        # TODO apply swap gates properly to re-organise the qubits
        return quantum_state  

@always_inline
fn apply_to_multi_qubits(mut new_state_vector: PureBasisState, gate: ComplexMatrix, 
    quantum_state: PureBasisState, size_of_state_vector: Int, size_of_block: Int, 
    size_of_half_block: Int, inclusion_mask: Int, desired_value_mask: Int):
    # For Method 1:
    indexes: List[Int] = List[Int](capacity=gate.size())  

    # For Method 2:
    # temp_vector_1 = PureBasisState(size=gate_size)
    # temp_vector_2 = PureBasisState(size=gate_size)

    for block_start in range(0, size_of_state_vector, size_of_block):
        for offset in range(size_of_half_block):
            # Method 1:
            indexes[0] = block_start | offset  # Initialize the first index
            
            for i in range(1, gate.size()):
                indexes[i] = indexes[i - 1] | size_of_half_block

            for i in range(gate.size()):
                for j in range(gate.size()):
                    if i == 0 and j == 0:
                        new_state_vector[indexes[i]] = gate[i, j] * quantum_state[indexes[j]]
                    else:
                        new_state_vector[indexes[i]] += gate[i, j] * quantum_state[indexes[j]]

            # # Method 2:
            # i1: Int = block_start | offset  # faster than, but equivalent to, block_start + offset
            
            # if (i1 & inclusion_mask) != desired_value_mask:
            #     continue  # skip this iteration if the control bits do not match
            
            # i2: Int = i1
            # for i3 in range(temp_vector_1.size()):
            #     temp_vector_1[i3] = quantum_state[i2]
            #     i2 += size_of_half_block 
            #     # i2 |= size_of_half_block 

            # # Multiply the gate matrix with the temporary vector
            # temp_vector_2.fill_zeros()
            # gate.mult(temp_vector_1, temp_vector_2)  

            # # Distribute the results back to the new state vector
            # i2 = i1
            # for i3 in range(temp_vector_1.size()):
            #     new_state_vector[i2] = temp_vector_2[i3]
            #     i2 += size_of_half_block 

@always_inline
fn apply_to_2_qubit(mut new_state_vector: PureBasisState, gate: ComplexMatrix, 
    quantum_state: PureBasisState, size_of_state_vector: Int, size_of_block: Int, 
    size_of_half_block: Int, inclusion_mask: Int, desired_value_mask: Int):
    for block_start in range(0, size_of_state_vector, size_of_block):
        for offset in range(size_of_half_block):

            i1: Int = block_start | offset
            
            if (i1 & inclusion_mask) != desired_value_mask:
                continue 
            
            i2: Int = i1 | size_of_half_block 
            i3: Int = i2 | size_of_half_block 

            new_state_vector[i1] = gate[0, 0] * quantum_state[i1] + gate[0, 1] * quantum_state[i2] + gate[0, 2] * quantum_state[i3]
            new_state_vector[i2] = gate[1, 0] * quantum_state[i1] + gate[1, 1] * quantum_state[i2] + gate[1, 2] * quantum_state[i3]
            new_state_vector[i3] = gate[2, 0] * quantum_state[i1] + gate[2, 1] * quantum_state[i2] + gate[2, 2] * quantum_state[i3]

@always_inline
fn apply_to_1_qubit(mut new_state_vector: PureBasisState, gate: ComplexMatrix, 
    quantum_state: PureBasisState, size_of_state_vector: Int, size_of_block: Int, 
    size_of_half_block: Int, inclusion_mask: Int, desired_value_mask: Int):
    for block_start in range(0, size_of_state_vector, size_of_block):
        for offset in range(size_of_half_block):

            i1: Int = block_start | offset  # faster than, but equivalent to, block_start + offset
            
            if (i1 & inclusion_mask) != desired_value_mask:
                continue  # skip this iteration if the control bits do not match
            
            i2: Int = i1 | size_of_half_block  # equivalent to i1 + size_of_half_block

            new_state_vector[i1] = gate[0, 0] * quantum_state[i1] + gate[0, 1] * quantum_state[i2]
            new_state_vector[i2] = gate[1, 0] * quantum_state[i1] + gate[1, 1] * quantum_state[i2]

fn qubitWiseMultiply(
    gate: ComplexMatrix,
    target_qubit: Int,
    owned quantum_state: PureBasisState,
    control_bits: List[List[Int]] = [],
) -> PureBasisState:
    """Applies a quantum gate to specific qubits in the quantum state.

    It will apply the gate starting from the target qubit assuming that the other 
    qubits that the gate acts on are following the target qubit.

    Args:
        gate: The 2x2 matrix representing the quantum gate.
        target_qubit: The index of the qubit on which the gate is applied.
        quantum_state: The current state of the quantum system.
        control_bits: A list of control bits, where each bit is represented as 
                    [wire_index, flag]. If flag is 1, it is a control bit; if 0, 
                    it is an anti-control bit.

    Returns:
        A new PureBasisState with the gate applied.
    """
    gate_size: Int = gate.size()
    target_qubits_count: Int = count_trailing_zeros(gate_size)
    if (target_qubit < 0) or (target_qubit >= quantum_state.number_qubits()):
        print("Error: target_qubit index out of bounds. Must be between 0 and", quantum_state.number_qubits() - 1)
        print("Skipping gate application.")
        return quantum_state 

    inclusion_mask: Int = 0
    desired_value_mask: Int = 0
    for control in control_bits:
        wire_index, flag = control[0], control[1]
        bit: Int = 1 << wire_index  # efficient way of computing 2^wire_index
        inclusion_mask |= bit  # turn on the bit
        if flag == 1:
            desired_value_mask |= bit  # turn on the bit

    size_of_state_vector: Int = quantum_state.size()
    size_of_half_block: Int = 1 << target_qubit  # 2^target_qubit
    size_of_block: Int = size_of_half_block << target_qubits_count
    new_state_vector: PureBasisState = quantum_state  # copies all amplitudes from quantum_state to new_state_vector

    if target_qubits_count == 1:
        apply_to_1_qubit(new_state_vector, gate, quantum_state, size_of_state_vector, size_of_block, size_of_half_block, inclusion_mask, desired_value_mask)
    elif target_qubits_count == 2:
        apply_to_2_qubit(new_state_vector, gate, quantum_state, size_of_state_vector, size_of_block, size_of_half_block, inclusion_mask, desired_value_mask)
    else:
        apply_to_multi_qubits(new_state_vector, gate, quantum_state, size_of_state_vector, size_of_block, size_of_half_block, inclusion_mask, desired_value_mask)

    return new_state_vector

# fn invert_gate_endian() #TODO

fn swapBits(
    k: Int,
    i: Int,
    j: Int,
) -> Int:
    """Swaps the ith and jth bits of the integer k.

    Args:
        k: The integer whose bits are to be swapped.
        i: The index of the first bit to swap.
        j: The index of the second bit to swap.

    Returns:
        The integer with the specified bits swapped.
    """
    if i == j:
        return k  # No need to swap if both indices are the same

    bit_i: Int = (k >> i) & 1  # Extract the ith bit
    bit_j: Int = (k >> j) & 1  # Extract the jth bit
    if bit_i == bit_j:
        return k  # No need to swap if both bits are the same

    # Create a mask to flip the ith and jth bits
    mask: Int = (1 << i) | (1 << j)
    # Flip the bits using XOR
    return k ^ mask  # XOR with the mask to swap the bits
    

fn applySwap(
    num_qubits: Int, 
    i: Int, 
    j: Int, 
    quantum_state: PureBasisState, 
    control_bits: List[
        List[Int]
    ] = [], 
) -> PureBasisState:
    """Applies a SWAP gate to two specific qubits in the quantum state.
    Control bits can limit the effect of the SWAP to a subset of the amplitudes in |a> (?).

    Args:
        num_qubits: The total number of qubits in the circuit.
        i: The index of the first qubit to swap.
        j: The index of the second qubit to swap.
        quantum_state: The current state of the quantum system.
        control_bits: A list of control bits, where each bit is [wire_index, flag].
                    If flag is 1, it is a control bit; if 0, it is an anti-control bit.

    Returns:
        A new PureBasisState with the SWAP gate applied.
    """

    new_state_vector: PureBasisState = quantum_state  # copies all amplitudes from quantum_state to new_state_vector
    if i == j:
        return new_state_vector  # No need to swap if both indices are the same

    inclusion_mask: Int = 0
    desired_value_mask: Int = 0
    for control in control_bits:
        wire_index, flag = control[0], control[1]
        bit: Int = 1 << wire_index  # efficient way of computing 2^wire_index
        inclusion_mask |= bit  # turn on the bit
        if flag == 1:
            desired_value_mask |= bit  # turn on the bit

    size_of_state_vector: Int = 1 << num_qubits  # 2^num_qubits
    for k in range(size_of_state_vector):
        if (k & inclusion_mask) != desired_value_mask:
            continue  # skip this iteration if the control bits do not match

        # Swap the bits at positions i and j
        swapped_k: Int = swapBits(k, i, j)

        # Update the state vector
        if swapped_k > k:  # this check ensures we don’t swap each pair twice
            # swap amplitudes
            new_state_vector[k] = quantum_state[swapped_k]
            new_state_vector[swapped_k] = quantum_state[k]

    return new_state_vector


fn optimised_apply_swap(
    num_qubits: Int, 
    i: Int, 
    j: Int,  
    quantum_state: PureBasisState,
    control_bits: List[List[Int]] = [],  
) -> PureBasisState:
    """Applies a SWAP gate to two specific qubits in the quantum state.
    Control bits can limit the effect of the SWAP to a subset of the amplitudes in |a> (?).

    Args:
        num_qubits: The total number of qubits in the circuit.
        i: The index of the first qubit to swap.
        j: The index of the second qubit to swap.
        quantum_state: The current state of the quantum system.
        control_bits: A list of control bits, where each bit is [wire_index, flag].
                    If flag is 1, it is a control bit; if 0, it is an anti-control bit.

    Returns:
        A new PureBasisState with the SWAP gate applied.
    """
    new_state_vector: PureBasisState = quantum_state  # copies all amplitudes from quantum_state to new_state_vector
    if i == j:
        return new_state_vector  # No need to swap if both indices are the same

    inclusion_mask: Int = 0
    desired_value_mask: Int = 0
    for control in control_bits:
        wire_index, flag = control[0], control[1]
        bit: Int = 1 << wire_index  # efficient way of computing 2^wire_index
        inclusion_mask |= bit  # turn on the bit
        if flag == 1:
            desired_value_mask |= bit  # turn on the bit

    size_of_state_vector: Int = 1 << num_qubits  # 2^num_qubits
    antimask_i: Int = ~(1 << i)  # mask to clear the ith bit
    mask_j = 1 << j  # mask to set the jth bit
    for k in range(size_of_state_vector):
        if (k & inclusion_mask) != desired_value_mask:
            continue  # skip this iteration if the control bits do not match

        ith_bit: Int = (k >> i) & 1  # Extract the ith bit
        if ith_bit == 1:
            jth_bit: Int = (k >> j) & 1  # Extract the jth bit
            if jth_bit == 0:
                # Turn off bit i and turn on bit j
                new_k: Int = (k & antimask_i) | mask_j
                # Swap the amplitudes
                new_state_vector[k] = quantum_state[new_k]
                new_state_vector[new_k] = quantum_state[k]

    return new_state_vector


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Examples             #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

fn simulate_figure1_circuit() -> None:
    """Simulates the circuit from Figure 1 in the paper.
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
    # Initialize the quantum circuit to the |000⟩ state
    quantum_state: PureBasisState = PureBasisState.from_bitstring("000")

    print("Initial quantum state:\n", quantum_state)

    # Gate 0
    quantum_state = qubitWiseMultiply(Hadamard.matrix, 1, quantum_state)

    print("After Hadamard gate on qubit 1:\n", quantum_state)

    # Gate 1
    quantum_state = qubitWiseMultiply(PauliX.matrix, 2, quantum_state)

    print("After Pauli-X gate on qubit 2:\n", quantum_state)

    # Gate 2
    quantum_state = qubitWiseMultiply(PauliX.matrix, 0, quantum_state, [[1, 1]])

    print(
        "After Pauli-X gate on qubit 0 with control on qubit 1:\n",
        quantum_state,
    )

    # Gate 3
    quantum_state = qubitWiseMultiply(PauliZ.matrix, 0, quantum_state)

    print("After Pauli-Z gate on qubit 0:\n", quantum_state)

    # Gate 4
    quantum_state = qubitWiseMultiply(
        PauliX.matrix, 2, quantum_state, [[1, 1]]
    )

    print(
        "After Pauli-X gate on qubit 2 with control on qubit 1:\n",
        quantum_state,
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

    num_qubits: Int = 3

    qc: GateCircuit = GateCircuit(num_qubits)

    qc.apply_gates(
        Hadamard(1),
        NOT(2),
        NOT(0, controls=[1]),
        PauliZ(0),
        NOT(2, controls=[1])
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

    print("Creating random circuit of", num_qubits, "qubits and", number_layers, "layers...")
    index: UnsafePointer[Int8] = UnsafePointer[Int8].alloc(number_layers*2*num_qubits)
    random.seed()  # Seed on current time
    random.randint(index, number_layers*2*num_qubits, 0, len(gates_list) - 1)
    print("Random gates choices generated, applying them to the circuit...")
    for iter in range(number_layers):
        for i in range(num_qubits):
            qc.apply(gates_list[Int(index[iter * num_qubits + i])](i))
        qc.barrier()
        for i in range(num_qubits - 1):
            qc.apply(
                gates_list[Int(index[iter * num_qubits + num_qubits + i])](i, controls=[(i + 1) % num_qubits]),
            )
        qc.barrier()

    print("Random circuit created has", qc.num_gates(), "total gates.")

    # print(qc)

    print("Running Simulations...")
    initial_state_bitstring: String = "0" * num_qubits  # Initial state |000...0⟩
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
    """Simulates the circuit from Figure 4 in the paper.
    """
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
    quantum_state = qubitWiseMultiply(Hadamard.matrix, 0, quantum_state)

    print("After Hadamard gate on qubit 1:\n", quantum_state)

    # Apply SWAP gate to qubits 0 and 2
    quantum_state = applySwap(num_qubits, 0, 2, quantum_state)

    print("After SWAP gate on qubits 0 and 1:\n", quantum_state)

    # Apply anti-control Pauli-X gate to qubit 1 with control on qubit 2
    quantum_state = qubitWiseMultiply(
        PauliX.matrix, 1, quantum_state, [[2, 0]]
    )

    print(
        "After anti-control Pauli-X gate on qubit 1 with control on qubit 2:\n",
        quantum_state,
    )

    # Apply control Pauli-X gate to qubit 0 with control on qubit 1
    quantum_state = qubitWiseMultiply(
        PauliX.matrix, 0, quantum_state, [[1, 1]]
    )

    print(
        "After control Pauli-X gate on qubit 0 with control on qubit 1:\n",
        quantum_state,
    )

    # Apply control Pauli-Y gate to qubit 0
    quantum_state = qubitWiseMultiply(
        PauliY.matrix, 0, quantum_state
    )

    print(
        "After control Pauli-Y gate on qubit 0 with control on qubit 1:\n",
        quantum_state,
    )

    # Apply SWAP gate to qubits 1 and 2 with control on qubit 0
    quantum_state = applySwap(num_qubits, 1, 2, quantum_state, [[0, 1]])

    print(
        "After SWAP gate on qubits 1 and 2 with control on qubit 0:\n",
        quantum_state,
    )

    # Apply control Pauli-Z gate to qubit 1
    quantum_state = qubitWiseMultiply(
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
    number_qubits: Int = 15
    number_layers: Int = 10
    if (len(args) == 3):
        try:
            number_qubits = Int(args[1])
            number_layers = Int(args[2])
        except ValueError:
            print("Invalid arguments. Using default values: 3 qubits and 10 layers.")
    else:
        print("Usage: ./main [number_of_qubits] [number_of_layers]")

    # simulate_figure1_circuit()

    # simulate_figure1_circuit_abstract()

    # simulate_random_circuit(number_qubits, number_layers)

    # simulate_figure4_circuit()

    # simulate_figure4_circuit_abstract()

    presentation()