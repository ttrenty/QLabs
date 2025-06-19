# Mojo Implementation of Reference:

# @misc{mcguffinHowWriteSimulator2025,
#   title = {How to Write a Simulator for Quantum Circuits from Scratch: A Tutorial},
#   author = {McGuffin, Michael J. and Robert, Jean-Marc and Ikeda, Kazuki},
#   date = {2025-06-09},
#   abstract = {This tutorial guides a competent programmer through the crafting of a
#   quantum circuit simulator from scratch, even for readers with almost no prior
#   experience in quantum computing. Open source simulators for quantum circuits already
#   exist, but a deeper understanding is gained by writing ones own. With roughly
#   1000-2000 lines of code, one can simulate Hadamard, Pauli X, Y, Z, SWAP, and other
#   quantum logic gates, with arbitrary combinations of control and anticontrol qubits,
#   on circuits of up to 20+ qubits, with no special libraries, on a personal computer.
#   for updating the state vector, and partial trace for finding a reduced density
#   matrix. We also discuss optimizations, and how to compute qubit phase, purity, and
#   other statistics. A complete example implementation in JavaScript is available at
#   https://github.com/MJMcGuffin/muqcs.js , which also demonstrates how to compute von
#   Neumann entropy, concurrence (to quantify entanglement), and magic, while remaining
#   much smaller and easier to study than other popular software packages.},
#   url = {https://arxiv.org/abs/2506.08142v1},
#   urldate = {2025-06-12},
# }


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Conventions          #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# Qubits are indexed from 0 to n-1, top to bottom.
# The top qubit is the least significant bit (LSB), and the bottom is the most significant bit (MSB),
# This leads to states being written in the order |q0 q1 q2 ... q(n-1)⟩.


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Imports              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

from complex import ComplexFloat64

from collections.linked_list import LinkedList

import random

from sys import argv

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Structs              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# << for shift left
# | for bitwise OR
# ^ for bitwise XOR
# & for bitwise AND
# ~ for bitwise NOT

# @fieldwise_init
# struct PerfectBasisState[ElementType: Copyable & Movable](Copyable):
#     """
#     Represents a pure quantum state as a basis state vertical vector.
#     """

#     var state_vector: List[ElementType]

#     fn __getitem__(self, index: Int) -> ElementType:
#         return self.state_vector[index]

#     fn __setitem__(mut self, index: Int, value: ElementType) -> None:
#         self.state_vector[index] = value

#     @staticmethod
#     fn from_bitstring(bitstring: String) -> Self:
#         """
#         Returns a PerfectBasisState corresponding to the given bitstring.

#         @param bitstring: A string of '0's and '1's representing the state, with
#                           the least significant qubit (top one) (LSB) at the start.

#         @return: A PerfectBasisState object with the appropriate data initialized.
#         """
#         num_qubits: Int = len(bitstring)
#         state_vector: List[ElementType] = [0.0] * (
#             1 << num_qubits
#         )  # 2^num_qubits
#         # Put coefficent correspondin to the bitstring to 1
#         index: Int = 0
#         for i, bit in enumerate(bitstring):
#             if bit == "1":
#                 index |= 1 << i  # Set the bit at position i

#         print("index:", index, "num_qubits:", num_qubits)
#         state_vector[index] = 1.0  # Set the amplitude for the state to 1
#         return Self(num_qubits, state_vector)


@fieldwise_init
struct PerfectBasisState(Copyable, Movable, Stringable, Writable):
    """
    Represents a pure quantum state as a basis state vertical vector.
    """

    var num_qubits: Int
    var state_vector: List[ComplexFloat64]
    # var state_vector: InlineArray[ComplexFloat64, 64]
    var size: Int
    

    fn __getitem__(self, index: Int) -> ComplexFloat64:
        return self.state_vector[index]

    fn __setitem__(mut self, index: Int, value: ComplexFloat64) -> None:
        self.state_vector[index] = value

    fn __str__(self) -> String:
        """
        Returns a beautifully formatted string representation of the PerfectBasisState.
        """
        string: String = "PerfectBasisState:\n"
        for i in range(self.size):
            amplitude = self.state_vector[i]
            amplitude_str: String = String(amplitude)
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
        """
        Returns a PerfectBasisState corresponding to the given bitstring.

        @param bitstring: A string of '0's and '1's representing the state, with
                          the least significant qubit (top one) (LSB) at the start.

        @return: A PerfectBasisState object with the appropriate data initialized.
        """
        num_qubits: Int = len(bitstring)
        # state_vector = InlineArray[ComplexFloat64, 64](fill=ComplexFloat64(0.0, 0.0))
        state_vector: List[ComplexFloat64] = [ComplexFloat64(0.0, 0.0)] * (
            1 << num_qubits
        )  # 2^num_qubits
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
        return Self(num_qubits, state_vector, len(state_vector))

    fn write_to[W: Writer](self, mut writer: W) -> None:
        writer.write(String(self))


# @fieldwise_init
# struct Gate[size: Int = 2, values: List[List[ComplexFloat64]] = List[List[ComplexFloat64]](
#     [ComplexFloat64(1, 0), ComplexFloat64(0, 0)],
#     [ComplexFloat64(0, 0), ComplexFloat64(1, 0)],
# )](Copyable, Movable):
#     """
#     Represents a quantum gate as a matrix.
#     """

#     fn __getitem__(self, row: Int, col: Int) -> ComplexFloat64:
#         # return self.matrix[row][col]
#         return values[row][col]

#     fn __str__(self) -> String:
#         """
#         Returns a string representation of the Gate.
#         """
#         string: String = "Gate:\n"
#         for row in range(size):
#             for col in range(size):
#                 string += String("{} ").format(values[row][col])  # invalid call to 'format': could not deduce parameter 'Ts' of callee 'format'. How to do that with compile time type?
#             string += "\n"
#         return string


#     fn write_to[W: Writer](self, mut writer: W) -> None:
#         writer.write(String(self))


@fieldwise_init
# struct Gate(Copyable, Movable, Representable, Stringable, Writable):
struct Gate(Copyable, Movable, Stringable, Writable):
    """
    Represents a quantum gate as a matrix.
    """

    var size: Int
    var matrix: List[List[ComplexFloat64]]
    var symbol: String

    fn __getitem__(self, row: Int, col: Int) -> ComplexFloat64:
        return self.matrix[row][col]

    fn __str__(self) -> String:
        """
        Returns a string representation of the Gate.
        """
        # string: String = "Gate:\n"
        # for row in range(self.size):
        #     for col in range(self.size):
        #         string += String(self[row, col])
        #     string += "\n"
        # return string
        return self.symbol

    fn write_to[W: Writer](self, mut writer: W) -> None:
        writer.write(String(self))

# @fieldwise_init
# struct GateCircuit(Copyable, Movable, Stringable, Writable):
struct GateCircuit(Movable, Stringable, Writable):
    """
    Represents a quantum circuit consisting of gates applied to qubits.
    """

    var num_qubits: Int
    var initial_state: PerfectBasisState
    var verbose: Bool  
    var barriers_positions: LinkedList[
        Int
    ]
    var gates: LinkedList[
        Tuple[Gate, List[Int], List[List[Int]]]
    ]  # (gate, target_qubits, control_qubits)

    fn __init__(
        out self,
        num_qubits: Int,
        initial_state: PerfectBasisState,
        verbose: Bool = False,
        barriers_positions: LinkedList[Int] = [],  # empty list of barriers
        gates: LinkedList[
            Tuple[Gate, List[Int], List[List[Int]]]
        ] = [],  # empty list of gates
    ):
        """
        Initializes a GateCircuit with the given parameters.

        @param num_qubits: The number of qubits in the circuit.
        @param initial_state: The initial state of the quantum system.
        @param verbose: Whether to print verbose output during execution.
        @param barriers_positions: Optional list of barrier positions, where 0 indicates a new gate and 1 indicates a barrier.
        @param gates: Optional list of gates in the circuit, where each gate is represented as a tuple (gate, target_qubits, control_qubits).
        """
        self.num_qubits = num_qubits
        self.initial_state = initial_state
        self.verbose = verbose
        self.barriers_positions = barriers_positions
        self.gates = gates

    fn __str__(self) -> String:
        """
        Returns a string representation of the GateCircuit.
        """
        # string: String = "GateCircuit:\n"
        # string += String("Number of qubits: {}\n").format(self.num_qubits)
        # string += String("Initial state: {}\n").format(self.initial_state)
        # string += "Gates:\n"
        # print("TODO")
        string: String = "GateCircuit:\n"
        string += "Number of qubits:" + String(self.num_qubits) + "\n"
        string += "Total number of gates: " + String(len(self.gates)) + "\n"
        string += "Initial state:\n" + String(self.initial_state) + "\n"
        # print the circuit in a human-readable format such as:
# |0> -------|X|--|Z|--
#             |       
# |0> --|H|---*----*---
#                  |    
# |0> --|X|-------|X|--
        string += "Gates:\n"
        # seperate each gates layers with "--", than if there is a gate on that wire use
        # |symbol|, if not use "---", then add another "--", and repeat for all gates and wires,
        # add inbetween lines with the "|" symbol for control wires, use * for control wires
        # and o for anti-control wires. don't forget to include the state of qubits at the start.
        # don't bring attention to the barriers
        # if len(self.gates) == 0:
        #     string += "No gates applied.\n"
        # else:
            # Create a list of strings for each qubit
            # qubit_lines: List[String] = ["" for _ in range(self.num_qubits)]
        #     for gate, target_qubits, control_qubits in self.gates:
        #         # Add the gate symbol to the target qubit line
        #         for target_qubit in target_qubits:
        #             qubit_lines[target_qubit] += "|" + gate.symbol + "|"
        #         # Add control and anti-control symbols
        #         for control_qubit in control_qubits:
        #             wire_index, flag = control_qubit[0], control_qubit[1]
        #             if flag == 1:  # Control
        #                 qubit_lines[wire_index] += "*"
        #             else:  # Anti-control
        #                 qubit_lines[wire_index] += "o"
        #     # Join the lines with newlines
        #     string += "\n".join(qubit_lines) + "\n"
        # # Add barriers
        return string

    fn write_to[W: Writer](self, mut writer: W) -> None:
        writer.write(String(self))

    fn apply(
        self,
        gate: Gate,
        target_qubit: Int,  # TODO more than one qubit gates
        control_qubits: List[Int] = [],
        is_anti_control: List[Bool] = [],
    ) -> Self:
        """
        Applies a quantum gate to a specific qubit in the circuit.

        @param gate: The quantum gate to apply.
        @param target_qubit: The index of the qubit to which the gate is applied.
        @param control_qubits: A list of control qubits.
        @param is_anti_control: A list indicating whether each control qubit is an anti-control (False) or control (True).
        """
        gates = self.gates
        gate_control_qubits: List[List[Int]] = []
        for i in range(len(control_qubits)):
            if is_anti_control[i]:
                gate_control_qubits.append(
                    [control_qubits[i], 0]
                )  # anti-control
            else:
                gate_control_qubits.append([control_qubits[i], 1])  # control

        # Append the gate to the circuit
        gates.append((gate, List[Int](target_qubit), gate_control_qubits))
        barriers_positions = self.barriers_positions
        barriers_positions.append(0)  # Mark a new gate position

        return Self(
            self.num_qubits,
            self.initial_state,
            self.verbose,
            barriers_positions,
            gates,
        )

    fn barrier(self) -> Self:
        """
        Adds a barrier to the circuit, which can be used to separate different layers of gates.
        """
        barriers_positions = self.barriers_positions
        barriers_positions.append(1)  # Mark a barrier position
        return Self(
            self.num_qubits,
            self.initial_state,
            self.verbose,
            barriers_positions,
            self.gates,
        )

    fn run(self, verbose_step_size: String = ShowOnlyEnd) -> PerfectBasisState:
        """
        Runs the quantum circuit and applies all gates to the initial state.
        """
        if self.verbose:
            print(
                "Running quantum circuit simulation with verbose step size:",
                verbose_step_size,
            )
            print("Initial state:\n", self.initial_state)

        # Start with the initial state
        quantum_state: PerfectBasisState = self.initial_state
        gate_index: Int = 0
        barrier_index: Int = 0
        layer_index: Int = 0
        for gate, target_qubits, control_qubits in self.gates:
            
            # print("Before qubit-wise multiplication:")
            # Apply the gate to the quantum state
            quantum_state = qubitWiseMultiply(
                self.num_qubits,
                gate,
                target_qubits[0],
                quantum_state,
                control_qubits,
            )
            gate_index += 1

            if self.verbose :
                try:
                    if verbose_step_size == ShowAfterEachGate:
                        print(String("New quantum state after gate {}:\n").format(gate_index), quantum_state)
                        
                    if verbose_step_size == ShowAfterEachLayer
                        and self.barriers_positions[barrier_index] == 1:
                        print(String("New quantum state after layer {}:\n").format(layer_index), quantum_state)
                        barrier_index += 1
                        layer_index += 1
                except IndexError:
                    # If we run out of barriers, just continue
                    print("Error during printing verbose output, continuing...")
                    if verbose_step_size == ShowAfterEachLayer
                        and self.barriers_positions[barrier_index] == 1:
                        barrier_index += 1
                        layer_index += 1
            barrier_index += 1

           

        if self.verbose and verbose_step_size == ShowOnlyEnd:
            print("Final quantum state:\n", quantum_state)

        return quantum_state

    fn num_gates(self) -> Int:
        """
        Returns the total number of gates in the circuit.
        """
        return len(self.gates)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Aliases              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

alias Hadamard = Gate(
    size=2,
    matrix=List[List[ComplexFloat64]](
        [ComplexFloat64(1 / 2**0.5, 0), ComplexFloat64(1 / 2**0.5, 0)],
        [ComplexFloat64(1 / 2**0.5, 0), ComplexFloat64(-1 / 2**0.5, 0)],
    ),
    symbol="H",
)

alias PauliX = Gate(
    size=2,
    matrix=List[List[ComplexFloat64]](
        [ComplexFloat64(0, 0), ComplexFloat64(1, 0)],
        [ComplexFloat64(1, 0), ComplexFloat64(0, 0)],
    ),
    symbol="X",
)

alias NOT = PauliX

alias PauliZ = Gate(
    size=2,
    matrix=List[List[ComplexFloat64]](
        [ComplexFloat64(1, 0), ComplexFloat64(0, 0)],
        [ComplexFloat64(0, 0), ComplexFloat64(-1, 0)],
    ),
    symbol="Z",
)

alias PauliY = Gate(
    size=2,
    matrix=List[List[ComplexFloat64]](
        [ComplexFloat64(0, 0), ComplexFloat64(0, -1)],
        [ComplexFloat64(0, 1), ComplexFloat64(0, 0)],
    ),
    symbol="Y",
)

alias ShowOnlyEnd = "ShowOnlyEnd"
alias ShowAfterEachLayer = "ShowAfterEachLayer"
alias ShowAfterEachGate = "ShowAfterEachGate"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Functions            #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


fn qubitWiseMultiply(
    num_qubits: Int,
    gate: Gate,
    target_qubit: Int,
    owned quantum_state: PerfectBasisState,
    control_bits: List[List[Int]] = [],
) -> PerfectBasisState:
    """
    Applies a quantum gate to a specific qubit in the quantum state.

    @param num_qubits: The total number of qubits in the circuit.
    @param gate: The 2x2 matrix representing the quantum gate.
    @param target_qubit: The index of the qubit to which the gate is applied.
    @param quantum_state: The current state of the quantum system.
    @param control_bits: A list of control bits, where each bit is represented as [wire_index, flag].
                        If flag is true, it is a control bit; if false, it is an anti-control bit.

    @return: A new PerfectBasisState with the gate applied.
    """
    # print("In qubitWiseMultiply:")
    inclusion_mask: Int = 0
    desired_value_mask: Int = 0
    for control in control_bits:
        wire_index, flag = control[0], control[1]
        bit: Int = 1 << wire_index  # efficient way of computing 2^wire_index
        inclusion_mask |= bit  # turn on the bit
        if flag == 1:
            desired_value_mask |= bit  # turn on the bit

    size_of_state_vector: Int = 1 << num_qubits  # 2^num_qubits
    size_of_half_block: Int = 1 << target_qubit  # 2^target_qubit
    size_of_block: Int = size_of_half_block << 1
    new_state_vector: PerfectBasisState = quantum_state  # copies all amplitudes from quantum_state to new_state_vector
    for block_start in range(0, size_of_state_vector, size_of_block):
        for offset in range(size_of_half_block):
            i1: Int = (
                block_start | offset
            )  # faster than, but equivalent to, block_start + offset
            if (i1 & inclusion_mask) != desired_value_mask:
                continue  # skip this iteration if the control bits do not match
            i2: Int = (
                i1 | size_of_half_block
            )  # equivalent to i1 + size_of_half_block
            new_state_vector[i1] = (
                gate[0, 0] * quantum_state[i1] + gate[0, 1] * quantum_state[i2]
            )
            new_state_vector[i2] = (
                gate[1, 0] * quantum_state[i1] + gate[1, 1] * quantum_state[i2]
            )
    return new_state_vector


fn swapBits(
    k: Int,
    i: Int,
    j: Int,
) -> Int:
    """
    Swaps the ith and jth bits of the integer k.

    @param k: The integer whose bits are to be swapped.
    @param i: The index of the first bit to swap.
    @param j: The index of the second bit to swap.

    @return: The integer with the specified bits swapped.
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
    num_qubits: Int,  # The number of qubits in the circuit
    i: Int,  # The index of the first qubit to swap (0 for LSB, num_qubits-1 for MSB)
    j: Int,  # The index of the second qubit to swap
    quantum_state: PerfectBasisState,  # The current state of the quantum system
    control_bits: List[
        List[Int]
    ] = [],  # A list of control bits, where each bit is [wire_index, flag]
) -> PerfectBasisState:
    """
    Applies a SWAP gate to two specific qubits in the quantum state.
    Control bits can limit the effect of the SWAP to a subset of the amplitudes in |a> (?).

    @param num_qubits: The total number of qubits in the circuit.
    @param i: The index of the first qubit to swap.
    @param j: The index of the second qubit to swap.
    @param quantum_state: The current state of the quantum system.
    @param control_bits: A list of control bits, where each bit is [wire_index, flag].
                        If flag is 1, it is a control bit; if 0, it is an anti-control bit.
    @return: A new PerfectBasisState with the SWAP gate applied.
    """
    new_state_vector: PerfectBasisState = quantum_state  # copies all amplitudes from quantum_state to new_state_vector
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
    num_qubits: Int,  # The number of qubits in the circuit
    i: Int,  # The index of the first qubit to swap (0 for LSB, num_qubits-1 for MSB)
    j: Int,  # The index of the second qubit to swap
    quantum_state: PerfectBasisState,  # The current state of the quantum system
    control_bits: List[
        List[Int]
    ] = [],  # A list of control bits, where each bit is [wire_index, flag]
) -> PerfectBasisState:
    """
    Applies a SWAP gate to two specific qubits in the quantum state.
    Control bits can limit the effect of the SWAP to a subset of the amplitudes in |a> (?).

    @param num_qubits: The total number of qubits in the circuit.
    @param i: The index of the first qubit to swap.
    @param j: The index of the second qubit to swap.
    @param quantum_state: The current state of the quantum system.
    @param control_bits: A list of control bits, where each bit is [wire_index, flag].
                        If flag is 1, it is a control bit; if 0, it is an anti-control bit.
    @return: A new PerfectBasisState with the SWAP gate applied.
    """
    new_state_vector: PerfectBasisState = quantum_state  # copies all amplitudes from quantum_state to new_state_vector
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
    # Number of qubits
    num_qubits: Int = 3

    # Initialize the quantum circuit to the |000⟩ state
    quantum_state: PerfectBasisState = PerfectBasisState.from_bitstring("000")

    print("Intial quantum state:\n", quantum_state)

    # Apply Hadamard gate to qubit 1
    quantum_state = qubitWiseMultiply(num_qubits, Hadamard, 1, quantum_state)

    print("After Hadamard gate on qubit 1:\n", quantum_state)

    # Apply Pauli-X gate to qubit 2
    quantum_state = qubitWiseMultiply(num_qubits, PauliX, 2, quantum_state)

    print("After Pauli-X gate on qubit 2:\n", quantum_state)

    # Apply Pauli-X gate to qubit 0 with control on qubit 1
    quantum_state = qubitWiseMultiply(
        num_qubits, PauliX, 0, quantum_state, [[1, 1]]
    )

    print(
        "After Pauli-X gate on qubit 0 with control on qubit 1:\n",
        quantum_state,
    )

    # Apply Pauli-Z gate to qubit 0
    quantum_state = qubitWiseMultiply(num_qubits, PauliZ, 0, quantum_state)

    print("After Pauli-Z gate on qubit 0:\n", quantum_state)

    # Apply Pauli-X gate to qubit 2 with control on qubit 1
    quantum_state = qubitWiseMultiply(
        num_qubits, PauliX, 2, quantum_state, [[1, 1]]
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

    # Create the initial state |000⟩
    initial_state_bitstring: String = "000"
    initial_state: PerfectBasisState = PerfectBasisState.from_bitstring(
        initial_state_bitstring
    )

    num_qubits: Int = len(initial_state_bitstring)

    qc: GateCircuit = GateCircuit(num_qubits, initial_state, verbose=True)

    qc = qc.apply(Hadamard, 1)
    qc = qc.apply(NOT, 2)
    qc = qc.barrier()
    qc = qc.apply(NOT, 0, control_qubits=[1], is_anti_control=[False])
    qc = qc.barrier()
    qc = qc.apply(PauliZ, 0)
    qc = qc.apply(NOT, 2, control_qubits=[1], is_anti_control=[False])

    # final_state: PerfectBasisState = qc.run(verbose_step_size=ShowOnlyEnd)
    final_state: PerfectBasisState = qc.run(
        verbose_step_size=ShowAfterEachLayer
    )
    # final_state: PerfectBasisState = qc.run(verbose_step_size=ShowAfterEachGate)

    print("Final quantum state:\n", final_state)


fn simulate_random_circuit(number_qubits: Int, number_layers: Int) -> None:
    initial_state_bitstring: String = "0" * number_qubits  # Initial state |000...0⟩

    initial_state: PerfectBasisState = PerfectBasisState.from_bitstring(
        initial_state_bitstring
    )

    num_qubits: Int = len(initial_state_bitstring)

    qc: GateCircuit = GateCircuit(num_qubits, initial_state, verbose=False)

    gates_list: List[Gate] = [Hadamard, PauliX, PauliY, PauliZ]

    # index: UnsafePointer[Int8] = UnsafePointer[Int8].alloc(1)
    # print("Creating random circuit...")
    # random.seed()  # Seed on current time
    # for _ in range(800):
    #     for i in range(num_qubits):
    #         random.randint(index, 1, 0, len(gates_list) - 1)
    #         qc = qc.apply(gates_list[Int(index[0])], i)
    #     qc = qc.barrier()
    #     for i in range(num_qubits - 1):
    #         random.randint(index, 1, 0, num_qubits - 1)
    #         qc = qc.apply(
    #             gates_list[Int(index[0])],
    #             i,
    #             control_qubits=[(i + 1) % num_qubits],
    #             is_anti_control=[False],
    #         )
    #     qc = qc.barrier()
    
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
    #             control_qubits=[(i + 1) % num_qubits],
    #             is_anti_control=[False],
    #         )
    #     qc = qc.barrier()

      
    index: UnsafePointer[Int8] = UnsafePointer[Int8].alloc(number_layers*2*num_qubits)
    print("Creating random circuit of", num_qubits, "qubits and", number_layers, "layers...")
    random.seed()  # Seed on current time
    random.randint(index, number_layers*2*num_qubits, 0, len(gates_list) - 1)
    for iter in range(number_layers):
        for i in range(num_qubits):
            qc = qc.apply(gates_list[Int(index[iter * num_qubits + i])], i)
        qc = qc.barrier()
        for i in range(num_qubits - 1):
            qc = qc.apply(
                gates_list[Int(index[iter * num_qubits + num_qubits + i])],
                i,
                control_qubits=[(i + 1) % num_qubits],
                is_anti_control=[False],
            )
        qc = qc.barrier()

    print("Random circuit created has", qc.num_gates(), "gates.")

    print("Running Simulations...")
    for i in range(1000):
        print("Iteration:", i, end="\r")
        _: PerfectBasisState = qc.run(verbose_step_size=ShowAfterEachLayer)
    print("")
    # final_state: PerfectBasisState = qc.run(verbose_step_size=ShowOnlyEnd)
    # print("Final quantum state:\n", final_state)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Main                 #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def main():
    args = argv()
    number_qubits: Int = 10
    number_layers: Int = 20
    if (len(args) == 3):
        try:
            number_qubits = Int(args[1])
            number_layers = Int(args[2])
        except ValueError:
            print("Invalid arguments. Using default values: 3 qubits and 10 layers.")
    else:
        print("Usage: ./main [number_of_qubits] [number_of_layers]")

    # simulate_figure1_circuit()

    simulate_figure1_circuit_abstract()

    simulate_random_circuit(number_qubits, number_layers)
