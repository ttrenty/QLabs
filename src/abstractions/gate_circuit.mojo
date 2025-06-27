# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Imports              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

from collections.linked_list import LinkedList

from ..base.gate import Gate, _START, _SEPARATOR

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Structs              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


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

        wires: List[String] = [
            String("--")
        ] * self.num_qubits  # Initialize wires for each qubit
        in_between: List[String] = [String("  ")] * (
            self.num_qubits - 1
        )  # Initialize control wires in-between qubits
        wires_current_gate: List[Int] = [
            0
        ] * self.num_qubits  # Track current gate on each wire
        for gate in self.gates:
            if (
                gate.symbol == _START.symbol
            ):  # If it's a start gate, initialize the wires
                continue
            if (
                gate.symbol == _SEPARATOR.symbol
            ):  # If it's a separator, add a new layer
                for i in range(self.num_qubits - 1):
                    wires[i] += String("|")
                    in_between[i] += String("|")
                wires[self.num_qubits - 1] += String("|")
            else:
                qubit_index: Int = gate.target_qubits[
                    0
                ]  # Assuming single target qubit for simplicity
                controls_flags: List[List[Int]] = gate.control_qubits_with_flags

                for control in controls_flags:
                    # control_index, flag = control[0], control[1]
                    control_index = control[0]
                    while (
                        wires_current_gate[qubit_index]
                        < wires_current_gate[control_index]
                    ):
                        wires[qubit_index] += String("-----")
                        # in_between[qubit_index] += String("     ")
                        wires_current_gate[
                            qubit_index
                        ] += 1  # Increment the current gate on this wire

                wires[qubit_index] += (
                    String("|") + gate.symbol + String("|")
                )  # Add the gate symbol to the wire
                wires[qubit_index] += String(
                    "--"
                )  # Add a separator after the gate

                wires_current_gate[
                    qubit_index
                ] += 1  # Increment the current gate on this wire

                if len(controls_flags) > 0:
                    for control in controls_flags:
                        control_index, flag = control[0], control[1]
                        if flag == 1:  # Control qubit
                            wires[control_index] += String("-*-")
                        else:  # Anti-control qubit
                            wires[control_index] += String("-o-")

                        while (
                            wires_current_gate[control_index]
                            < wires_current_gate[qubit_index]
                        ):
                            wires[control_index] += String("--")
                            wires_current_gate[
                                control_index
                            ] += 1  # Increment the current gate on this wire

        for i in range(self.num_qubits):
            string += wires[i] + "\n"  # Print each wire
            if i < self.num_qubits - 1:
                string += (
                    in_between[i] + "\n"
                )  # Print the in-between control wires

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
