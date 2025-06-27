# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Imports              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

from ..base.qubits_operations import (
    qubit_wise_multiply,
    qubit_wise_multiply_extended,
    apply_swap,
)

from ..base.state_and_matrix import (
    PureBasisState,
)
from ..base.gate import _START, _SEPARATOR, SWAP

from .gate_circuit import GateCircuit


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Aliases              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

alias ShowAfterEachGate = "ShowAfterEachGate"
alias ShowAfterEachLayer = "ShowAfterEachLayer"
alias ShowOnlyEnd = "ShowOnlyEnd"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Structs              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


struct StateVectorSimulator(Copyable, Movable):
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
        # initial_state: Optional[PureBasisState] = None, # TODO ask how to use that with return of next_layer()
        # initial_state: __type_of(Self.initial_state), # doesn't work
        initial_state: PureBasisState,
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
        # new_initial_state = initial_state.or_else(PureBasisState.from_bitstring("0" * circuit.num_qubits))
        new_initial_state = initial_state
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
    ) -> __type_of(quantum_state):
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
        while (
            gate.symbol in [_SEPARATOR.symbol, _START.symbol] and len(gates) > 0
        ):
            try:
                gate = gates.pop(
                    0
                )  # Get the next gate, skipping any layer separators
            except e:
                print("Error: No gates left to apply. Returning current state.")
                return quantum_state  # No gates left to apply

        new_quantum_state = qubit_wise_multiply(
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
        """Resets the simulator to its initial state and circuit."""
        self.circuit = self.original_circuit  # Reset circuit to original
        self.initial_state = self.original_initial_state  # Reset initial state

    fn next_layer(
        self,
        quantum_state: PureBasisState,
    ) -> (Self, __type_of(quantum_state)):
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

        new_quantum_state: __type_of(
            quantum_state
        ) = quantum_state  # Start with the current state

        gate = _START
        while len(gates) > 0 and gate.symbol != _SEPARATOR.symbol:
            try:
                gate = gates.pop(0)
            except e:
                print("Error: No gates left to apply. Returning current state.")
                return (self, quantum_state)  # No gates left to apply

            new_quantum_state = qubit_wise_multiply(
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
    ) -> (Self, __type_of(quantum_state)):
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
            if gate.symbol == _SEPARATOR.symbol:
                continue
            elif gate.symbol == SWAP.symbol:
                if len(gate.target_qubits) != 2:
                    print("Error: SWAP gate must have exactly 2 target qubits.")
                    continue
                quantum_state = apply_swap(
                    self.circuit.num_qubits,
                    gate.target_qubits[0],
                    gate.target_qubits[1],
                    quantum_state,
                    gate.control_qubits_with_flags,
                )
            else:
                # Apply the next gate
                quantum_state = qubit_wise_multiply_extended(
                    len(gate.target_qubits),  # Number of target qubits
                    gate.matrix,
                    gate.target_qubits,  # Assuming single target qubit
                    quantum_state,
                    gate.control_qubits_with_flags,
                )

            i += 1
            if self.verbose:
                if self.verbose_step_size == ShowAfterEachGate:
                    print(
                        "New quantum state after gate " + String(i) + ":\n",
                        quantum_state,
                    )
                elif (
                    self.verbose_step_size == ShowAfterEachLayer
                    and gate.symbol == _SEPARATOR.symbol
                ):
                    print(
                        "New quantum state after layer "
                        + String(layer_index)
                        + ":\n",
                        quantum_state,
                    )
                if gate.symbol == _SEPARATOR.symbol:
                    layer_index += 1  # Increment layer index after a separator

        if self.verbose and self.verbose_step_size == ShowOnlyEnd:
            print("Final quantum state:\n", quantum_state)

        return quantum_state
