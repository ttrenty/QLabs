# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Imports              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

from sys import has_accelerator
from gpu import thread_idx, block_dim, block_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor, IntTuple, print_layout

from ..base.qubits_operations import (
    qubit_wise_multiply,
    qubit_wise_multiply_extended,
    apply_swap,
)

from ..base.state_and_matrix import (
    StateVector,
)
from ..base.gate import _START, _SEPARATOR, SWAP

from ..base.gate import (
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
    iSWAP,
)

from ..base.gpu import qubit_wise_multiply_inplace_gpu

from ..local_stdlib.complex import ComplexFloat32

from .gate_circuit import GateCircuit

alias dtype = DType.float32

alias GATE_SIZE = 2
alias NUMBER_CONTROL_BITS = 1

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Aliases              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

alias ShowAfterEachGate = "ShowAfterEachGate"
alias ShowAfterEachLayer = "ShowAfterEachLayer"
alias ShowOnlyEnd = "ShowOnlyEnd"


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Structs              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


struct StateVectorSimulator[
    gpu_num_qubits: Int = 0,
    gpu_gate_ordered_set: List[Gate] = [
        Hadamard,
        PauliX,
        PauliY,
        PauliZ,
        iSWAP,
    ],
    gpu_control_gate_count: Int = 0,
    gpu_control_bits_list: List[List[List[Int]]] = [[[]]],
](Copyable, Movable):
    """A noiseless simulator for quantum circuits that uses State Vector Simulation.

    Parameters:
        gpu_num_qubits: The number of qubits in the quantum circuit for GPU simulation.
        gpu_gate_ordered_set: The ordered set of gates to be used in the GPU simulation.
        gpu_control_gate_count: The number of control gates in the circuit for GPU simulation.
        gpu_control_bits_list: A list of control bits for each gate in the circuit for GPU simulation.

    This simulator applies quantum gates to a state vector representing the quantum state.
    """

    var circuit: GateCircuit
    """The quantum circuit containing the gates to be applied."""
    var original_circuit: GateCircuit
    """The original circuit before any modifications, used for resetting the simulator."""
    var initial_state: StateVector
    """The initial state of the quantum system before any gates are applied."""
    var original_initial_state: StateVector
    """The original initial state before any modifications, used for resetting the simulator."""
    var use_gpu_if_available: Bool
    """Whether to use GPU acceleration if available (not implemented yet)."""
    var verbose: Bool
    """Whether to print verbose output during simulation steps."""
    var verbose_step_size: String
    """The verbosity level for simulation output, controlling how often state updates are printed."""

    @always_inline
    fn __init__(
        out self,
        owned circuit: GateCircuit,
        # initial_state: Optional[StateVector] = None, # TODO ask how to use that with return of next_layer()
        # initial_state: __type_of(Self.initial_state), # doesn't work
        initial_state: StateVector,
        use_gpu_if_available: Bool = False,
        verbose: Bool = False,
        verbose_step_size: String = "ShowOnlyEnd",
    ):
        """Initializes a StateVectorSimulator with the given parameters.

        Args:
            circuit: The quantum circuit containing the gates to be applied.
            initial_state: The initial state of the quantum system.
            use_gpu_if_available: Whether to use GPU acceleration if available (not implemented yet).
            verbose: Whether to print verbose output during simulation steps.
            verbose_step_size: The verbosity level for simulation output.
        """
        # new_initial_state = initial_state.or_else(StateVector.from_bitstring("0" * circuit.num_qubits))
        new_initial_state = initial_state
        self.circuit = circuit
        self.original_circuit = circuit
        self.initial_state = new_initial_state
        self.original_initial_state = new_initial_state
        self.use_gpu_if_available = use_gpu_if_available
        self.verbose = verbose
        self.verbose_step_size = verbose_step_size
        if self.use_gpu_if_available:

            @parameter
            if not has_accelerator():
                print(
                    "No compatible GPU found. Falling back to CPU simulation."
                )
                self.use_gpu_if_available = (
                    False  # Disable GPU if not available
                )
            else:
                try:
                    ctx = DeviceContext()
                    print("Using GPU simulation on device:", ctx.name())
                except e:
                    print("Failed to create GPU context:", e)
                    print("Falling back to CPU simulation.")
                    self.use_gpu_if_available = False
        if (
            self.use_gpu_if_available
            and self.verbose
            and self.verbose_step_size != ShowOnlyEnd
        ):
            print(
                (
                    "Cannot have verbose output other than ShowOnlyEnd when"
                    " using GPU simulation."
                ),
                "Defaulting to ShowOnlyEnd.",
            )
            self.verbose_step_size = (
                ShowOnlyEnd  # Force ShowOnlyEnd for GPU simulation
            )

    @always_inline
    fn next_gate(
        mut self,
        owned quantum_state: StateVector,
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
        quantum_state: StateVector,
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
                self.use_gpu_if_available,
                self.verbose,
                self.verbose_step_size,
            ),
            new_quantum_state,
        )

    fn next_block(
        self,
        quantum_state: StateVector,
    ) -> (Self, __type_of(quantum_state)):
        return self.next_layer(quantum_state)  # For now, treat blocks as layers

    fn run(self) -> StateVector:
        """Runs the quantum circuit simulation.

        Applies all gates in sequence to the initial state and computes the
        final quantum state. Will print verbose output if enabled and at
        the specified verbosity level.

        Returns:
            The final `StateVector` after all gates have been applied.
        """
        if self.verbose:
            print(
                "Running quantum circuit simulation with verbose step size:",
                self.verbose_step_size,
            )
            print("Initial state:\n", self.initial_state)

        if self.use_gpu_if_available:

            @parameter
            if has_accelerator():
                try:
                    ctx = DeviceContext()
                    final_state = self.run_gpu(ctx)
                    return final_state
                except e:
                    print("Failed to create GPU context:", e)
                    print("Falling back to CPU simulation.")

        # Start with the initial state
        quantum_state: StateVector = self.initial_state
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

    def run_gpu(self, ctx: DeviceContext) -> StateVector:
        """Runs the quantum circuit simulation.

        Applies all gates in sequence to the initial state and computes the
        final quantum state. Will print verbose output if enabled and at
        the specified verbosity level.

        Returns:
            The final `StateVector` after all gates have been applied.
        """

        quantum_state: StateVector = self.initial_state

        alias num_qubits = gpu_num_qubits

        alias circuit_number_control_gates = gpu_control_gate_count
        alias circuit_control_bits_layout = Layout.row_major(
            circuit_number_control_gates, NUMBER_CONTROL_BITS, 2
        )

        gate_set_dic: Dict[String, Int] = {}
        for i in range(len(gpu_gate_ordered_set)):
            gate_set_dic[gpu_gate_ordered_set[i].symbol] = i

        alias gate_set_size = len(gpu_gate_ordered_set)
        alias gate_set_1qubit_layout = Layout.row_major(
            gate_set_size, GATE_SIZE, GATE_SIZE
        )

        alias state_vector_size = 1 << num_qubits
        alias state_vector_layout = Layout.row_major(state_vector_size)

        alias total_threads = state_vector_size

        alias max_threads_per_block = 1024  # Maximum threads per block in CUDA
        alias blocks_per_grid = (
            total_threads + max_threads_per_block - 1
        ) // max_threads_per_block

        alias threads_per_block = (
            max_threads_per_block,
            1,
            1,
        )

        @parameter
        if total_threads < max_threads_per_block:
            alias threads_per_block = (
                total_threads,
                1,
                1,
            )  # 1D block of threads

        # var control_bits_list: List[List[List[Int]]] = [
        #     [[1, 1]],  # Control on qubit 1 and is control because flag=1
        #     [[1, 1]],  # Control on qubit 1 and is control because flag=1
        # ]
        var control_bits_list = gpu_control_bits_list

        # -- Create GPU variables -- #
        # These don't need to be initialized to zero, they will be filled later

        host_quantum_state_re = ctx.enqueue_create_host_buffer[dtype](
            state_vector_size
        )
        host_quantum_state_im = ctx.enqueue_create_host_buffer[dtype](
            state_vector_size
        )

        host_gate_set_re = ctx.enqueue_create_host_buffer[dtype](
            gate_set_size * GATE_SIZE * GATE_SIZE
        )
        host_gate_set_im = ctx.enqueue_create_host_buffer[dtype](
            gate_set_size * GATE_SIZE * GATE_SIZE
        )

        host_control_bits_circuit = ctx.enqueue_create_host_buffer[DType.int32](
            circuit_number_control_gates * NUMBER_CONTROL_BITS * 2
        )

        # -- Initialize the quantum circuit to the |000⟩ state -- #

        # Wait for host buffers to be ready
        ctx.synchronize()

        # -- Fill host buffers -- #

        for i in range(state_vector_size):
            host_quantum_state_re[i] = quantum_state[i].re
            host_quantum_state_im[i] = quantum_state[i].im

        for i in range(gate_set_size):
            gate = gpu_gate_ordered_set[i]
            for j in range(GATE_SIZE):
                for k in range(GATE_SIZE):
                    index = gate_set_1qubit_layout(
                        IntTuple(i, j, k)
                    )  # Get the index in the 1D buffer
                    host_gate_set_re[index] = gate[j, k].re
                    host_gate_set_im[index] = gate[j, k].im

        for i in range(circuit_number_control_gates):
            for j in range(NUMBER_CONTROL_BITS):
                for k in range(2):
                    index = circuit_control_bits_layout(IntTuple(i, j, k))
                    host_control_bits_circuit[index] = control_bits_list[i][j][
                        k
                    ]

        # -- Copy host buffers to device buffers -- #
        quantum_state_re = ctx.enqueue_create_buffer[dtype](state_vector_size)
        quantum_state_im = ctx.enqueue_create_buffer[dtype](state_vector_size)

        gate_set_re = ctx.enqueue_create_buffer[dtype](
            gate_set_size * GATE_SIZE * GATE_SIZE
        )
        gate_set_im = ctx.enqueue_create_buffer[dtype](
            gate_set_size * GATE_SIZE * GATE_SIZE
        )

        control_bits_circuit = ctx.enqueue_create_buffer[DType.int32](
            circuit_number_control_gates * NUMBER_CONTROL_BITS * 2
        )
        current_control_gate_circuit = ctx.enqueue_create_buffer[DType.int32](1)

        # Create other buffers for functions

        quantum_state_out_re = ctx.enqueue_create_buffer[dtype](
            state_vector_size
        )
        quantum_state_out_im = ctx.enqueue_create_buffer[dtype](
            state_vector_size
        )

        quantum_state_re.enqueue_copy_from(host_quantum_state_re)
        quantum_state_im.enqueue_copy_from(host_quantum_state_im)

        gate_set_re.enqueue_copy_from(host_gate_set_re)
        gate_set_im.enqueue_copy_from(host_gate_set_im)

        control_bits_circuit.enqueue_copy_from(host_control_bits_circuit)

        ctx.enqueue_memset(current_control_gate_circuit, 0)
        ctx.enqueue_memset(quantum_state_out_re, 0.0)
        ctx.enqueue_memset(quantum_state_out_im, 0.0)

        # -- Create layout tensors for GPU operations -- #
        gate_set_re_tensor = LayoutTensor[
            mut=False, dtype, gate_set_1qubit_layout
        ](gate_set_re.unsafe_ptr())
        gate_set_im_tensor = LayoutTensor[
            mut=False, dtype, gate_set_1qubit_layout
        ](gate_set_im.unsafe_ptr())

        quantum_state_re_tensor = LayoutTensor[
            mut=True, dtype, state_vector_layout
        ](quantum_state_re.unsafe_ptr())
        quantum_state_im_tensor = LayoutTensor[
            mut=True, dtype, state_vector_layout
        ](quantum_state_im.unsafe_ptr())

        quantum_state_out_re_tensor = LayoutTensor[
            mut=True, dtype, state_vector_layout
        ](quantum_state_out_re.unsafe_ptr())
        quantum_state_out_im_tensor = LayoutTensor[
            mut=True, dtype, state_vector_layout
        ](quantum_state_out_im.unsafe_ptr())

        control_bits_circuit_tensor = LayoutTensor[
            mut=False, DType.int32, circuit_control_bits_layout
        ](control_bits_circuit.unsafe_ptr())
        current_control_gate_circuit_tensor = LayoutTensor[
            mut=True, DType.int32, Layout.row_major(1)
        ](current_control_gate_circuit.unsafe_ptr())

        # -- Apply circuit operations -- #

        current_buffer: Int = 0
        for gate in self.circuit.gates:  # Iterate over the gates in the circuit
            if gate.symbol == _SEPARATOR.symbol:
                continue
            elif gate.symbol == SWAP.symbol:
                print("SWAP gate not implemented in GPU version.")
            else:
                # Apply the next gate
                if current_buffer == 0:
                    ctx.enqueue_function[
                        qubit_wise_multiply_inplace_gpu[
                            state_vector_size=state_vector_size,
                            gate_set_size=gate_set_size,
                            circuit_number_control_gates=circuit_number_control_gates,
                            number_control_bits=0,
                        ]
                    ](
                        gate_set_re_tensor,
                        gate_set_im_tensor,
                        gate_set_dic[gate.symbol],
                        GATE_SIZE,  # Assumming single target qubit
                        gate.target_qubits[0],  # target_qubit
                        quantum_state_re_tensor,
                        quantum_state_im_tensor,
                        num_qubits,  # number_qubits
                        quantum_state_out_re_tensor,
                        quantum_state_out_im_tensor,
                        control_bits_circuit_tensor,
                        current_control_gate_circuit_tensor,
                        grid_dim=blocks_per_grid,
                        block_dim=threads_per_block,
                    )
                    current_buffer = 1
                else:
                    ctx.enqueue_function[
                        qubit_wise_multiply_inplace_gpu[
                            state_vector_size=state_vector_size,
                            gate_set_size=gate_set_size,
                            circuit_number_control_gates=circuit_number_control_gates,
                            number_control_bits=0,
                        ]
                    ](
                        gate_set_re_tensor,
                        gate_set_im_tensor,
                        gate_set_dic[gate.symbol],
                        GATE_SIZE,  # Assumming single target qubit
                        gate.target_qubits[0],  # target_qubit
                        quantum_state_out_re_tensor,
                        quantum_state_out_im_tensor,
                        num_qubits,  # number_qubits
                        quantum_state_re_tensor,
                        quantum_state_im_tensor,
                        control_bits_circuit_tensor,
                        current_control_gate_circuit_tensor,
                        grid_dim=blocks_per_grid,
                        block_dim=threads_per_block,
                    )
                    current_buffer = 0

        if current_buffer == 0:
            with quantum_state_re.map_to_host() as host_re, quantum_state_im.map_to_host() as host_im:
                for i in range(state_vector_size):
                    quantum_state[i] = ComplexFloat32(host_re[i], host_im[i])
        else:
            with quantum_state_out_re.map_to_host() as host_re, quantum_state_out_im.map_to_host() as host_im:
                for i in range(state_vector_size):
                    quantum_state[i] = ComplexFloat32(host_re[i], host_im[i])

        if self.verbose and self.verbose_step_size == ShowOnlyEnd:
            print("Final quantum state:\n", quantum_state)

        return quantum_state
