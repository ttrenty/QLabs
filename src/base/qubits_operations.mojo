# << for shift left
# | for bitwise OR
# ^ for bitwise XOR
# & for bitwise AND
# ~ for bitwise NOT

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Imports              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

from bit import count_trailing_zeros

from .state_and_matrix import (
    PureBasisState,
    ComplexMatrix,
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Functions            #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


@always_inline
fn qubit_wise_multiply_extended(
    target_qubits_count: Int,
    gate: ComplexMatrix,
    target_qubits: List[Int],
    owned quantum_state: PureBasisState,
    control_bits: List[List[Int]] = [],
) -> __type_of(quantum_state):
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
        return qubit_wise_multiply(
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
        if (
            target_qubits[i] + 1 != target_qubits[i + 1]
        ):  # TODO add the instruction to the
            # user that they have to provide qubits in increasing order.
            adjacent_qubits = False
            break
    if adjacent_qubits:
        return qubit_wise_multiply(
            gate,
            target_qubits[0],
            quantum_state,
            control_bits,
        )
    else:
        # TODO apply swap gates properly to re-organise the qubits
        return quantum_state


@always_inline
fn _apply_to_multi_qubits(
    mut new_state_vector: PureBasisState,
    gate: ComplexMatrix,
    quantum_state: PureBasisState,
    size_of_state_vector: Int,
    size_of_block: Int,
    size_of_half_block: Int,
    inclusion_mask: Int,
    desired_value_mask: Int,
):
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
                        new_state_vector[indexes[i]] = (
                            gate[i, j] * quantum_state[indexes[j]]
                        )
                    else:
                        new_state_vector[indexes[i]] += (
                            gate[i, j] * quantum_state[indexes[j]]
                        )

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
fn _apply_to_2_qubit(
    mut new_state_vector: PureBasisState,
    gate: ComplexMatrix,
    quantum_state: PureBasisState,
    size_of_state_vector: Int,
    size_of_block: Int,
    size_of_half_block: Int,
    inclusion_mask: Int,
    desired_value_mask: Int,
):
    for block_start in range(0, size_of_state_vector, size_of_block):
        for offset in range(size_of_half_block):
            i1: Int = block_start | offset

            if (i1 & inclusion_mask) != desired_value_mask:
                continue

            i2: Int = i1 | size_of_half_block
            i3: Int = i2 | size_of_half_block

            new_state_vector[i1] = (
                gate[0, 0] * quantum_state[i1]
                + gate[0, 1] * quantum_state[i2]
                + gate[0, 2] * quantum_state[i3]
            )
            new_state_vector[i2] = (
                gate[1, 0] * quantum_state[i1]
                + gate[1, 1] * quantum_state[i2]
                + gate[1, 2] * quantum_state[i3]
            )
            new_state_vector[i3] = (
                gate[2, 0] * quantum_state[i1]
                + gate[2, 1] * quantum_state[i2]
                + gate[2, 2] * quantum_state[i3]
            )


@always_inline
fn _apply_to_1_qubit(
    mut new_state_vector: PureBasisState,
    gate: ComplexMatrix,
    quantum_state: PureBasisState,
    size_of_state_vector: Int,
    size_of_block: Int,
    size_of_half_block: Int,
    inclusion_mask: Int,
    desired_value_mask: Int,
):
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


fn qubit_wise_multiply(
    gate: ComplexMatrix,
    target_qubit: Int,
    owned quantum_state: PureBasisState,
    control_bits: List[List[Int]] = [],
) -> __type_of(quantum_state):
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
        print(
            "Error: target_qubit index out of bounds. Must be between 0 and",
            quantum_state.number_qubits() - 1,
        )
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
    new_state_vector = quantum_state  # copies all amplitudes from quantum_state to new_state_vector

    if target_qubits_count == 1:
        _apply_to_1_qubit(
            new_state_vector,
            gate,
            quantum_state,
            size_of_state_vector,
            size_of_block,
            size_of_half_block,
            inclusion_mask,
            desired_value_mask,
        )
    elif target_qubits_count == 2:
        _apply_to_2_qubit(
            new_state_vector,
            gate,
            quantum_state,
            size_of_state_vector,
            size_of_block,
            size_of_half_block,
            inclusion_mask,
            desired_value_mask,
        )
    else:
        _apply_to_multi_qubits(
            new_state_vector,
            gate,
            quantum_state,
            size_of_state_vector,
            size_of_block,
            size_of_half_block,
            inclusion_mask,
            desired_value_mask,
        )

    return new_state_vector


# fn invert_gate_endian() #TODO


fn _swap_bits(
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


fn educational_apply_swap(
    num_qubits: Int,
    i: Int,
    j: Int,
    quantum_state: PureBasisState,
    control_bits: List[List[Int]] = [],
) -> __type_of(quantum_state):
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

    new_state_vector = quantum_state  # copies all amplitudes from quantum_state to new_state_vector
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
        swapped_k: Int = _swap_bits(k, i, j)

        # Update the state vector
        if swapped_k > k:  # this check ensures we don’t swap each pair twice
            # swap amplitudes
            new_state_vector[k] = quantum_state[swapped_k]
            new_state_vector[swapped_k] = quantum_state[k]

    return new_state_vector


fn apply_swap(
    num_qubits: Int,
    i: Int,
    j: Int,
    quantum_state: PureBasisState,
    control_bits: List[List[Int]] = [],
) -> __type_of(quantum_state):
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
    new_state_vector = quantum_state  # copies all amplitudes from quantum_state to new_state_vector
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


fn _rearrange_bits(
    value: Int,
    new_indexes: List[Int],
) -> Int:
    """Rearranges the bits of the integer i according to the indices in a.

    Args:
        value: The integer whose bits are to be rearranged.
        new_indexes: A list of new positions for the bits.

    Examples:
    ```mojo
    _rearrange_bits(i,[1,0])  # Returns the two least-significant bits of i,
                            # swapped, and none of the other bits.

    _rearrange_bits(i,[0,1,2])    # Returns only the three least-significant bits of i,
                                # with their positions unchanged.

    _rearrange_bits(i,[3,0,1,2])  # Returns only the four least-significant bits of i,
                                # shifted left (to one position more significant) and
                                # wrapped around.
    ```
    Returns:
        The integer value with its bits rearranged according to the indices in new_indexes.
    """
    return_value: Int = 0
    for position in range(len(new_indexes)):
        if new_indexes[position] >= 0:
            return_value |= ((value >> position) & 1) << new_indexes[position]
    return return_value


fn educational_partial_trace[
    use_lookup_table: Bool = True
](
    n: Int,
    input_matrix: ComplexMatrix,
    qubits_to_trace_out: List[Int],
) -> ComplexMatrix:
    """Performs a partial trace over specified qubits in a quantum state.

    Args:
        n: The total number of qubits in the circuit.
        input_matrix: A (2^n)×(2^n) matrix of complex numbers representing the quantum state.
        qubits_to_trace_out: An array of indices of qubits to trace out, in ascending order
                            and without duplicates.

    Returns:
        A (2^(n-T))×(2^(n-T)) matrix where T is the number of traced-out qubits.
    """
    is_traced_out: List[Bool] = [False] * n
    for i in range(len(qubits_to_trace_out)):
        is_traced_out[qubits_to_trace_out[i]] = True

    qubits_to_keep: List[Int] = []
    for i in range(n):
        if not is_traced_out[i]:
            qubits_to_keep.append(i)

    # is_traced_out: List[Bool] = [False] * n
    # qubits_to_keep: List[Int] = []

    # current_index: Int = 0
    # current_traced_index: Int = 0
    # for _ in range(n):
    #     if qubits_to_trace_out[current_traced_index] == current_index:
    #         is_traced_out[current_index] = True
    #         current_traced_index += 1
    #         if current_traced_index >= len(qubits_to_trace_out):
    #             break
    #     else:
    #         qubits_to_keep.append(i)
    #     current_index += 1

    # # Ensure all qubits have been added to qubits_to_keep
    # for i in range(current_index, n):
    #     qubits_to_keep.append(i)

    num_qubits_to_trace_out: Int = len(qubits_to_trace_out)
    num_qubits_to_keep: Int = len(qubits_to_keep)
    if num_qubits_to_trace_out + num_qubits_to_keep != n:
        print(
            "Error: The total number of qubits to trace out and keep does not"
            " match the total number of qubits."
        )
        return ComplexMatrix(0, 0)  # Return an empty matrix

    # This is 2^num_qubits_to_trace_out == the dimension of the space being traced out
    traced_dimension: Int = 1 << num_qubits_to_trace_out
    # This is 2^num_qubits_to_keep == the dimension of the resulting matrix
    result_dimension: Int = 1 << num_qubits_to_keep

    lookup_table: Dict[Int, Int] = {}

    @parameter
    if use_lookup_table:
        for tmp in range(result_dimension):
            lookup_table[tmp] = _rearrange_bits(tmp, qubits_to_keep)

    output_matrix: ComplexMatrix = ComplexMatrix(
        result_dimension, result_dimension
    )
    for shared_bits in range(
        traced_dimension
    ):  # bits common to input_row and input_col

        @parameter
        if use_lookup_table:
            shared_bits_rearranged: Int = lookup_table.get(shared_bits, 0)
        else:
            shared_bits_rearranged: Int = _rearrange_bits(
                shared_bits, qubits_to_trace_out
            )

        for output_row in range(result_dimension):

            @parameter
            if use_lookup_table:
                input_row: Int = lookup_table.get(output_row, 0)
            else:
                input_row: Int = shared_bits_rearranged | _rearrange_bits(
                    output_row, qubits_to_keep
                )
            for output_col in range(result_dimension):

                @parameter
                if use_lookup_table:
                    input_col: Int = lookup_table.get(output_col, 0)
                else:
                    input_col: Int = shared_bits_rearranged | _rearrange_bits(
                        output_col, qubits_to_keep
                    )

                output_matrix[output_row, output_col] += input_matrix[
                    input_row, input_col
                ]

    return output_matrix


fn partial_trace[
    use_lookup_table: Bool = True
](
    quantum_state: PureBasisState,
    qubits_to_trace_out: List[Int],
) -> ComplexMatrix:
    """Performs a partial trace over specified qubits in a quantum state.

    Args:
        quantum_state: A PureBasisState representing the quantum state.
        qubits_to_trace_out: An array of indices of qubits to trace out, in ascending order
                            and without duplicates.

    Returns:
        A (2^(n-T))×(2^(n-T)) matrix where T is the number of traced-out qubits.
    """
    n = quantum_state.number_qubits()
    conj_quantum_state = quantum_state.conjugate()

    # is_traced_out: List[Bool] = [False] * n
    # for i in range(len(qubits_to_trace_out)):
    #     is_traced_out[qubits_to_trace_out[i]] = True

    # qubits_to_keep: List[Int] = []
    # for i in range(n):
    #     if not is_traced_out[i]:
    #         qubits_to_keep.append(i)

    is_traced_out: List[Bool] = [False] * n
    qubits_to_keep: List[Int] = []

    current_index: Int = 0
    current_traced_index: Int = 0
    for _ in range(n):
        if current_traced_index >= len(qubits_to_trace_out):
            break
        if qubits_to_trace_out[current_traced_index] == current_index:
            is_traced_out[current_index] = True
            current_traced_index += 1
            current_index += 1
            if current_traced_index >= len(qubits_to_trace_out):
                break
        else:
            qubits_to_keep.append(current_index)
            current_index += 1

    # Ensure all qubits have been added to qubits_to_keep
    for i in range(current_index, n):
        qubits_to_keep.append(i)

    num_qubits_to_trace_out: Int = len(qubits_to_trace_out)
    num_qubits_to_keep: Int = len(qubits_to_keep)
    if num_qubits_to_trace_out + num_qubits_to_keep != n:
        print(
            "Error: The total number of qubits to trace out (",
            num_qubits_to_trace_out,
            ") and keep (",
            num_qubits_to_keep,
            ") does not match the total number of qubits (",
            n,
            ").",
        )
        return ComplexMatrix(0, 0)  # Return an empty matrix

    # This is 2^num_qubits_to_trace_out == the dimension of the space being traced out
    traced_dimension: Int = 1 << num_qubits_to_trace_out
    # This is 2^num_qubits_to_keep == the dimension of the resulting matrix
    result_dimension: Int = 1 << num_qubits_to_keep

    lookup_table: Dict[Int, Int] = {}

    @parameter
    if use_lookup_table:
        for tmp in range(result_dimension):
            lookup_table[tmp] = _rearrange_bits(tmp, qubits_to_keep)

    output_matrix: ComplexMatrix = ComplexMatrix(
        result_dimension, result_dimension
    )
    for shared_bits in range(
        traced_dimension
    ):  # bits common to input_row and input_col

        @parameter
        if use_lookup_table:
            shared_bits_rearranged: Int = lookup_table.get(shared_bits, 0)
        else:
            shared_bits_rearranged: Int = _rearrange_bits(
                shared_bits, qubits_to_trace_out
            )

        for output_row in range(result_dimension):

            @parameter
            if use_lookup_table:
                input_row: Int = lookup_table.get(output_row, 0)
            else:
                input_row: Int = shared_bits_rearranged | _rearrange_bits(
                    output_row, qubits_to_keep
                )
            for output_col in range(result_dimension):

                @parameter
                if use_lookup_table:
                    input_col: Int = lookup_table.get(output_col, 0)
                else:
                    input_col: Int = shared_bits_rearranged | _rearrange_bits(
                        output_col, qubits_to_keep
                    )

                output_matrix[output_row, output_col] += (
                    quantum_state[input_row] * conj_quantum_state[input_col]
                )

    return output_matrix
