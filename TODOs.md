# TODOs

## Ordered by Priority (5 ↑) / Difficulty (5 ↓)

### Implementations

- 5 / 4 : GPU implementation of:
    - qubit_wise_multiply() (with different type of control gates and for multiple qubits)
    - apply_swap()
    - partial_trace()
    - StateVector.to_density_matrix()

- 2 / 4 : Efficient support for tracking a state statistic like entropy during the execution of the circuit by the simulator.

### Tests

- 5 / 2 : Test qubit_wise_multiply_extended() that can take multiple qubits gates (2 and more, iSWAP for example)

- 5 / 2 : Test for everything that will be implement in GPU
    - qubit_wise_multiply() (with different type of control gates and for multiple qubits)
    - apply_swap()
    - partial_trace()

### Benchmarks

- 3 / 2 : partial_trace() Reproduce table from page 10

## Droped for now

- 4 / 3 : Implement end of circuit measurement gates with some of those options:
    - https://docs.pennylane.ai/en/stable/introduction/measurements.html

- 4 / 4 : Gradient computation with Finite Difference

- 3 / 2 :  Use a separate list for things that are not real gate to not slow down the main run logic

- 3 / 3 : Setup automatic Doc generation with pixi but also on github.io repository page

- 3 / 4 : Compile time circuit creation?

- 3 / 4 : Gradient computation with Parameter-Shift

- 3 / 4 : Implement mid circuit measurement gates (Section 7 of paper)

- 3 / 100 : Gradient computation with Adjoint Method

- 2 / 4 : qubit_wise_multiply_extended() but for gates applied to non-adjacent qubits

- 2 / 3 : Implement concurence (2-qubits entanglement metric) computePairwiseQubitConcurrences()

- 1 / 3 : Implement naive implementation of the functions to compare performances
    - matrix multiplication (but starting from right or smart)
    - partial trace