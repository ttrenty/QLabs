# TODOs

## Ordered by Priority (5 ↑) / Difficulty (5 ↓)

### Implementations

- 4 / 3 : Implement measurement gates

- 4 / 5 : Implement the computation of statistics (6.5 and 6.6)

- 3 / 3 : Implement naive implementation of the functions to compare performances
    - matrix multiplication (but starting from right or smart)
    - partial trace

- 5 / 5 : Start adding support for GPU in the base classes if needed (not possible to use SIMD(complexfloat64) anymore, or keep them but seperate them when moving data to GPU)
    - struct PureBasisState
    - struct ComplexMatrix
    - struct Gate

- 5 / ? : GPU implementation of:
    - qubit_wise_multiply()
    - apply_swap()
    - partial_trace()


### Tests

- 5 / 2 : Test qubit_wise_multiply_extended() that can take multiple qubits gates (2 and more, iSWAP for example)

- 5 / 2 : Test for everything that will be implement in GPU
    - qubit_wise_multiply()
    - apply_swap()
    - partial_trace()
    - struct PureBasisState's methods
    - struct ComplexMatrix's methods
    - struct Gate's Gate

### Benchmarks

- 3 / 2 : Reproduce table from page 10

## Droped for now

- 4 / 4 : Gradient computation with Finite Difference

- 3 / 2 :  Use a separate list for things that are not real gate to not slow down the main run logic

- 3 / 4 : Compile time circuit creation?

- 3 / 4 : Gradient computation with Parameter-Shift

- 3 / 100 : Gradient computation with Adjoint Method

- 2 / 4 : qubit_wise_multiply_extended() but for gates applied to non-adjacent qubits
