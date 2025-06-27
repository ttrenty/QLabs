# Mojo Implementation of Reference

**Title:** *How to Write a Simulator for Quantum Circuits from Scratch: A Tutorial*  
**Authors:** Michael J. McGuffin, Jean-Marc Robert, and Kazuki Ikeda  
**Date:** 2025-06-09  

**Abstract**

> This tutorial guides a competent programmer through the crafting of a quantum circuit simulator from scratch, even for readers with almost no prior experience in quantum computing. Open source simulators for quantum circuits already exist, but a deeper understanding is gained by writing ones own. With roughly 1000-2000 lines of code, one can simulate Hadamard, Pauli X, Y, Z, SWAP, and other quantum logic gates, with arbitrary combinations of control and anticontrol qubits, on circuits of up to 20+ qubits, with no special libraries, on a personal computer. We explain key algorithms for a simulator, in particular: qubit-wise multiplication for updating the state vector, and partial trace for finding a reduced density matrix. We also discuss optimizations, and how to compute qubit phase, purity, and other statistics. A complete example implementation in JavaScript is available at this https URL , which also demonstrates how to compute von Neumann entropy, concurrence (to quantify entanglement), and magic, while remaining much smaller and easier to study than other popular software packages. 

**URL:** [https://arxiv.org/abs/2506.08142v1](https://arxiv.org/abs/2506.08142v1) (accessed 2025-06-12)

---

## Conventions

* **Qubit ordering**
  Qubits are labeled **0 … n − 1** from **top → bottom**.

  * Qubit 0 (top line) is the **least-significant bit (LSB)**.
  * Qubit n − 1 (bottom line) is the **most-significant bit (MSB)**.

* **State-vector notation**
  Basis states are written as
  \| q<sub>n-1</sub> … q<sub>1</sub> q<sub>0</sub> ⟩.

* **Tensor-product layout for gate layers**
  When assembling a layer of single-qubit gates, tensors are taken **right-to-left**.

  * Example: a Pauli-X on qubit 0 in an *n*-qubit circuit is
    $I ⊗ I ⊗ ··· ⊗ I ⊗ X$
    (with *n − 1* identity gates followed by **X**).

* This little-endian ordering matches the default of many software frameworks (Qiskit, Cirq, Braket, …), but it is the **opposite** of the big-endian convention common in theoretical physics.
