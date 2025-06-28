# QLabs

**A Quantum Circuit Composer & Simulator in Mojo** 🔥⚛️


## Education 

This project reimplements and extends the ideas from the following tutorial paper:

> **How to Write a Simulator for Quantum Circuits from Scratch: A Tutorial**  
> *Michael J. McGuffin, Jean-Marc Robert, and Kazuki Ikeda*  
> Published: 2025-06-09 on [arXiv:2506.08142v1](https://arxiv.org/abs/2506.08142v1) (last accessed: 2025-06-12)

###  🎯 Project Objectives

* **Mojo Implementation:** Re-implement the approach from the paper in Mojo for more Pythonic synthax and better readability.
* **Learning by Doing:** Gain hands-on experience with quantum circuit simulation to better understand the capabilities and limitations of classical simulation.
* **Performance & Safety:** Leverage Mojo's strong static typing and compilation for blazing-fast and safe operations.
* **Hardware Acceleration:** Utilize Mojo’s universal GPU programming support to accelerate simulations.

### 🔥 Current Implementation

The current implementation uses a State Vector approach, which is an efficient method for simulating small-scale quantum circuits (20–30 qubits) with high precision. This approach also enables relatively straightforward exact gradient computations.

An alternative implementation for the futur could be using the Tensor Network approach. This method is more suitable for larger circuits but offers lower precision and would involves more computationally expensive gradient calculations.

## Usage

### ⚙️ Environment Setup

Follow these steps to set up your environment, build the library and run some examples:

If you don't have Pixi installed yet:
```bash
curl -sSf https://pixi.sh/install.sh | bash
```
Install all project dependencies:
```
pixi install
```

Build and run examples of the simulator:
```bash
pixi run main
```


## 📄 License

This project is open-source and licensed under Apache License 2.0.
