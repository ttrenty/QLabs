# QLabs

**A Quantum Circuit Composer & Simulator in Mojo** 🔥⚛️

This project reimplements and extends the ideas from the following tutorial paper:

> **How to Write a Simulator for Quantum Circuits from Scratch: A Tutorial**  
> *Michael J. McGuffin, Jean-Marc Robert, and Kazuki Ikeda*  
> Published: 2025-06-09 on [arXiv:2506.08142v1](https://arxiv.org/abs/2506.08142v1) (last accessed: 2025-06-12)


## 🎯 Project Objectives

* **Mojo Implementation:** Re-implement the approach from the paper in Mojo for more Pythonic synthax and better readability.
* **Learning by Doing:** Gain hands-on experience with quantum circuit simulation to better understand the capabilities and limitations of classical simulation.
* **Performance & Safety:** Leverage Mojo's strong static typing and compilation for blazing-fast and safe operations.
* **Hardware Acceleration:** Utilize Mojo’s universal GPU programming support to accelerate simulations.


## ⚙️ Environment Setup

Follow these steps to set up your environment and build the binary:

```bash
# If you don't have Pixi installed yet:
curl -sSf https://pixi.sh/install.sh | bash

# Install all project dependencies:
pixi install

# Build and run the simulator:
pixi run mojo build src/main.mojo && ./main
```


## 🔥 Current Implementation(s)

The current implemtnation uses State Vector which is an efficient method for simulating
with high precision smaller scale qubits system of 20-30 qubits. It also allow for 
easily accessible gradient computations.

Another possible implemenation would be using Tensor Network, this method is effective 
for large circuits, however with less precision in the calculations and with more
expensive gradients computations.

## 📄 License

This project is open-source and licensed under Apache License 2.0.
