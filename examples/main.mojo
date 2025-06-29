# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Imports              #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

from low_level import (
    simulate_figure1_circuit,
    simulate_figure1_circuit_inplace,
    simulate_figure4_circuit,
)

from gpu_low_level import (
    simulate_figure1_circuit_gpu,
)

from circuit_level import (
    simulate_figure1_circuit_abstract,
    simulate_figure4_circuit_abstract,
    simulate_random_circuit,
    try_density_matrix,
    try_get_purity,
)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# MARK:         Main                 #
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def main():
    print("QLabs: Quantum Circuit Composer & Simulator in Mojo")

    print("=========================================")
    print("low_level: simulate_figure1_circuit()")
    simulate_figure1_circuit()

    print("=========================================")
    print("low_level: simulate_figure1_circuit_inplace()")
    simulate_figure1_circuit_inplace()

    print("=========================================")
    print("low_level: simulate_figure4_circuit()")
    simulate_figure4_circuit()

    print("=========================================")
    print("circuit_level: simulate_figure1_circuit_abstract()")
    simulate_figure1_circuit_abstract()

    print("=========================================")
    print("circuit_level: simulate_random_circuit()")
    simulate_random_circuit[number_qubits=20](number_layers=5)

    print("=========================================")
    print("circuit_level: simulate_figure4_circuit_abstract()")
    simulate_figure4_circuit_abstract()

    print("=========================================")
    print("circuit_level: try_density_matrix()")
    try_density_matrix()

    print("=========================================")
    print("circuit_level: try_get_purity()")
    try_get_purity()

    print("=========================================")
    print("gpu_low_level: simulate_figure1_circuit_gpu()")
    simulate_figure1_circuit_gpu[3]()
