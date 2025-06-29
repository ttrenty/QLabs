from sys import has_accelerator

from bench_simulate_random_circuit import bench_simulate_random_circuit
from bench_qubit_wise_multiply import (
    bench_qubit_wise_multiply,
    bench_qubit_wise_multiply_inplace,
    bench_qubit_wise_multiply_extended,
)
from bench_qubit_wise_multiply_gpu import (
    bench_qubit_wise_multiply_inplace_gpu,
)


def main():
    print("Running all benchmarks...")
    # bench_qubit_wise_multiply()
    bench_qubit_wise_multiply_inplace[
        min_number_qubits=5,
        max_number_qubits=25,
        number_qubits_step_size=2,
        min_number_layers=5,
        max_number_layers=4000,
        number_layers_step_size=400,
        fixed_number_qubits=11,
        fixed_number_layers=20,
    ]()

    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        bench_qubit_wise_multiply_inplace[
            min_number_qubits=5,
            max_number_qubits=25,
            number_qubits_step_size=2,
            min_number_layers=5,
            max_number_layers=4000,
            number_layers_step_size=400,
            fixed_number_qubits=11,
            fixed_number_layers=20,
        ]()

    # bench_qubit_wise_multiply_extended()
    # bench_simulate_random_circuit()
    print("All benchmarks completed.")
