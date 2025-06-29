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
        min_number_qubits=1,
        max_number_qubits=20,
        number_qubits_step_size=1,
        min_number_layers=5,
        max_number_layers=3500,
        number_layers_step_size=400,
        fixed_number_qubits=13,
        fixed_number_layers=5,
    ]()

    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        bench_qubit_wise_multiply_inplace_gpu[
            min_number_qubits=1,
            max_number_qubits=26,  # 29 is OOM for my 3070 Ti Laptop GPU
            number_qubits_step_size=1,
            min_number_layers=5,
            max_number_layers=7000,
            number_layers_step_size=400,
            fixed_number_qubits=13,
            fixed_number_layers=5,
        ]()

    # bench_qubit_wise_multiply_extended()
    # bench_simulate_random_circuit()
    print("All benchmarks completed.")
