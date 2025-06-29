from sys import has_accelerator

from bench_simulate_random_circuit import bench_simulate_random_circuit
from bench_qubit_wise_multiply import (
    bench_qubit_wise_multiply,
    bench_qubit_wise_multiply_inplace,
    bench_qubit_wise_multiply_inplace_gpu,
    bench_qubit_wise_multiply_extended,
)


def main():
    print("Running all benchmarks...")
    # bench_qubit_wise_multiply()
    # bench_qubit_wise_multiply_inplace()

    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        bench_qubit_wise_multiply_inplace_gpu()

    # bench_qubit_wise_multiply_extended()
    # bench_simulate_random_circuit()
    print("All benchmarks completed.")
