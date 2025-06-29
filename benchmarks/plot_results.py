import pandas as pd
import matplotlib.pyplot as plt

# --- 2. Data Loading and Parsing ---


def process_benchmark_data(filepath):
    """
    Reads a benchmark CSV, extracts qubit and layer counts from the 'name'
    column, and returns a clean, sorted DataFrame.
    """
    # Read the CSV file
    df = pd.read_csv(filepath)

    # Rename column for easier access (removes space and parentheses)
    df = df.rename(columns={"met (ms)": "time_ms"})

    # Use regular expressions to extract numbers of qubits and layers
    # '(\d+)q' finds a sequence of digits followed by 'q'
    # '(\d+)l' finds a sequence of digits followed by 'l'
    df["qubits"] = df["name"].str.extract(r"(\d+)q").astype(int)
    df["layers"] = df["name"].str.extract(r"(\d+)l").astype(int)

    # Sort values for correct line plotting
    if "layers" in filepath:
        df = df.sort_values("layers")
    elif "qubits" in filepath:
        df = df.sort_values("qubits")
    return df


# Load and process all four data files
layers_gpu_df = process_benchmark_data(
    "data/qubit_wise_multiply_inplace_gpu_layers.csv"
)
qubits_gpu_df = process_benchmark_data(
    "data/qubit_wise_multiply_inplace_gpu_qubits.csv"
)
layers_cpu_df = process_benchmark_data("data/qubit_wise_multiply_inplace_layers.csv")
qubits_cpu_df = process_benchmark_data("data/qubit_wise_multiply_inplace_qubits.csv")

# --- 3. Plotting ---

# Create a figure with two subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Qubit-wise Multiplication Benchmark", fontsize=16)

# Plot 1: Performance vs. Number of Layers
ax1.plot(
    layers_cpu_df["layers"] * layers_cpu_df["qubits"][0],
    layers_cpu_df["time_ms"],
    marker="o",
    linestyle="-",
    label="CPU",
)
ax1.plot(
    layers_gpu_df["layers"]
    * layers_cpu_df["qubits"][0],  # Scale x-axis by number of qubits
    layers_gpu_df["time_ms"],
    marker="s",
    linestyle="--",
    label="GPU",
)
ax1.set_title(
    f"Execution Time vs. Number of Layers\n({layers_cpu_df['qubits'][0]} Qubits)"
)
ax1.set_xlabel(
    f"Number of Gates\n(Number of Layers x {layers_cpu_df['qubits'][0]} Qubits)"
)
ax1.set_ylabel("Mean Execution Time (ms)")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.6)

# Plot 2: Performance vs. Number of Qubits
ax2.plot(
    qubits_cpu_df["qubits"],
    qubits_cpu_df["time_ms"],
    marker="o",
    linestyle="-",
    label="CPU",
)
ax2.plot(
    qubits_gpu_df["qubits"],
    qubits_gpu_df["time_ms"],
    marker="s",
    linestyle="--",
    label="GPU",
)
ax2.set_title(
    f"Execution Time vs. Number of Qubits\n({qubits_cpu_df['layers'][0]} Layers)"
)
ax2.set_xlabel("Number of Qubits")
# We can make the y-axis a log scale if the values vary widely
ax2.set_ylabel("Mean Execution Time (ms) - Log Scale")
ax2.set_yscale("log")  # Use a logarithmic scale to better see the differences
ax2.legend()
ax2.grid(True, which="both", linestyle="--", alpha=0.6)
# set the x-ticks to be the x-values of the qubits
unique_cpu_qubits = qubits_cpu_df["qubits"].unique()
unique_gpu_qubits = qubits_gpu_df["qubits"].unique()
unique_qubits = sorted(set(unique_cpu_qubits) | set(unique_gpu_qubits))
ax2.set_xticks(unique_qubits)
ax2.set_xticklabels(unique_qubits, rotation=45)


# Adjust layout to prevent labels from overlapping
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust rect to make space for suptitle

# --- 4. Saving and Displaying ---
pdf_filename = "data/benchmark_results.pdf"
plt.savefig(pdf_filename, bbox_inches="tight")

print(f"\nPlot successfully saved as '{pdf_filename}'")

# # Display the plot on the screen
# plt.show()
