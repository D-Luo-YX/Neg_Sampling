import matplotlib.pyplot as plt
import numpy as np

# Load data from text files
def load_auc_data(file_path):
    """
    Load AUC data from a text file, returning a dictionary.
    Each key is a dataset name, and the value is a list of AUC values.
    """
    data = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        dataset = None
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            if line.startswith("Dataset:"):
                datasets = eval(line.split(":")[1].strip())  # Parse dataset names
                continue
            elif line.endswith("AUC Ideal:") or line.endswith("AUC Not Ideal:"):
                dataset = line.split(" ")[0]
                data[dataset] = []
            elif dataset:
                try:
                    data[dataset].append(float(line))
                except ValueError:
                    print(f"Skipping invalid line: {line}")
    return data

# Compute average difference between ideal and not ideal
def compute_avg_diff(ideal_data, not_ideal_data):
    """
    Compute the average difference between ideal and not ideal results.
    """
    datasets = ideal_data.keys()
    avg_diff = {dataset: np.mean(np.array(ideal_data[dataset]) - np.array(not_ideal_data[dataset]))
                for dataset in datasets}
    return avg_diff

# Visualization function
def visualize_auc_differences(ideal_data, not_ideal_data):
    """
    Visualize the differences between ideal and not ideal AUC values.
    """
    datasets = list(ideal_data.keys())
    x_pos = np.arange(len(datasets))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, dataset in enumerate(datasets):
        # Calculate differences
        ideal_values = np.array(ideal_data[dataset])
        not_ideal_values = np.array(not_ideal_data[dataset])
        differences = ideal_values - not_ideal_values

        # Calculate Min, Max, and Avg for differences
        diff_min, diff_max, diff_avg = differences.min(), differences.max(), differences.mean()

        # Add rectangle to highlight the range (Min-Max)
        rect_height = diff_max - diff_min
        rect = plt.Rectangle((x_pos[i] - width / 2, diff_min), width, rect_height,
                             color='skyblue', alpha=0.4, label='Range (Min-Max)' if i == 0 else "")
        ax.add_patch(rect)

        # Mark and annotate Min, Max, and Avg
        ax.scatter([x_pos[i]], [diff_min], color='gray', label='Min' if i == 0 else "", zorder=5)
        ax.scatter([x_pos[i]], [diff_max], color='gray', label='Max' if i == 0 else "", zorder=5)
        ax.scatter([x_pos[i]], [diff_avg], color='blue', label='Avg' if i == 0 else "", zorder=5)

        ax.text(x_pos[i] + 0.1, diff_min, f'Min: {diff_min:.4f}', fontsize=9, va='center', color='red')
        ax.text(x_pos[i] + 0.1, diff_max, f'Max: {diff_max:.4f}', fontsize=9, va='center', color='green')
        ax.text(x_pos[i] + 0.1, diff_avg, f'Avg: {diff_avg:.4f}', fontsize=9, va='center', color='purple')

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.set_title("NNESF: AUC Differences (Ideal - Not Ideal)", fontsize=14)
    ax.set_ylabel("AUC Differences", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ideal_file = "result/auc_ideal.txt"  # Replace with your local ideal file path
    not_ideal_file = "result/auc_not_ideal.txt"  # Replace with your local not_ideal file path

    # Load data
    ideal_data = load_auc_data(ideal_file)
    not_ideal_data = load_auc_data(not_ideal_file)
    print(ideal_data)
    print(not_ideal_data)
    # Ensure both files contain the same datasets
    assert set(ideal_data.keys()) == set(not_ideal_data.keys()), "Datasets in the two files do not match!"

    # Compute average differences
    avg_diff = compute_avg_diff(ideal_data, not_ideal_data)
    print(avg_diff)
    # Visualize results
    visualize_auc_differences(ideal_data, not_ideal_data)