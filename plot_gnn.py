import random
import numpy as np
import matplotlib.pyplot as plt


def read_metrics_from_file(file_path):
    """
    Read and parse metrics data from a file with specific format.

    Parameters:
        file_path (str): Path to the file containing metrics data.

    Returns:
        dict: Parsed metrics data as a dictionary.
    """
    metrics = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.split(':', 1)
            key = key.strip()
            value = eval(value.strip())  # Convert string list to Python list
            metrics[key] = value
    return metrics

def read_function(filename):
    metrics = read_metrics_from_file(f'result/GNN/result/{filename}_result.txt')
    return metrics

def calculate_auc_difference(metrics):
    """
    Calculate the difference between auc_i and auc.

    Parameters:
        metrics (dict): Dictionary containing metric lists, including 'auc' and 'auc_i'.

    Returns:
        list: Differences between auc_i and auc.
    """
    if 'auc' not in metrics or 'auc_i' not in metrics:
        raise KeyError("Metrics dictionary must contain 'auc' and 'auc_i'.")

    return [auc_i - auc for auc, auc_i in zip(metrics['auc'], metrics['auc_i'])]
def visualize_auc_differences(all_differences):
    """
    Visualize the differences between auc_i and auc for multiple datasets.

    Parameters:
        all_differences (dict): Dictionary containing differences for each dataset.
    """
    datasets = list(all_differences.keys())
    x_pos = np.arange(len(datasets))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, dataset in enumerate(datasets):
        differences = np.array(all_differences[dataset])

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
    ax.set_title("simple-GNN model AUC Differences Visualization", fontsize=14)
    ax.set_ylabel("AUC Differences", fontsize=12)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    files = ['facebook', 'PB', 'USAir', 'Yeast']
    all_differences = {}

    for filename in files:
        metrics = read_function(filename)
        differences = calculate_auc_difference(metrics)
        all_differences[filename] = differences

    for name, diff in all_differences.items():
        print(f"{name}: {diff}")

    visualize_auc_differences(all_differences)