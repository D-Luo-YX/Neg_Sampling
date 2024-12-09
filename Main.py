import numpy as np
import networkx as nx
import random
import os
import torch
import matplotlib.pyplot as plt
import  Samples_Generated
from Models.GNN.GNN_model import GNN_model_Training

def set_random_seed(seed):
    np.random.seed(seed)  # 设置 numpy 随机数种子
    random.seed(seed)  # 设置 Python random 模块的随机数种子
    torch.manual_seed(seed)  # 设置 PyTorch 的随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置 CUDA 随机数种子
        torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机数种子
        torch.backends.cudnn.deterministic = True  # 确保 CUDA 算法确定性
        torch.backends.cudnn.benchmark = False  # 禁用优化以保证可复现性

def save_data_to_txt(filename, auc, acc, f1, auc_i, acc_i, f1_i, d_auc, d_acc, d_f1):
    with open(filename, 'w') as f:
        f.write(f"auc: {auc}\n")
        f.write(f"acc: {acc}\n")
        f.write(f"f1: {f1}\n")
        f.write(f"auc_i: {auc_i}\n")
        f.write(f"acc_i: {acc_i}\n")
        f.write(f"f1_i: {f1_i}\n")
        f.write(f"d_auc: {d_auc}\n")
        f.write(f"d_acc: {d_acc}\n")
        f.write(f"d_f1: {d_f1}\n")
    print(f"Data successfully saved to {filename}")

def calculate_stats(values):
    return min(values), max(values), sum(values) / len(values)

def ploted(auc, acc, f1):
    acc_stats = calculate_stats(auc)

    acc_min, acc_max, acc_avg = acc_stats

    fig, ax = plt.subplots(figsize=(6, 8))

    # Plot vertical line for ACC range
    x_pos = 1
    ax.vlines(x=x_pos, ymin=acc_min, ymax=acc_max, colors='blue', linestyles='--', label='ACC Range')

    # Add a rectangle to highlight the range
    rect_height = acc_max - acc_min
    rect_width = 0.2
    rect = plt.Rectangle((x_pos - rect_width / 2, acc_min), rect_width, rect_height, color='skyblue', alpha=0.4, label='Range (Min-Max)')
    ax.add_patch(rect)

    # Mark and annotate Min, Max, and Avg
    ax.scatter([x_pos], [acc_min], color='gray', label='Min', zorder=5)
    ax.scatter([x_pos], [acc_max], color='gray', label='Max', zorder=5)
    ax.scatter([x_pos], [acc_avg], color='blue', label='Avg', zorder=5)

    ax.text(x_pos + 0.1, acc_min, f'Min: {acc_min:.2f}', fontsize=10, va='center', color='red')
    ax.text(x_pos + 0.1, acc_max, f'Max: {acc_max:.2f}', fontsize=10, va='center', color='green')
    ax.text(x_pos + 0.1, acc_avg, f'Avg: {acc_avg:.2f}', fontsize=10, va='center', color='purple')

    # Customize the plot
    ax.set_xlim(0.5, 1.5)
    # ax.set_ylim(-1, acc_max + 0.1)  # Start y-axis at 0
    ax.set_ylim(-1,1)
    ax.set_xticks([])
    ax.set_ylabel('AUC-ROC Differences Values')
    ax.set_title('The difference between different models under a 1:1 positive and negative sample ratio')
    ax.legend()

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

seed = 1
set_random_seed(seed)

iteration = 10
auc, acc, f1, auc_i, acc_i, f1_i, d_auc, d_acc, d_f1 = [[] for _ in range(9)]
# auc = 0
# acc = 0
# fi  = 0
# auc_i = 0
# acc_i = 0
# fi_i = 0

# filename = 'USAir_unweighted'
files = ['PB','USAir_unweighted','Yeast','facebook']
# filename = 'PB'
for filename in files:
    Samples_Generated.file_processing(filename)
    for i in range(iteration):
        a, b, c, a_i, b_i, c_i = GNN_model_Training(filename,epochs=10,seed=seed)
        auc.append(a)
        acc.append(b)
        f1.append(c)
        auc_i.append(a_i)
        acc_i.append(b_i)
        f1_i.append(c_i)
        d_auc.append((a_i-a))
        d_acc.append((b_i-b))
        d_f1.append((c_i-c))
        seed += 1
    output_file = f'result/GNN/{filename}_ori_result.txt'
    save_data_to_txt(output_file, auc, acc, f1, auc_i, acc_i, f1_i, d_auc, d_acc, d_f1)

