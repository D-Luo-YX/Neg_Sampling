import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import networkx as nx
import os
from sklearn.metrics import f1_score, accuracy_score
import random

def set_random_seed(seed):
    np.random.seed(seed)  # 设置 numpy 随机数种子
    random.seed(seed)  # 设置 Python random 模块的随机数种子
    torch.manual_seed(seed)  # 设置 PyTorch 的随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 设置 CUDA 随机数种子
        torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机数种子
        torch.backends.cudnn.deterministic = True  # 确保 CUDA 算法确定性
        torch.backends.cudnn.benchmark = False  # 禁用优化以保证可复现性

class LinkPredictionDataset(Dataset):
    def __init__(self, file_path):
        self.data = []
        with open(file_path, 'r') as f:
            for line in f:
                node1, node2, label = map(int, line.strip().split())
                self.data.append((node1, node2, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCNModel, self).__init__()
        self.gcn1 = nn.Linear(input_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, adj):
        h = self.relu(self.gcn1(torch.mm(adj, x)))
        h = self.relu(self.gcn2(torch.mm(adj, h)))
        out = self.sigmoid(self.output(h))
        return out

def read_edges(file):
        edges, labels = [], []
        with open(file, 'r') as f:
            for line in f:
                node1, node2, label = map(int, line.strip().split())
                edges.append((node1, node2))
                labels.append(label)
        return edges, labels

def read_full_graph_edges(file):
    edges, labels = [], []
    with open(file, 'r') as f:
        for line in f:
            node1, node2 = map(int, line.strip().split())
            edges.append((node1, node2))
    return edges

def load_graphs(train_file, val_file, test_file, full_file):

    # 加载完整原图
    full_edges = read_full_graph_edges(full_file)
    full_graph = nx.Graph()
    full_graph.add_edges_from(full_edges)

    # 加载数据集
    train_edges, train_labels = read_edges(train_file)
    val_edges, val_labels = read_edges(val_file)
    test_edges, test_labels = read_edges(test_file)

    train_graph = nx.Graph()
    train_graph.add_edges_from([(u, v) for (u, v), l in zip(train_edges, train_labels) if l == 1])

    val_graph = train_graph.copy()
    val_graph.add_edges_from([(u, v) for (u, v), l in zip(val_edges, val_labels) if l == 1])

    test_graph = val_graph.copy()
    test_graph.add_edges_from([(u, v) for (u, v), l in zip(test_edges, test_labels) if l == 1])

    return full_graph, train_graph, val_graph, test_graph, train_edges, train_labels, val_edges, val_labels, test_edges, test_labels

def generate_graph_data(graph, full_graph, input_dim):
    num_nodes = max(full_graph.nodes) + 1  # 使用完整原图的节点数量
    # 确保图包含完整节点范围
    missing_nodes = set(range(num_nodes)) - set(graph.nodes)
    graph.add_nodes_from(missing_nodes)
    # 生成邻接矩阵和节点特征
    adj_matrix = nx.to_numpy_array(graph, nodelist=range(num_nodes))  # 确保矩阵大小一致
    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
    node_features = torch.rand((num_nodes, input_dim))  # 随机生成节点特征
    return adj_matrix, node_features

def train(model, optimizer, criterion, dataloader, adj_matrix, node_features):
    model.train()
    total_loss = 0
    for node1, node2, label in dataloader:
        node1, node2, label = node1.long(), node2.long(), label.float()
        optimizer.zero_grad()
        output = model(node_features, adj_matrix)
        scores = (output[node1] + output[node2]) / 2
        loss = criterion(scores, label.unsqueeze(1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, adj_matrix, node_features):
    model.eval()
    labels, preds = [], []
    with torch.no_grad():
        for node1, node2, label in dataloader:
            node1, node2, label = node1.long(), node2.long(), label.float()
            output = model(node_features, adj_matrix)
            scores = (output[node1] + output[node2]) / 2
            labels.extend(label.tolist())
            preds.extend(scores.squeeze().tolist())
    auc = roc_auc_score(labels, preds)
    fpr, tpr, thresholds = roc_curve(labels, preds)
    best_threshold = thresholds[np.argmax(tpr - fpr)]
    return auc, best_threshold

def evaluate_with_best_threshold(model, dataloader, adj_matrix, node_features, threshold):
    model.eval()
    labels, preds = [], []
    with torch.no_grad():
        for node1, node2, label in dataloader:
            node1, node2, label = node1.long(), node2.long(), label.float()
            output = model(node_features, adj_matrix)
            scores = (output[node1] + output[node2]) / 2
            labels.extend(label.tolist())
            preds.extend(scores.squeeze().tolist())
    # 转为二值化结果
    preds_binary = [1 if p >= threshold else 0 for p in preds]
    return labels, preds, preds_binary

def training_model(train_file, val_file, test_file, full_file, epochs=50, save_model=False, model_path="best_model.pth"):
    # 加载图
    full_graph, train_graph, val_graph, test_graph, train_edges, train_labels, val_edges, val_labels, test_edges, test_labels = load_graphs(
        train_file, val_file, test_file, full_file)

    input_dim = 8  # 节点特征维度
    hidden_dim = 32  # 隐藏层维度

    # 生成图数据
    train_adj, train_features = generate_graph_data(train_graph, full_graph, input_dim)
    val_adj, val_features = generate_graph_data(val_graph, full_graph, input_dim)
    test_adj, test_features = generate_graph_data(test_graph, full_graph, input_dim)
    # 数据加载器
    train_loader = DataLoader(LinkPredictionDataset(train_file), batch_size=32, shuffle=True)
    val_loader = DataLoader(LinkPredictionDataset(val_file), batch_size=32, shuffle=False)
    test_loader = DataLoader(LinkPredictionDataset(test_file), batch_size=32, shuffle=False)
    # 模型、优化器和损失函数
    model = GCNModel(input_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss()

    best_auc = 0
    best_epoch = 0
    best_threshold = 0
    # 训练和验证
    for epoch in range(epochs):
        train_loss = train(model, optimizer, criterion, train_loader, train_adj, train_features)
        val_auc, threshold = evaluate(model, val_loader, val_adj, val_features)

        # print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_threshold = threshold
            best_epoch = epoch
            if save_model:
                torch.save(model.state_dict(), model_path)

    print(f"Best Val AUC: {best_auc:.4f} at Epoch {best_epoch + 1} with Threshold {best_threshold:.4f}")

    # 测试
    if save_model:
        model.load_state_dict(torch.load(model_path))

    test_labels, test_preds, test_preds_binary = evaluate_with_best_threshold(
        model, test_loader, test_adj, test_features, best_threshold)

    test_auc = roc_auc_score(test_labels, test_preds)

    test_accuracy = accuracy_score(test_labels, test_preds_binary)
    test_f1 = f1_score(test_labels, test_preds_binary)

    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    return test_auc, test_accuracy, test_f1

def GNN_model_Training(filename, epochs = 3, seed=1):

    set_random_seed(seed)

    train_file = f"processing_data/{filename}/train.txt"
    val_file = f"processing_data/{filename}/val.txt"
    test_file = f"processing_data/{filename}/test.txt"
    full_file = f"original_graphs/{filename}.txt"

    auc, acc, fi = training_model(train_file, val_file, test_file, full_file, epochs = epochs, save_model=False,
                   model_path="best_gcn_model.pth")

    set_random_seed(seed)

    train_file_i = f"processing_data/{filename}/train_i.txt"
    val_file_i = f"processing_data/{filename}/val_i.txt"
    test_file_i = f"processing_data/{filename}/test_i.txt"
    full_file = f"original_graphs/{filename}.txt"

    auc_i, acc_i ,fi_i = training_model(train_file_i, val_file_i, test_file_i, full_file, epochs = epochs, save_model=False,
                   model_path="best_gcn_model_i.pth")

    return auc, acc, fi, auc_i, acc_i, fi_i

if __name__ == "__main__":
    current_dir = os.getcwd()

    new_dir = os.path.abspath(os.path.join(current_dir, "../../"))
    os.chdir(new_dir)
    print("Current working directory:", os.getcwd())

    filename = 'facebook'
    GNN_model_Training(filename)