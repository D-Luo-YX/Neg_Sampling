# DGCNN部分

import torch
import torch.nn as nn
import torch.nn.functional as F


class DGCNN(nn.Module):
    def __init__(self, output_dim, num_node_feats, num_edge_feats, latent_dim=[32, 32, 32, 1], k=30,
                 conv1d_channels=[16, 32], conv1d_kws=[0, 5], conv1d_activation='ReLU'):
        print('Initializing DGCNN')
        super(DGCNN, self).__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.num_edge_feats = num_edge_feats
        self.k = k
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws[0] = self.total_latent_dim

        # 定义图卷积层
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(nn.Linear(num_node_feats, latent_dim[0]))
        for i in range(len(latent_dim) - 1):
            self.gnn_layers.append(nn.Linear(latent_dim[i], latent_dim[i + 1]))

        # 定义1D卷积层
        self.conv1d_list = nn.ModuleList()
        in_channels = [conv1d_kws[0]] + conv1d_channels[:-1]
        for in_c, out_c, kw in zip(in_channels, conv1d_channels, conv1d_kws):
            self.conv1d_list.append(nn.Conv1d(in_c, out_c, kw))
        self.activation = getattr(F, conv1d_activation.lower())

        # 定义全连接层
        self.fc = nn.Linear(conv1d_channels[-1], output_dim)

    def forward(self, x, edge_index, edge_attr=None):
        """前向传播"""
        # 图卷积层
        for gnn_layer in self.gnn_layers:
            x = F.relu(gnn_layer(x))

        # 动态池化
        x = self.dynamic_pooling(x)

        # 添加批次维度并调整为 [batch_size, channels, width]
        x = x.unsqueeze(0).transpose(1, 2)

        # 1D卷积层
        for conv in self.conv1d_list:
            x = self.activation(conv(x))

        # 平展并全连接
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def dynamic_pooling(self, x):
        """动态池化操作"""
        # 简单实现，假设直接取前k个节点特征
        if x.size(0) > self.k:
            x = x[:self.k, :]
        return x


# SEAL部分

import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score


class SEAL:
    def __init__(self, dgcnn_model, hop=1):
        print('Initializing SEAL')
        self.dgcnn = dgcnn_model
        self.hop = hop

    def extract_subgraph(self, graph, edge):
        """提取目标边的局部子图"""
        nodes = set(edge)
        for _ in range(self.hop):
            neighbors = set()
            for node in nodes:
                neighbors.update(graph.neighbors(node))
            nodes.update(neighbors)
        subgraph = graph.subgraph(nodes).copy()
        return subgraph

    def prepare_input(self, subgraph, edge):
        """将子图转换为模型的输入格式"""
        # 计算节点特征和边特征
        node_features = np.array([[subgraph.degree(n)] for n in subgraph.nodes])  # 调整为二维数组
        edge_index = np.array(list(subgraph.edges)).T
        return torch.tensor(node_features, dtype=torch.float32), torch.tensor(edge_index, dtype=torch.long)

    def train(self, graph, train_edges, train_labels):
        """训练SEAL模型"""
        self.dgcnn.train()
        optimizer = torch.optim.Adam(self.dgcnn.parameters(), lr=0.01)

        for edge, label in zip(train_edges, train_labels):
            subgraph = self.extract_subgraph(graph, edge)
            node_features, edge_index = self.prepare_input(subgraph, edge)

            optimizer.zero_grad()
            output = self.dgcnn(node_features, edge_index)
            loss = F.binary_cross_entropy_with_logits(output, torch.tensor([label], dtype=torch.float32))
            loss.backward()
            optimizer.step()

    def predict(self, graph, test_edges):
        """使用SEAL模型进行预测"""
        self.dgcnn.eval()
        predictions = []

        for edge in test_edges:
            subgraph = self.extract_subgraph(graph, edge)
            node_features, edge_index = self.prepare_input(subgraph, edge)

            with torch.no_grad():
                output = self.dgcnn(node_features, edge_index)
                predictions.append(torch.sigmoid(output).item())

        return predictions


# 示例用法
graph = nx.karate_club_graph()
train_edges = [(0, 1), (2, 3)]
train_labels = [1, 0]
test_edges = [(0, 2), (1, 3)]

dgcnn = DGCNN(output_dim=1, num_node_feats=1, num_edge_feats=1)
seal = SEAL(dgcnn_model=dgcnn)
seal.train(graph, train_edges, train_labels)
predictions = seal.predict(graph, test_edges)
print(predictions)
