import torch
import os
import torch.nn.functional as F
from torch_geometric.utils import subgraph
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def load_graph_and_edges(file_path):
    """
    Load graph edges from a file and separate positive/negative edges.
    Args:
        file_path (str): Path to the .txt file containing edges.
    Returns:
        edge_index (torch.Tensor): Edge index for the graph (only positive edges).
        positive_edges (list): List of positive edges (u, v, 1).
        negative_edges (list): List of negative edges (u, v, 0).
    """
    positive_edges = []
    negative_edges = []

    # Read file and parse edges
    with open(file_path, 'r') as f:
        for line in f:
            u, v, label = map(int, line.strip().split())
            if label == 1:
                positive_edges.append((u, v))
            elif label == 0:
                negative_edges.append((u, v))

    # Construct edge_index from positive edges
    edge_index = torch.tensor(positive_edges, dtype=torch.long).t()

    return edge_index, positive_edges, negative_edges

def extract_subgraph(u, v, edge_index, num_hops, node_features):
    """
    Extract a subgraph for the edge (u, v) with num_hops hops.
    Args:
        u (int): Source node of the edge.
        v (int): Target node of the edge.
        edge_index (torch.Tensor): Edge index of the entire graph.
        num_hops (int): Number of hops for the subgraph.
        node_features (torch.Tensor): Node features of the entire graph.
    Returns:
        torch_geometric.data.Data: Subgraph data containing x, edge_index, and other attributes.
    """
    # Find k-hop neighbors of u and v
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        (u, v), num_hops, edge_index, relabel_nodes=True
    )

    # Get the node features for the subgraph
    sub_node_features = node_features[subset]

    # Create a PyG Data object for the subgraph
    subgraph_data = Data(x=sub_node_features, edge_index=sub_edge_index)

    # Add edge indicator (used in SEAL for marking the target edge)
    subgraph_data.edge_indicator = torch.zeros(sub_node_features.size(0), dtype=torch.float)
    subgraph_data.edge_indicator[mapping[:2]] = 1.0  # First two mapped nodes correspond to u and v

    return subgraph_data

def prepare_seal_dataset(edge_list, edge_index, num_hops, node_features):
    """
    Prepare SEAL dataset for edges with labels.
    Args:
        edge_list (list): List of edges [(u, v, label)].
        edge_index (torch.Tensor): Edge index of the graph.
        num_hops (int): Number of hops for subgraph extraction.
        node_features (torch.Tensor): Node features.
    Returns:
        list: Dataset containing subgraphs for each edge.
    """
    dataset = []
    for u, v, label in edge_list:
        # Extract subgraph
        subgraph_data = extract_subgraph(u, v, edge_index, num_hops, node_features)
        subgraph_data.y = torch.tensor([label], dtype=torch.long)  # Add label
        dataset.append(subgraph_data)
    return dataset

# def prepare_seal_dataset(edge_list, edge_index, num_hops, node_features):
#     """
#     Prepare SEAL dataset for edges with labels.
#     Args:
#         edge_list (list): List of edges [(u, v, label)].
#         edge_index (torch.Tensor): Edge index of the graph.
#         num_hops (int): Number of hops for subgraph extraction.
#         node_features (torch.Tensor): Node features.
#     Returns:
#         list: Dataset containing subgraphs for each edge.
#     """
#     dataset = []
#     for u, v, label in edge_list:
#         try:
#             # Extract subgraph for the edge
#             subgraph_data = extract_subgraph(u, v, edge_index, num_hops, node_features)
#             subgraph_data.y = torch.tensor([label], dtype=torch.long)  # Add label
#             dataset.append(subgraph_data)
#         except Exception as e:
#             print(f"Failed to extract subgraph for edge ({u}, {v}): {e}")
#     return dataset

def load_and_prepare_dataset(file_path, num_hops, node_feature_dim):
    """
    Load the graph data and prepare the SEAL dataset.
    Args:
        file_path (str): Path to the .txt file containing edges.
        num_hops (int): Number of hops for subgraph extraction.
        node_feature_dim (int): Dimension of node features.
    Returns:
        train_dataset, val_dataset, test_dataset: SEAL datasets for training, validation, and testing.
    """
    # Step 1: Load graph and edges
    edge_index, positive_edges, negative_edges = load_graph_and_edges(file_path)

    # Combine positive and negative edges
    all_edges = [(u, v, 1) for u, v in positive_edges] + [(u, v, 0) for u, v in negative_edges]

    # Step 2: Initialize node features (e.g., all ones or random features)
    num_nodes = edge_index.max().item() + 1
    node_features = torch.rand((num_nodes, node_feature_dim), dtype=torch.float)

    # Step 3: Prepare SEAL datasets
    seal_dataset = prepare_seal_dataset(all_edges, edge_index, num_hops, node_features)

    return seal_dataset

def create_data_loaders(seal_dataset, batch_size):
    """
    Create DataLoader for the SEAL dataset.
    Args:
        seal_dataset (list): SEAL dataset.
        batch_size (int): Batch size for the DataLoader.
    Returns:
        DataLoader: PyTorch Geometric DataLoader for the dataset.
    """
    loader = DataLoader(seal_dataset, batch_size=batch_size, shuffle=True)
    return loader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_sort_pool

class DGCNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, k=0.6):
        """
        DGCNN model for SEAL framework.
        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden layer dimension.
            output_dim (int): Output feature dimension (usually 1 for binary classification).
            num_layers (int): Number of graph convolutional layers.
            k (float or int): SortPooling parameter (top-k nodes to retain).
        """
        super(DGCNN, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.k = k
        self.mlp_hidden = hidden_dim * num_layers
        self.conv1d = nn.Conv1d(1, 16, kernel_size=self.mlp_hidden, stride=self.mlp_hidden)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(16 * ((self.k // 2) if isinstance(self.k, int) else 1), 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x, edge_index, batch):
        # Apply GCN layers
        xs = []
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs.append(x)

        # Concatenate features from all layers
        x = torch.cat(xs, dim=-1)

        # Apply SortPooling
        x = global_sort_pool(x, batch, self.k)

        # Reshape for Conv1d
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1d(x))
        x = self.pool(x).view(x.size(0), -1)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

def train_dgcnn(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Model output
        out = model(data.x, data.edge_index, data.batch).squeeze(dim=-1)  # Adjust output shape

        # Compute loss
        loss = criterion(out, data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate_dgcnn(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            data = data.to(device)

            # Model output
            out = model(data.x, data.edge_index, data.batch).squeeze(dim=-1)  # Adjust output shape

            # Binary classification prediction
            pred = (out > 0).long()
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)

    return correct / total
if __name__ == '__main__':

    current_dir = os.getcwd()

    new_dir = os.path.abspath(os.path.join(current_dir, "../../"))
    os.chdir(new_dir)
    print("Current working directory:", os.getcwd())
    # Path to input data
    filename = 'facebook'
    train_file = f"processing_data/{filename}/train.txt"
    test_file = f"processing_data/{filename}/test.txt"

    # Hyperparameters
    num_hops = 2
    node_feature_dim = 5
    hidden_dim = 64
    output_dim = 1
    num_layers = 3
    k = 30
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Load data
    train_dataset = load_and_prepare_dataset(train_file, num_hops, node_feature_dim)
    test_dataset = load_and_prepare_dataset(test_file, num_hops, node_feature_dim)

    train_loader = create_data_loaders(train_dataset, batch_size)
    test_loader = create_data_loaders(test_dataset, batch_size)

    # Model, optimizer, and loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DGCNN(node_feature_dim, hidden_dim, output_dim, num_layers, k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(1, num_epochs + 1):
        train_loss = train_dgcnn(model, train_loader, optimizer, criterion, device)
        test_accuracy = evaluate_dgcnn(model, test_loader, device)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Accuracy = {test_accuracy:.4f}")


# if __name__ == '__main__':
#     # Example graph edge_index
#     # edge_index = torch.tensor([[0, 1, 2, 3],
#     #                            [1, 2, 3, 0]], dtype=torch.long)
#     current_dir = os.getcwd()
#
#     new_dir = os.path.abspath(os.path.join(current_dir, "../../"))
#     os.chdir(new_dir)
#     print("Current working directory:", os.getcwd())
#
#     num_hops = 2
#     node_feature_dim = 5
#     batch_size = 32
#
#     filename = 'PB'
#     file_path = f"processing_data/{filename}/train.txt"
#     # Example node features (4 nodes, 5-dimensional features)
#     edge_index, positive_edges, negative_edges = load_graph_and_edges(file_path)
#     print(f"Positive edges: {len(positive_edges)}, Negative edges: {len(negative_edges)}")
#
#     all_edges = [(u, v, 1) for u, v in positive_edges] + [(u, v, 0) for u, v in negative_edges]
#
#     num_nodes = edge_index.max().item() + 1
#
#     node_features = torch.rand((num_nodes, node_feature_dim), dtype=torch.float)
#
#     seal_dataset = load_and_prepare_dataset(file_path, num_hops, node_feature_dim)
#
#     train_loader = create_data_loaders(seal_dataset, batch_size)
#
#     print(f"Dataset size: {len(seal_dataset)}")
#     for batch in train_loader:
#         print(f"Batch node feature shape: {batch.x.shape}")
#         print(f"Batch edge index shape: {batch.edge_index.shape}")
#         print(f"Batch labels shape: {batch.y.shape}")
#         break