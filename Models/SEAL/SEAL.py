import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import subgraph, to_undirected
import os

def load_data(file_path):
    """
    Load edge list and labels from a text file.
    Args:
        file_path (str): Path to the .txt file.
    Returns:
        list: List of edges [(node1, node2, label)].
    """
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            u, v, label = map(int, line.strip().split())
            edges.append((u, v, label))
    return edges

def build_graph_from_edges(edges):
    """
    Build edge_index and node features from edge list.
    Args:
        edges (list): List of edges [(node1, node2, label)].
    Returns:
        edge_index (Tensor): Edge index of the graph.
        node_features (Tensor): Node features (initialized as ones).
    """
    # Extract all unique nodes
    nodes = set()
    for u, v, _ in edges:
        nodes.add(u)
        nodes.add(v)
    num_nodes = max(nodes) + 1

    # Build edge_index
    edge_index = []
    for u, v, label in edges:
        edge_index.append([u, v])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    edge_index = to_undirected(edge_index)  # Ensure the graph is undirected

    # Initialize node features (e.g., all ones)
    node_features = torch.ones((num_nodes, 5), dtype=torch.float)  # Example: 5-dimensional features

    return edge_index, node_features

# ===== Custom DGCNN Implementation =====
class DGCNN(nn.Module):
    def __init__(self, output_dim, num_node_feats, latent_dim=[32, 32, 32, 1], k=30):
        super(DGCNN, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_node_feats = num_node_feats
        self.k = k
        self.total_latent_dim = sum(latent_dim)

        self.conv_params = nn.ModuleList()
        self.conv_params.append(nn.Linear(num_node_feats, latent_dim[0]))
        for i in range(1, len(latent_dim)):
            self.conv_params.append(nn.Linear(latent_dim[i - 1], latent_dim[i]))

        self.conv1d_params1 = nn.Conv1d(1, 16, kernel_size=min(5, self.total_latent_dim), stride=1)
        self.maxpool1d = nn.MaxPool1d(2, 2)
        self.conv1d_params2 = nn.Conv1d(16, 32, kernel_size=min(5, self.total_latent_dim // 2), stride=1)

        dense_dim = max((k - 2) // 2 + 1, 1)
        dense_dim = max((dense_dim - 5 + 1), 1) * 32
        self.dense_dim = dense_dim

        if output_dim > 0:
            self.out_params = nn.Linear(self.dense_dim, output_dim)

    def forward(self, x, edge_index, batch):
        h = x
        for conv in self.conv_params:
            h = F.relu(conv(h))

        # Sort Pooling
        max_nodes = h.size(0) // (batch.max().item() + 1) if batch.numel() > 0 else 1
        k = min(self.k, max_nodes)
        h = torch.cat([h[batch == i][:k].mean(0).unsqueeze(0) for i in range(batch.max() + 1)], dim=0)

        # Add channel dimension (B, C, L)
        h = h.unsqueeze(1)

        if h.size(1) != 1:
            h = h.mean(dim=1, keepdim=True)  # Ensure single channel

        # Ensure input shape is valid for adaptive_avg_pool1d
        if h.size(2) < 5:
            h = F.adaptive_avg_pool1d(h, 5)

        h = F.relu(self.conv1d_params1(h))
        h = self.maxpool1d(h)
        h = F.relu(self.conv1d_params2(h))
        h = h.view(h.size(0), -1)

        if self.output_dim > 0:
            h = self.out_params(h)

        return h    

# ===== SEAL Model =====
class SEAL(nn.Module):
    def __init__(self, num_features, embedding_dim, k=30):
        """
        SEAL model for link prediction.
        Args:
            num_features (int): Node feature dimension.
            embedding_dim (int): Dimension of embeddings.
            k (int): Number of nodes to keep in sorted pooling.
        """
        super(SEAL, self).__init__()
        self.k = k
        self.dgcnn = DGCNN(output_dim=2, num_node_feats=num_features, latent_dim=[embedding_dim, embedding_dim, embedding_dim, 1], k=k)

    def forward(self, x, edge_index, batch):
        """
        Forward pass for SEAL.
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge indices.
            batch (Tensor): Batch indices for subgraph.
        Returns:
            Tensor: Predicted scores for edges.
        """
        return self.dgcnn(x, edge_index, batch)

# ===== Helper Functions =====
def extract_subgraph(u, v, edge_index, num_hops, node_features):
    subset, edge_index = subgraph((u, v), edge_index, relabel_nodes=True)
    x = node_features[subset]
    return Data(x=x, edge_index=edge_index)

def prepare_seal_dataset(edge_list, edge_index, num_hops, node_features):
    dataset = []
    for u, v, label in edge_list:
        subgraph_data = extract_subgraph(u, v, edge_index, num_hops, node_features)
        subgraph_data.y = torch.tensor([label], dtype=torch.long)  # Add label
        dataset.append(subgraph_data)
    return dataset

def train_seal(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_seal(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            total += data.y.size(0)
    return correct / total

def run_seal_pipeline(train_edges, val_edges, test_edges, edge_index, node_features, num_hops, batch_size, epochs):
    train_dataset = prepare_seal_dataset(train_edges, edge_index, num_hops, node_features)
    val_dataset = prepare_seal_dataset(val_edges, edge_index, num_hops, node_features)
    test_dataset = prepare_seal_dataset(test_edges, edge_index, num_hops, node_features)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = SEAL(num_features=node_features.size(1), embedding_dim=32, k=30)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss = train_seal(model, train_loader, optimizer, criterion)
        val_acc = evaluate_seal(model, val_loader)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

    test_acc = evaluate_seal(model, test_loader)
    print(f"Test Accuracy: {test_acc:.4f}")

    return {"train_loss": train_loss, "val_acc": val_acc, "test_acc": test_acc}

if __name__ == "__main__":
    current_dir = os.getcwd()

    new_dir = os.path.abspath(os.path.join(current_dir, "../../"))
    os.chdir(new_dir)
    print("Current working directory:", os.getcwd())
    filename = 'PB'
    # Load data
    train_edges = load_data(f"processing_data/{filename}/train.txt")
    val_edges = load_data(f"processing_data/{filename}/val.txt")
    test_edges = load_data(f"processing_data/{filename}/test.txt")

    # Build graph (using train edges for constructing the graph)
    all_edges = train_edges + val_edges + test_edges
    edge_index, node_features = build_graph_from_edges(all_edges)

    results = run_seal_pipeline(
        train_edges=train_edges,
        val_edges=val_edges,
        test_edges=test_edges,
        edge_index=edge_index,
        node_features=node_features,
        num_hops=2,  # Number of hops for subgraph extraction
        batch_size=32,
        epochs=10
    )

    # Print results
    print("SEAL Results:")
    print(f"Train Loss: {results['train_loss']:.4f}")
    print(f"Validation Accuracy: {results['val_acc']:.4f}")
    print(f"Test Accuracy: {results['test_acc']:.4f}")
