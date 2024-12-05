import numpy as np
import networkx as nx
import random

def load_graph_from_txt(file_path):
    G = nx.Graph()

    # with open(file_path, 'r') as file:
    #     for line in file:
    #         node1, node2 = line.strip().split()
    #         G.add_edge(node1, node2)
    with open(file_path, 'r') as file:
        for line in file:
            node1, node2 = map(int, line.strip().split())
            G.add_edge(node1, node2)
    return G

def split_graph_edges(G, train_ratio, validation_ratio, test_ratio, seed=None):

    if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
        raise ValueError("Ratios must sum to 1.")

    # Get all edges and shuffle
    edges = list(G.edges())
    if seed is not None:
        random.seed(seed)
    random.shuffle(edges)

    # Calculate split indices
    total_edges = len(edges)
    train_size = int(total_edges * train_ratio)
    validation_size = int(total_edges * validation_ratio)

    # Split the edges
    train_edges = edges[:train_size]
    validation_edges = edges[train_size:train_size + validation_size]
    test_edges = edges[train_size + validation_size:]

    return train_edges, validation_edges, test_edges

def generate_data_method_not_ideal(G, train_ratio, validation_ratio, test_ratio, neg_ratio=1.0, seed=None):
    """
    Generate positive and negative samples with train and validation edges forming a new subgraph.
    Negative samples for test are selected from the entire original graph.

    Parameters:
        G (networkx.Graph): The input graph.
        train_ratio (float): Ratio of edges for training.
        validation_ratio (float): Ratio of edges for validation.
        test_ratio (float): Ratio of edges for testing.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        dict: Train, validation, and test sets with positive and negative samples.
    """
    pos_train_edges, pos_val_edges, pos_test_edges = split_graph_edges(G, train_ratio, validation_ratio, test_ratio, seed)

    subgraph = nx.Graph()
    subgraph.add_edges_from(pos_train_edges + pos_val_edges)

    neg_train_edges = random.sample(list(nx.non_edges(subgraph)), int(len(pos_train_edges)*neg_ratio))
    neg_val_edges = random.sample(list(nx.non_edges(subgraph)), int(len(pos_val_edges)*neg_ratio))

    neg_test_edges = random.sample(list(nx.non_edges(G)), int(len(pos_test_edges)*neg_ratio))

    return {
        "train": {"positive": pos_train_edges, "negative": neg_train_edges},
        "validation": {"positive": pos_val_edges, "negative": neg_val_edges},
        "test": {"positive": pos_test_edges, "negative": neg_test_edges},
    }

def generate_data_method_ideal(G, train_ratio, validation_ratio, test_ratio, neg_ratio=1.0, seed=None):
    """
    Generate positive and negative samples, where all negative samples are drawn from the entire graph.

    Parameters:
        G (networkx.Graph): The input graph.
        train_ratio (float): Ratio of edges for training.
        validation_ratio (float): Ratio of edges for validation.
        test_ratio (float): Ratio of edges for testing.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        dict: Train, validation, and test sets with positive and negative samples.
    """
    # Split the graph into positive edge sets
    pos_train_edges, pos_val_edges, pos_test_edges = split_graph_edges(G, train_ratio, validation_ratio, test_ratio, seed)

    # Generate all possible negative samples from the original graph
    all_neg_edges = list(nx.non_edges(G))
    if seed is not None:
        random.seed(seed)
    random.shuffle(all_neg_edges)

    # Calculate the number of negative samples needed for each set
    neg_train_size = int(len(pos_train_edges)*neg_ratio)
    neg_val_size = int(len(pos_val_edges)*neg_ratio)
    neg_test_size = int(len(pos_test_edges)*neg_ratio)

    # Split the negative samples
    neg_train_edges = all_neg_edges[:neg_train_size]
    neg_val_edges = all_neg_edges[neg_train_size:neg_train_size + neg_val_size]
    neg_test_edges = all_neg_edges[neg_train_size + neg_val_size:neg_train_size + neg_val_size + neg_test_size]

    return {
        "train": {"positive": pos_train_edges, "negative": neg_train_edges},
        "validation": {"positive": pos_val_edges, "negative": neg_val_edges},
        "test": {"positive": pos_test_edges, "negative": neg_test_edges},
    }



Original_graph = load_graph_from_txt(f"original_graphs/facebook.txt")
t,v,p = generate_data_method_not_ideal(Original_graph,0.81,0.09,0.1,1.0)
t_i,v_i,p_i = generate_data_method_not_ideal(Original_graph,0.81,0.09,0.1,1.0)