import numpy as np
import networkx as nx
import random
import os

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

def generate_negative_samples_randomly(G, target_count, seed=None):
    """
    Generate a specified number of negative samples by random selection.

    Parameters:
        G (networkx.Graph): The input graph.
        target_count (int): The number of negative samples to generate.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        list: A list of negative edges (u, v).
    """
    if seed is not None:
        random.seed(seed)

    nodes = list(G.nodes())
    existing_edges = set(G.edges())  # Use a set for O(1) lookups
    non_edges = []
    potential_negatives = len(nodes) * (len(nodes) - 1) // 2 - len(existing_edges)

    # If the number of potential negatives is less than required, generate all non-edges
    if target_count > potential_negatives:
        print(f"Insufficient potential negatives. Returning all possible negative edges.")
        non_edges = list(nx.non_edges(G))
    else:
        while len(non_edges) < target_count:
            u, v = random.sample(nodes, 2)  # Randomly pick two distinct nodes
            if (u, v) not in existing_edges and (v, u) not in existing_edges:
                non_edges.append((u, v))
                existing_edges.add((u, v))  # Avoid duplicates

    return non_edges


# def generate_data_method_not_ideal(G, train_ratio=0.81, validation_ratio=0.09, test_ratio=0.1, neg_ratio=1.0, seed=None):
#     """
#     Generate positive and negative samples with train and validation edges forming a new subgraph.
#     Negative samples for test are selected from the entire original graph.
#
#     Parameters:
#         G (networkx.Graph): The input graph.
#         train_ratio (float): Ratio of edges for training.
#         validation_ratio (float): Ratio of edges for validation.
#         test_ratio (float): Ratio of edges for testing.
#         seed (int, optional): Random seed for reproducibility.
#
#     Returns:
#         dict: Train, validation, and test sets with positive and negative samples.
#     """
#     pos_train_edges, pos_val_edges, pos_test_edges = split_graph_edges(G, train_ratio, validation_ratio, test_ratio, seed)
#
#     subgraph = nx.Graph()
#     subgraph.add_edges_from(pos_train_edges + pos_val_edges)
#
#     num_neg_train = int(len(pos_train_edges) * neg_ratio)
#     num_neg_val = int(len(pos_val_edges) * neg_ratio)
#     num_neg_test = int(len(pos_test_edges) * neg_ratio)
#
#     neg_train = generate_negative_samples_randomly(subgraph, num_neg_train, seed)
#     neg_val = generate_negative_samples_randomly(subgraph, num_neg_val, seed)
#     neg_test = generate_negative_samples_randomly(G, num_neg_test, seed)
#
#     train_data = [(u, v, 1) for u, v in pos_train_edges] + [(u, v, 0) for u, v in neg_train]
#     val_data = [(u, v, 1) for u, v in pos_val_edges] + [(u, v, 0) for u, v in neg_val]
#     test_data = [(u, v, 1) for u, v in pos_test_edges] + [(u, v, 0) for u, v in neg_test]
#
#     if seed is not None:
#         random.seed(seed)
#     random.shuffle(train_data)
#     random.shuffle(val_data)
#     random.shuffle(test_data)
#
#     return train_data, val_data, test_data

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

    num_neg_train = int(len(pos_train_edges) * neg_ratio)
    num_neg_val = int(len(pos_val_edges) * neg_ratio)
    num_neg_test = int(len(pos_test_edges) * neg_ratio)

    neg_train = generate_negative_samples_randomly(G, num_neg_train, seed)
    neg_val = generate_negative_samples_randomly(G, num_neg_val, seed)
    neg_test = generate_negative_samples_randomly(G, num_neg_test, seed)



    return pos_train_edges, pos_val_edges, pos_test_edges, neg_train, neg_val, neg_test


def generate_data_method_not_ideal(pos_train_edges, pos_val_edges, pos_test_edges, neg_train, neg_val, neg_test, theta=0.001, seed=None):
    """
    Replace a portion of neg_train with edges from pos_test_edges.

    Parameters:
        pos_train_edges (list of tuple): Positive training edges in the format (node, node).
        pos_val_edges (list of tuple): Positive validation edges in the format (node, node).
        pos_test_edges (list of tuple): Positive test edges in the format (node, node).
        neg_train (list of tuple): Negative training edges in the format (node, node).
        neg_val (list of tuple): Negative validation edges in the format (node, node).
        neg_test (list of tuple): Negative test edges in the format (node, node).
        theta (float): Proportion of pos_test_edges to replace in neg_train.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        tuple: Updated pos_train_edges, pos_val_edges, pos_test_edges, neg_train, neg_val, neg_test.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Determine the number of edges to replace
    num_replace = int(theta * len(pos_test_edges))

    if num_replace > len(neg_train):
        raise ValueError("Replacement size exceeds the size of neg_train.")

    if num_replace > len(pos_test_edges):
        raise ValueError("Replacement size exceeds the size of pos_test_edges.")

    # Randomly sample edges to replace from pos_test_edges
    edges_to_replace = random.sample(pos_test_edges, num_replace)

    # Randomly sample indices in neg_train to replace
    replace_indices = random.sample(range(len(neg_train)), num_replace)

    # Perform the replacement
    for idx, edge in zip(replace_indices, edges_to_replace):
        neg_train[idx] = edge

    return pos_train_edges, pos_val_edges, pos_test_edges, neg_train, neg_val, neg_test

def combining_dataset(pos_train_edges, pos_val_edges, pos_test_edges, neg_train, neg_val, neg_test, seed=1):

    train_data = [(u, v, 1) for u, v in pos_train_edges] + [(u, v, 0) for u, v in neg_train]
    val_data = [(u, v, 1) for u, v in pos_val_edges] + [(u, v, 0) for u, v in neg_val]
    test_data = [(u, v, 1) for u, v in pos_test_edges] + [(u, v, 0) for u, v in neg_test]

    if seed is not None:
        random.seed(seed)
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    return train_data, val_data, test_data

def save_data_to_file(data, file_path):
    """
    Save edge data to a text file.

    Parameters:
        data (dict): The data to save, including positive and negative edges.
        file_path (str): Path to the file where data will be saved.
    """

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as file:
        for u, v, label in data:
            file.write(f"{u} {v} {label}\n")
def check_datasets_consistency(dataset1, dataset2):
    """
    Check if two datasets are consistent (contain the same edges).

    Parameters:
        dataset1 (list of tuple): First dataset of edges in the format (node, node).
        dataset2 (list of tuple): Second dataset of edges in the format (node, node).

    Returns:
        bool: True if datasets are consistent, False otherwise.
        set: Differences between the datasets.
    """
    set1 = set(dataset1)
    set2 = set(dataset2)

    if set1 == set2:
        return True, set()
    else:
        differences = set1.symmetric_difference(set2)
        return False, differences

def file_processing(file_path,theta = 0.001,random_seed=1):

    filename = file_path
    Original_graph = load_graph_from_txt(f"original_graphs/{filename}.txt")

    pt_i,pv_i,pp_i,nt_i,nv_i,np_i = generate_data_method_ideal(Original_graph,0.81,0.09,0.1,1.0)
    t_i, v_i, p_i = combining_dataset(pt_i, pv_i, pp_i, nt_i, nv_i, np_i)

    save_data_to_file(t_i, f"processing_data/{filename}/train_i.txt")
    save_data_to_file(v_i, f"processing_data/{filename}/val_i.txt")
    save_data_to_file(p_i, f"processing_data/{filename}/test_i.txt")

    pt,pv,pp,nt,nv,np = generate_data_method_not_ideal(pt_i,pv_i,pp_i,nt_i,nv_i,np_i,theta=theta,seed=random_seed)
    t, v, p = combining_dataset(pt,pv,pp,nt,nv,np)
    print(check_datasets_consistency(t_i,t))
    save_data_to_file(t, f"processing_data/{filename}/train.txt")
    save_data_to_file(v, f"processing_data/{filename}/val.txt")
    save_data_to_file(p, f"processing_data/{filename}/test.txt")

    return print("Data Processing Complete")

if __name__ == "__main__":
    file_processing("facebook")