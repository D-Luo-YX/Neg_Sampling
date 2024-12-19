import numpy as np
import scipy.io as scio
import networkx as nx


def txt_to_mat(input_txt_path, output_mat_path):
    """
    Convert a node-node formatted .txt file to a .mat file with a sparse adjacency matrix.

    Parameters:
    - input_txt_path (str): Path to the input .txt file (node-node format).
    - output_mat_path (str): Path to save the output .mat file.
    """
    # Load the edges from the .txt file
    edges = []
    with open(input_txt_path, 'r') as file:
        for line in file:
            node1, node2 = map(int, line.strip().split())
            edges.append((node1, node2))

    # Create a graph using NetworkX
    G = nx.Graph()
    G.add_edges_from(edges)

    # Convert the graph to a sparse adjacency matrix
    adjacency_matrix = nx.to_scipy_sparse_matrix(G, format='csc')

    # Save the adjacency matrix to a .mat file
    scio.savemat(output_mat_path, {'net': adjacency_matrix})

# Example usage
# txt_to_mat('path/to/input.txt', 'path/to/output.mat')
if __name__ == '__main__':
    filename = 'Wiki-Vote'
    txt_to_mat(f'dataset/{filename}.txt', f'dataset/{filename}.mat')