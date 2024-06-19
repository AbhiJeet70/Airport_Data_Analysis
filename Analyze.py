import os
import numpy as np
import networkx as nx
from torch_geometric.datasets import Airports
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt

def analyze_and_visualize_airport_data(dataset_name, root="./datasets/Airport"):
    # Load the dataset
    dataset = Airports(root=root, name=dataset_name)
    data = dataset[0]

    print(f'Dataset: {dataset_name}')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(data)
    print('===============================================================')

    # Get node features and the graph
    node_features = data.x.numpy()
    edge_index = data.edge_index.numpy()

    # Create a NetworkX graph from the edge index
    G = nx.Graph()
    edges = edge_index.T  # Transpose to get edges as pairs of nodes
    G.add_edges_from(edges)

    # Use KMeans to cluster the nodes based on their features
    num_clusters = dataset.num_classes  # Number of clusters is the number of classes
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(node_features)

    # Get the cluster labels for each node
    cluster_labels = kmeans.labels_

    # Dictionary to store nodes in each cluster
    cluster_nodes = {i: [] for i in range(num_clusters)}

    # Assign nodes to their respective clusters
    for idx, label in enumerate(cluster_labels):
        cluster_nodes[label].append(idx)

    # Function to get the most important nodes using betweenness centrality
    def get_most_important_nodes(G, nodes, top_k=5):
        subgraph = G.subgraph(nodes)
        centrality = nx.betweenness_centrality(subgraph)
        sorted_nodes = sorted(centrality.items(), key=lambda item: item[1], reverse=True)
        return [node for node, centrality in sorted_nodes[:top_k]]

    # Find the most important nodes in each cluster
    important_nodes = {}

    for cluster, nodes in cluster_nodes.items():
        important_nodes[cluster] = get_most_important_nodes(G, nodes)

    # Analyze attributes of important nodes
    important_nodes_attributes = {}

    for cluster, nodes in important_nodes.items():
        important_nodes_attributes[cluster] = {
            'nodes': nodes,
            'degrees': [G.degree(node) for node in nodes],
            'features': [node_features[node] for node in nodes]
        }

    print('Most important nodes and their attributes in each cluster:')
    for cluster, attributes in important_nodes_attributes.items():
        print(f'Cluster {cluster}:')
        for i, node in enumerate(attributes['nodes']):
            print(f'  Node {node}: Degree {attributes["degrees"][i]}, Features {attributes["features"][i]}')

    # Visualization
    pos = nx.spring_layout(G)  # Spring layout for better visualization

    plt.figure(figsize=(12, 8))

    # Draw the nodes with different colors for each cluster
    colors = plt.cm.rainbow(np.linspace(0, 1, num_clusters))

    for cluster in range(num_clusters):
        nx.draw_networkx_nodes(G, pos, nodelist=cluster_nodes[cluster], node_color=[colors[cluster]]*len(cluster_nodes[cluster]), label=f'Cluster {cluster}')

    # Draw the edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)

    # Highlight the important nodes
    for cluster, nodes in important_nodes.items():
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color='black', node_size=200, edgecolors='yellow', linewidths=2)

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_color='black')

    plt.title(f'Graph Visualization with Clusters and Important Nodes for {dataset_name}')
    plt.legend()
    plt.show()

# Dataset names
dataset_names = ["USA", "Brazil", "Europe"]

# Loop over each dataset name and analyze/visualize
for name in dataset_names:
    analyze_and_visualize_airport_data(name)
