### Airport Graph Data Analysis and Visualization

This repository contains Python code for analyzing and visualizing airport networks using graph-based techniques.

#### Prerequisites

Ensure you have installed the following Python packages:

- `torch-geometric`: For handling graph data.
- `scikit-learn`: For KMeans clustering.
- `networkx`: For graph manipulation and visualization.
- `matplotlib`: For plotting graphs.


```bash
pip install torch torch-geometric scikit-learn networkx matplotlib
```

#### Dataset

The code uses the `Airports` dataset from `torch_geometric.datasets`. This dataset represents airports and their connections as a graph. Various attributes (features) of airports are used for clustering and analysis.

#### Functionality

The main functionalities of the code include:

1. **Loading and Analyzing the Dataset:**
   - The dataset is loaded and basic information such as the number of graphs (airport networks), features, and classes are displayed.

2. **Graph Construction:**
   - The airport connections are represented as a NetworkX graph using edge indices provided by `torch_geometric`.

3. **Clustering of Airports:**
   - KMeans clustering is applied to cluster airports based on their node features.

4. **Identifying Important Nodes:**
   - The most important nodes or airports within each cluster are identified using betweenness centrality, which measures the influence of airports in the network.

5. **Visualization:**
   - The airports and their connections are visualized using NetworkX, where each cluster is represented by a different color, and important airports are highlighted.

#### Usage

To analyze and visualize different datasets (e.g., "USA", "Brazil", "Europe"), modify the `dataset_names` list in the code and run it. Each dataset will be processed individually, and a visualization will be displayed showing the airport networks, clusters, and important airports.

```python

dataset_names = ["USA", "Brazil", "Europe"]
for name in dataset_names:
    analyze_and_visualize_airport_data(name)
```

#### Results

The output includes printed information about each dataset's characteristics and a graphical representation of airport networks with clusters and highlighted important airports.

