import torch
from torch_geometric.data import Data

# Path to the processed graph dataset
file_path = '/home/es21btech11028/IR2Vec/tryouts/Data_things/processed_graph_data.pt'

# Load the dataset
data, slices = torch.load(file_path)

# Check if it's a single graph or multiple graphs
if isinstance(data, list):
    print(f"Number of graphs: {len(data)}")
    for i, graph in enumerate(data[:5]):  # Inspect the first 5 graphs
        print(f"\nGraph {i + 1}:")
        print(f"  Number of nodes: {graph.x.size(0)}")
        print(f"  Node features shape: {graph.x.shape}")
        print(f"  Edge index shape: {graph.edge_index.shape}")
        print(f"  Number of edges: {graph.edge_index.size(1)}")
        print(f"  Graph label: {graph.y if hasattr(graph, 'y') else 'No label'}")
else:
    print("Single graph dataset:")
    print(f"  Number of nodes: {data.x.size(0)}")
    print(f"  Node features shape: {data.x.shape}")
    print(f"  Edge index shape: {data.edge_index.shape}")
    print(f"  Number of edges: {data.edge_index.size(1)}")
    print(f"  Graph label: {data.y if hasattr(data, 'y') else 'No label'}")
