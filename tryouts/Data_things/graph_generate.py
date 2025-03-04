# import os
# import torch
# import pandas as pd
# from torch_geometric.data import Data, InMemoryDataset
# from tqdm import tqdm

# class GraphDataset(InMemoryDataset):
#     def __init__(self, root, transform=None, pre_transform=None):
#         super().__init__(root, transform, pre_transform)
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self):
#         return []  # We are processing directly from CSV files

#     @property
#     def processed_file_names(self):
#         return ['graph_data.pt']

#     def process(self):
#         data_list = []
#         base_path = "/home/es21btech11028/IR2Vec/tryouts/Data_things/GraphOutputs"
        
#         for class_folder in os.listdir(base_path):
#             class_path = os.path.join(base_path, class_folder)
#             if os.path.isdir(class_path):
#                 class_label = hash(class_folder) % 104  # Convert class name to a unique integer label
#                 for csv_file in os.listdir(class_path):
#                     if csv_file.endswith(".csv"):
#                         graph_data = self.load_csv(os.path.join(class_path, csv_file), class_label)
#                         if graph_data:
#                             data_list.append(graph_data)

#         # Collate data into a single batch
#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])

#     def load_csv(self, csv_path, class_label):
#         with open(csv_path, 'r') as f:
#             lines = f.readlines()

#         lines = [line.strip() for line in lines if line.strip()]
#         split_idx = lines.index("Source,Target")

#         # Load node features
#         node_data = lines[1:split_idx]
#         node_features = []
#         for line in node_data:
#             parts = line.split(',', 1)
#             if len(parts) < 2:
#                 continue
#             node_features.append(list(map(float, parts[1].split())))

#         # Load edges
#         edge_data = lines[split_idx + 1:]
#         edge_index = []
#         for line in edge_data:
#             parts = line.split(',')
#             if len(parts) != 2:
#                 continue
#             edge_index.append([int(parts[0]), int(parts[1])])

#         # Convert to tensors
#         x = torch.tensor(node_features, dtype=torch.float)
#         edge_index = torch.tensor(edge_index, dtype=torch.long).T
#         y = torch.tensor([class_label], dtype=torch.long)  # Store class label

#         return Data(x=x, edge_index=edge_index, y=y)

# # Example usage
# dataset = GraphDataset(root='graph_dataset')
# print(f'Loaded {len(dataset)} graphs!')


import os
import torch
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

class GraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # We are processing directly from CSV files

    @property
    def processed_file_names(self):
        return ['graph_data.pt']

    def process(self):
        data_list = []
        base_path = "/home/es21btech11028/IR2Vec/tryouts/Data_things/GraphOutputs"
        
        for class_folder in os.listdir(base_path):
            class_path = os.path.join(base_path, class_folder)
            if os.path.isdir(class_path):
                class_label = hash(class_folder) % 104  # Convert class name to a unique integer label
                for csv_file in os.listdir(class_path):
                    if csv_file.endswith(".csv"):
                        graph_data = self.load_csv(os.path.join(class_path, csv_file), class_label)
                        if graph_data:
                            data_list.append(graph_data)

        # Collate data into a single batch
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def load_csv(self, csv_path, class_label):
        with open(csv_path, 'r') as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines if line.strip()]
        try:
            split_idx = lines.index("Source,Target")
        except ValueError:
            print(f"Marker 'Source,Target' not found in {csv_path}")
            return None

        # Load node features
        node_data = lines[1:split_idx]
        node_features = []
        for line in node_data:
            parts = line.split(',', 1)
            if len(parts) < 2:
                continue
            try:
                features = list(map(float, parts[1].split()))
            except Exception as e:
                print(f"Error parsing features in {csv_path}: {e}")
                continue
            node_features.append(features)

        # Load edges
        edge_data = lines[split_idx + 1:]
        edge_index = []
        for line in edge_data:
            parts = line.split(',')
            if len(parts) != 2:
                continue
            try:
                edge_index.append([int(parts[0]), int(parts[1])])
            except Exception as e:
                print(f"Error parsing edge in {csv_path}: {e}")
                continue

        if not node_features or not edge_index:
            print(f"Empty nodes or edges in {csv_path}")
            return None

        # Convert to tensors
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).T
        y = torch.tensor([class_label], dtype=torch.long)  # This is your label

        print(f"Processed {csv_path}: x {x.shape}, edge_index {edge_index.shape}, y {y.shape}")
        return Data(x=x, edge_index=edge_index, y=y)


# Example usage
dataset = GraphDataset(root='graph_dataset')
print(f'Loaded {len(dataset)} graphs!')

