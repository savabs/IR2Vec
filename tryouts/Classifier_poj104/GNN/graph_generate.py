import os
import torch
from torch_geometric.data import Data
import pandas as pd
from io import StringIO


# Path to the main directory containing 104 class folders
main_dir = '/home/es21btech11028/IR2Vec/tryouts/Data_things/GraphOutputs'

# List all class folders
class_folders = [f for f in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, f))]

# Initialize an empty list to store graphs
graph_data_list = []

for class_folder in class_folders:
    class_path = os.path.join(main_dir, class_folder)
    
    # List all CSV files in the current class folder
    csv_files = [f for f in os.listdir(class_path) if f.endswith('.csv')]
    
    for csv_file in csv_files:
        csv_path = os.path.join(class_path, csv_file)
        
        # Read the CSV file
        with open(csv_path, 'r') as file:
            content = file.read()
        
        # Split into node features and edge index sections
        node_features_section, edge_index_section = content.split("\n\n")
        
        # Process node features
        # node_features_df = pd.read_csv(pd.compat.StringIO(node_features_section), sep=",")
        # node_features_df = pd.read_csv(StringIO(node_features_section), sep=",")
        # print(node_features_df.head())
        # print(node_features_df.dtypes)
        # x = torch.tensor(node_features_df.iloc[:, 1:].values, dtype=torch.float)  # Feature vectors
        
        node_features_df = pd.read_csv(StringIO(node_features_section), sep=",")
        # print("Before processing:")
        # print(node_features_df)

        # Split the FeatureVector column into separate numeric columns
        feature_vectors = node_features_df['FeatureVector'].str.split(expand=True).astype(float)
        # print("Feature vectors after splitting:")
        # print(feature_vectors)

        # Replace the original FeatureVector column with the split columns
        node_features_df = pd.concat([node_features_df['NodeID'], feature_vectors], axis=1)
        # print("Processed DataFrame:")
        # print(node_features_df)

        # Convert to PyTorch tensor
        x = torch.tensor(node_features_df.iloc[:, 1:].values, dtype=torch.float32)  # Exclude NodeID column
        # print("Node features tensor:")
        # print(x)
        # Process edge index
        edge_index_df = pd.read_csv(StringIO(edge_index_section), sep=",")
        edge_index = torch.tensor(edge_index_df.values.T, dtype=torch.long)  # Transpose 
        
        # Create a Data object for the graph
        graph_data = Data(x=x, edge_index=edge_index)
        
        # Add graph-level label (optional)
        graph_data.y = torch.tensor([int(class_folder)])  # Assuming folder name represents the class
        
        # Append the graph to the list
        graph_data_list.append(graph_data)

print(f"Created {len(graph_data_list)} graphs!")

# Save the dataset as a PyTorch Geometric InMemoryDataset-compatible format
torch.save((graph_data_list, None), '/home/es21btech11028/IR2Vec/tryouts/Data_things/processed_graph_data.pt')
