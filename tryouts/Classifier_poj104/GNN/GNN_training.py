import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import torch_geometric
from tqdm import tqdm
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.data import InMemoryDataset, Data
# ------------------------------
# Enhanced GTAT Layer with Multi-Head Attention and Residualsimport torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

# Define parameters (these could be loaded from arguments or a config file)
LEARNING_RATE = 0.00005
HIDDEN_DIM = 2048
NUM_EPOCHS = 40
MODEL_TYPE = "EnchancedGTAT_GAT_LSTM2"  # could also be "DeepGNN", etc.
ITERATION = 1

# ------------------
# SOTA: GTAT Layer with Cross Attention
# Inspired by Graph Topology Attention Networks [1]
# ------------------------------

# ------------------------------
# Custom GraphDataset (Assumes processed file exists)
# ------------------------------
class GraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])  # Load processed data
    
    @property
    def raw_file_names(self):
        return []  # Not used since we directly process CSVs or other formats
    
    @property
    def processed_file_names(self):
        return ['/home/es21btech11028/IR2Vec/tryouts/Data_things/processed_graph_data.pt']  # Update this to match your new dataset file name

    def process(self):
        # If needed, implement processing logic here to create `new_graph_data.pt`
        pass

class GATLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, num_heads=4, dropout=0.3):
        super(GATLayer, self).__init__(aggr='add', flow='source_to_target')  # Aggregate via summation
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        
        # Linear transformation for node features (shared across heads)
        self.lin = nn.Linear(in_channels, out_channels * num_heads, bias=False)
        
        # Attention mechanism: one per head
        # Output dim is 1 per head to compute scalar attention scores
        self.attn = nn.Parameter(torch.Tensor(num_heads, 2 * out_channels, 1))
        
        # Layer normalization and residual
        self.norm = nn.LayerNorm(out_channels * num_heads)
        self.residual = nn.Linear(in_channels, out_channels * num_heads)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.attn)
        nn.init.xavier_uniform_(self.residual.weight)
        if self.residual.bias is not None:
            nn.init.zeros_(self.residual.bias)

    def forward(self, x, edge_index):
        # Input x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        print(f"Inside GATLayer forward: x.shape = {x.shape}, edge_index.max() = {edge_index.max().item()}")

        # INTEGRATION: Fixed neighborhood sampling
        num_nodes = x.size(0)
        assert edge_index.max() < num_nodes, f"edge_index max {edge_index.max()} >= num_nodes {num_nodes}"
        # First validate edge indices to prevent CUDA errors
        if edge_index.numel() > 0:
            mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            edge_index = edge_index[:, mask]
            
            # Apply fixed neighborhood sampling if needed
            MAX_NEIGHBORS = 16  # Adjust based on your dataset
            
            # Only apply sampling if we have valid edges
            if edge_index.size(1) > 0:
                new_edges = []
                unique_source_nodes = torch.unique(edge_index[0])
                # For each node, get its outgoing edges
                for node in unique_source_nodes:
                    # Find neighbors of this node
                    mask = edge_index[0] == node
                    node_edges = edge_index[:, mask]
                    
                    # If more neighbors than MAX_NEIGHBORS, sample randomly
                    if node_edges.size(1) > MAX_NEIGHBORS:
                        perm = torch.randperm(node_edges.size(1))[:MAX_NEIGHBORS]
                        node_edges = node_edges[:, perm]
                    
                    # Keep all edges for nodes with <= MAX_NEIGHBORS
                    new_edges.append(node_edges)
                
                # Only combine if we have any edges
                if new_edges:
                    edge_index = torch.cat(new_edges, dim=1)
        
        # Continue with existing implementation
        x_transformed = self.lin(x)
        x_transformed = x_transformed.view(-1, self.num_heads, self.out_channels)
        print(f"x_transformed.shape = {x_transformed.shape}")
        assert x_transformed.size(0) == num_nodes, "x_transformed size mismatch"
        # Residual connection
        residual = self.residual(x)
        
        # Skip propagation if we have no edges
        if edge_index.size(1) == 0:
            out = torch.zeros_like(x_transformed.view(-1, self.num_heads * self.out_channels))
            return F.elu(self.norm(out + residual))
        
        # Propagate messages (attention mechanism)
        out = self.propagate(edge_index, x=x_transformed)
        
        # Concatenate heads
        out = out.view(-1, self.num_heads * self.out_channels)
        
        # Apply normalization and residual
        out = self.norm(out + residual)
        
        return F.elu(out)

    
    def message(self, x_i, x_j, index, size_i):
        # Print shapes for debugging
        print(f"x_i shape: {x_i.shape}, x_j shape: {x_j.shape}")
        
        # Concatenate source and target features
        x_cat = torch.cat([x_i, x_j], dim=-1)
        
        # Flatten x_cat to 2D to simplify attention computation
        if x_cat.dim() == 3:
            num_edges, mid_dim, features = x_cat.shape
            x_cat_flat = x_cat.view(num_edges, -1)  # [num_edges, mid_dim*features]
        else:
            x_cat_flat = x_cat
        
        # Create a compatible attention weight vector
        attn_flat = self.attn.view(-1)  # Flatten attention weights
        
        # Ensure compatible dimensions with repeat or slice
        if attn_flat.size(0) > x_cat_flat.size(1):
            attn_flat = attn_flat[:x_cat_flat.size(1)]
        else:
            repeat_factor = (x_cat_flat.size(1) + attn_flat.size(0) - 1) // attn_flat.size(0)
            attn_flat = attn_flat.repeat(repeat_factor)[:x_cat_flat.size(1)]
        
        # Simple dot product for attention
        attn_scores = (x_cat_flat * attn_flat.unsqueeze(0)).sum(dim=1)
        
        # Apply non-linearity
        attn_scores = F.leaky_relu(attn_scores, negative_slope=0.2)
        
        # Normalize with softmax
        attn_weights = softmax(attn_scores, index, num_nodes=size_i)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to features (handling 2D or 3D tensors)
        if x_j.dim() == 3:
            return x_j * attn_weights.unsqueeze(-1).unsqueeze(-1)
        else:
            return x_j * attn_weights.unsqueeze(-1)
    
    
    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, out_channels={self.out_channels}, heads={self.num_heads})"

# Update EnhancedDeepGTAT to use GATLayer
class EnhancedDeepGTAT_GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_heads=4):
        super(EnhancedDeepGTAT_GAT, self).__init__()
        # GAT layers
        self.layer1 = GATLayer(input_dim, hidden_dim // num_heads, num_heads)
        self.layer2 = GATLayer(hidden_dim, hidden_dim // num_heads, num_heads)
        self.layer3 = GATLayer(hidden_dim, hidden_dim // num_heads, num_heads)
        self.layer4 = GATLayer(hidden_dim, hidden_dim // num_heads, num_heads)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.3)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Linear layer for classification
        self.lin = nn.Linear(hidden_dim, num_classes)
        self.num_classes = num_classes
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        num_nodes = x.size(0)
        if edge_index.max() >= num_nodes or edge_index.min() < 0:
            raise ValueError(f"Invalid edge_index: max index {edge_index.max().item()}, num_nodes {num_nodes}")
        
        print(x.shape)
        print(edge_index.shape)
        # Check for empty edge_index
        if edge_index.numel() == 0:
            print("WARNING: Empty edge_index detected!")
            # Handle empty graphs by returning zeros or other placeholder
            return torch.zeros(batch.max().item() + 1, self.num_classes).to(x.device)
            
        # Check batch structure
        if batch.max().item() + 1 > x.size(0):
            print(f"WARNING: Invalid batch structure: {batch.max().item() + 1} graphs but only {x.size(0)} nodes")
        

        try:
            # Pass through GAT layers
            x = self.layer1(x, edge_index)
            x = self.layer2(x, edge_index)
            x = self.layer3(x, edge_index)
            x = self.layer4(x, edge_index)
        except RuntimeError as e:
            print(f"Error in GAT layers: {e}")
            print(f"Graph info: {len(batch)} nodes, {edge_index.size(1)} edges")
            raise 

        
        # Split node representations into sequences per graph
        num_graphs = batch.max().item() + 1
        sequences = [x[batch == i] for i in range(num_graphs)]
        
        # Get sequence lengths
        lengths = [seq.size(0) for seq in sequences]
        
        # Pad sequences
        padded_sequences = pad_sequence(sequences, batch_first=True)
        
        # Pack sequences
        packed_sequences = pack_padded_sequence(
            padded_sequences,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        
        # Process through LSTM
        _, (hn, _) = self.lstm(packed_sequences)
        
        # Graph representation from last hidden state
        graph_repr = hn[0]  # [num_graphs, hidden_dim]
        
        # Classification
        out = self.lin(graph_repr)
        return out

# ------------------------------
# Main: Load dataset, train, evaluate, and save checkpoints
# ------------------------------
if __name__ == "__main__":
    # Load the processed dataset
    dataset = GraphDataset(root='/home/es21btech11028/IR2Vec/tryouts/Data_things/')
    
    print(f'Loaded {len(dataset)} graphs!')

    # Shuffle and split dataset (80% train, 20% validation)
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    for data in train_loader:
        x, edge_index = data.x, data.edge_index
        num_nodes = x.size(0)
        if edge_index.numel() > 0 and edge_index.max() >= num_nodes:
            print(f"Batch issue: max edge index {edge_index.max().item()} >= num_nodes {num_nodes}")
        # Set up device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    
    # Choose model type: "DeepGNN" for the baseline or "GTAT" for the SOTA variant
    
    input_dim = dataset[0].x.shape[1]  # e.g., 300 features per node
    num_classes = 104                  # As per your folder structure
    
    # if MODEL_TYPE == MODEL_TYPE:
    model = EnhancedDeepGTAT_GAT(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_classes=num_classes).to(device)
    print("Using SOTA DeepGTAT model with cross attention.")
    # else:
    #     model = DeepGNN(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_classes=num_classes).to(device)
    #     print("Using baseline DeepGNN model.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    # Create dynamic checkpoint filenames using f-strings
    checkpoint_path = f"checkpoint_{MODEL_TYPE}_ITERATION_{ITERATION}_HIDDEN_{HIDDEN_DIM}_LSTM.pth"
    best_model_path = f"best_model_{MODEL_TYPE}_ITERATION_{ITERATION}_HIDDEN_{HIDDEN_DIM}.pth"

    print("Checkpoint will be saved at:", checkpoint_path)

    
    # Optionally resume training if checkpoint exists
    start_epoch = 1
    best_val_acc = 0.0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint["best_val_acc"]
        print(f"Resuming training from epoch {start_epoch} with best validation accuracy {best_val_acc:.4f}")
    
    # After loading the dataset
    dataset = GraphDataset(root='/home/es21btech11028/IR2Vec/tryouts/Data_things/graph_dataset')
    print(f'Loaded {len(dataset)} graphs!')

    for data in train_loader:
        data = data.to(device)
        print(f"Data type: {type(data)}")
        if isinstance(data, torch_geometric.data.Batch):
            print(f"Batch size: {data.num_graphs}")
            print(f"Total nodes: {data.x.size(0)}")
            print(f"Edge index shape: {data.edge_index.shape}")
            print(f"Max edge index: {data.edge_index.max().item()}")
        else:
            print(f"Single graph - Nodes: {data.x.size(0)}, Max edge index: {data.edge_index.max().item()}")
        optimizer.zero_grad()
        out = model(data)
        # ... rest of the loop

    valid_indices = []
    for i, data in enumerate(dataset):
        num_nodes = data.x.size(0)
        if data.edge_index.numel() > 0:
            max_index = data.edge_index.max().item()
            if max_index < num_nodes:
                valid_indices.append(i)
            else:
                print(f"Skipping graph {i}: max index {max_index} >= nodes {num_nodes}")

    dataset = dataset[valid_indices]  # Keep only valid graphs
    
    num_epochs = NUM_EPOCHS
    for epoch in range(start_epoch, num_epochs + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        # Training loop with progress bar
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False)
        for data in train_bar:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.num_graphs
            preds = out.argmax(dim=1)
            correct_train += (preds == data.y).sum().item()
            total_train += data.num_graphs
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / total_train
        train_acc = correct_train / total_train

        # Validation loop with progress bar
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            val_bar = tqdm(test_loader, desc=f"Epoch {epoch} Validation", leave=False)
            for data in val_bar:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y)
                val_loss += loss.item() * data.num_graphs
                preds = out.argmax(dim=1)
                correct_val += (preds == data.y).sum().item()
                total_val += data.num_graphs
                val_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_val_loss = val_loss / total_val
        val_acc = correct_val / total_val
        epoch_duration = time.time() - epoch_start_time

        # Save checkpoint at each epoch
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
        }
        torch.save(checkpoint, checkpoint_path)
        # Save best model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint["best_val_acc"] = best_val_acc
            torch.save(checkpoint, best_model_path)
            best_model_str = " [Best Model]"
        else:
            best_model_str = ""
        
        # Print out epoch metrics
        print(
            f"Epoch {epoch:03d} | Time: {epoch_duration:.2f}s | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}{best_model_str}"
        )
