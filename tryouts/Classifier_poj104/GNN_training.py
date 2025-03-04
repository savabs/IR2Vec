import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
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


# Define parameters (these could be loaded from arguments or a config file)
LEARNING_RATE = 0.0001
HIDDEN_DIM = 1024
NUM_EPOCHS = 40
MODEL_TYPE = "EnchancedGTAT_LSTM"  # could also be "DeepGNN", etc.
ITERATION = 1


# ------------------------------
# Custom GraphDataset (Assumes processed file exists)
# ------------------------------
class GraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return []  # Not used, since we're processing CSVs directly
    
    @property
    def processed_file_names(self):
        return ['graph_data.pt']

    def process(self):
        # Your processing code would go here if needed.
        pass

# ------------------------------
# Baseline Deep GNN Model for Graph Classification
# ------------------------------
class DeepGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(DeepGNN, self).__init__()
        # GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        # Dropout
        self.dropout = nn.Dropout(p=0.5)
        # LSTM layer: input_size matches GNN output (hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        # Linear layer for classification
        self.lin = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Pass through GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Split node representations into sequences per graph
        num_graphs = batch.max().item() + 1
        sequences = [x[batch == i] for i in range(num_graphs)]  # List of tensors, each [num_nodes_i, hidden_dim]
        
        # Get sequence lengths
        lengths = [seq.size(0) for seq in sequences]
        
        # Pad sequences to the maximum length in the batch
        padded_sequences = pad_sequence(sequences, batch_first=True)  # [num_graphs, max_len, hidden_dim]
        
        # Pack padded sequences for LSTM
        packed_sequences = pack_padded_sequence(
            padded_sequences,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        
        # Process through LSTM
        _, (hn, _) = self.lstm(packed_sequences)  # hn: [num_layers, num_graphs, hidden_dim]
        
        # Use the last hidden state as graph representation
        graph_repr = hn[0]  # [num_graphs, hidden_dim], num_layers=1
        
        # Classification
        out = self.lin(graph_repr)  # [num_graphs, num_classes]
        return out

# ------------------------------
# SOTA: GTAT Layer with Cross Attention
# Inspired by Graph Topology Attention Networks [1]
# ------------------------------

# ------------------------------
# Custom GraphDataset (Assumes processed file exists)
# ------------------------------
class GraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return []  # Not used
    
    @property
    def processed_file_names(self):
        return ['graph_data.pt']

    def process(self):
        pass


# ------------------------------
# Enhanced GTAT Layer with Multi-Head Attention and Residuals
# ------------------------------
class EnhancedGTATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4):
        super(EnhancedGTATLayer, self).__init__()
        
        # Node feature extraction branch (GCN + LayerNorm)
        self.gcn_node = GCNConv(in_channels, out_channels)
        self.norm_node = nn.LayerNorm(out_channels)

        # Topology feature extraction branch (GCN instead of Linear)
        self.lin_topo = nn.Linear(in_channels, out_channels)
        self.norm_topo = nn.LayerNorm(out_channels)

        # Multi-head attention for fusion
        self.num_heads = num_heads
        self.attn_fc = nn.ModuleList([
            nn.Linear(2 * out_channels, out_channels) for _ in range(num_heads)
        ])
        
        # Deeper fusion network (MLP)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(out_channels * num_heads, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.ReLU()
        )
        
        # Residual connection
        self.residual = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Node-based representation
        node_feat = F.relu(self.norm_node(self.gcn_node(x, edge_index)))

        # Topology-based representation
        # topo_feat = F.relu(self.norm_topo(self.lin_topo(x, edge_index)))
        topo_feat = F.relu(self.norm_topo(self.lin_topo(x)))

        # Multi-head attention fusion
        combined_feats = []
        for head in range(self.num_heads):
            combined = torch.cat([node_feat, topo_feat], dim=-1)
            attn_weights = torch.sigmoid(self.attn_fc[head](combined))
            fused_feat = attn_weights * node_feat + (1 - attn_weights) * topo_feat
            combined_feats.append(fused_feat)

        # Concatenate all heads and pass through MLP for deeper fusion
        combined_feats = torch.cat(combined_feats, dim=-1)
        fused_output = self.fusion_mlp(combined_feats)

        # Add residual connection
        return fused_output + self.residual(x)


# ------------------------------
# Enhanced DeepGTAT Model for Graph Classification
# ------------------------------
class EnhancedDeepGTAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(EnhancedDeepGTAT, self).__init__()
        # Enhanced GTAT layers
        self.layer1 = EnhancedGTATLayer(input_dim, hidden_dim)
        self.layer2 = EnhancedGTATLayer(hidden_dim, hidden_dim)
        self.layer3 = EnhancedGTATLayer(hidden_dim, hidden_dim)
        self.layer4 = EnhancedGTATLayer(hidden_dim, hidden_dim)
        self.layer5 = EnhancedGTATLayer(hidden_dim, hidden_dim)
        self.layer6 = EnhancedGTATLayer(hidden_dim, hidden_dim)
        # Dropout
        self.dropout = nn.Dropout(p=0.4)
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        # Linear layer
        self.lin = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Pass through GTAT layers
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.layer3(x, edge_index)
        x = self.layer4(x, edge_index)
        x = self.layer5(x, edge_index)
        x = self.layer6(x, edge_index)
        x = self.dropout(x)
        
        # Split node representations into sequences per graph
        num_graphs = batch.max().item() + 1
        sequences = [x[batch == i] for i in range(num_graphs)]  # List of tensors
        
        # Get sequence lengths
        lengths = [seq.size(0) for seq in sequences]
        
        # Pad sequences
        padded_sequences = pad_sequence(sequences, batch_first=True)  # [num_graphs, max_len, hidden_dim]
        
        # Pack sequences
        packed_sequences = pack_padded_sequence(
            padded_sequences,
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        
        # Process through LSTM
        _, (hn, _) = self.lstm(packed_sequences)  # hn: [1, num_graphs, hidden_dim]
        
        # Graph representation from last hidden state
        graph_repr = hn[0]  # [num_graphs, hidden_dim]
        
        # Classification
        out = self.lin(graph_repr)  # [num_graphs, num_classes]
        return out


# ------------------------------
# Weight Initialization Function (Xavier Initialization)
# ------------------------------
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ------------------------------
# ------------------------------
# Main: Load dataset, train, evaluate, and save checkpoints
# ------------------------------
if __name__ == "__main__":
    # Load the processed dataset
    dataset = GraphDataset(root='/home/es21btech11028/IR2Vec/tryouts/Data_things/graph_dataset')
    print(f'Loaded {len(dataset)} graphs!')

    # Shuffle and split dataset (80% train, 20% validation)
    torch.manual_seed(42)
    dataset = dataset.shuffle()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Choose model type: "DeepGNN" for the baseline or "GTAT" for the SOTA variant
    
    input_dim = dataset[0].x.shape[1]  # e.g., 300 features per node
    num_classes = 104                  # As per your folder structure
    
    if MODEL_TYPE == MODEL_TYPE:
        model = EnhancedDeepGTAT(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_classes=num_classes).to(device)
        print("Using SOTA DeepGTAT model with cross attention.")
    else:
        model = DeepGNN(input_dim=input_dim, hidden_dim=HIDDEN_DIM, num_classes=num_classes).to(device)
        print("Using baseline DeepGNN model.")
    
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
