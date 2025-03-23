import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm  # Import tqdm for progress bars
from torch.utils.data import random_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import sys
from datetime import datetime

# Create a log file with timestamp
log_filename = f"training_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
sys.stdout = open(log_filename, "w")
sys.stderr = sys.stdout  # Redirect errors to the same log file

print(f"Logging to {log_filename}")
class AdvancedGAT_LSTM(nn.Module):
    def __init__(self, in_features=300, hidden_features=256, out_features=1024, n_heads=8, num_classes=104, lstm_hidden=1024, dropout=0.3):
        super(AdvancedGAT_LSTM, self).__init__()

        self.gat1 = GATConv(in_features, hidden_features, heads=n_heads, dropout=dropout, concat=True)
        self.norm1 = nn.LayerNorm(hidden_features * n_heads)

        self.gat2 = GATConv(hidden_features * n_heads, hidden_features, heads=n_heads, dropout=dropout, concat=True)
        self.norm2 = nn.LayerNorm(hidden_features * n_heads)

        self.gat3 = GATConv(hidden_features * n_heads, hidden_features, heads=n_heads, dropout=dropout, concat=True)
        self.norm3 = nn.LayerNorm(hidden_features * n_heads)

        self.gat4 = GATConv(hidden_features * n_heads, out_features, heads=n_heads, dropout=dropout, concat=False)
        self.norm4 = nn.LayerNorm(out_features)

        # 🟢 LSTM Layer for Graph Representations
        self.lstm = nn.LSTM(out_features, lstm_hidden, batch_first=True, bidirectional=True)

        # Fully connected classification layers
        # self.fc1 = nn.Linear(out_features, 512)  # for the non lstm one 

        self.fc1 = nn.Linear(lstm_hidden*2 , 512) # *2 for bidirectional
        self.fc2 = nn.Linear(512, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batch_size = batch.max().item() + 1

        # 🟢 GAT Layers
        x1 = F.elu(self.gat1(x, edge_index))
        x1 = self.norm1(x1)
        x1 = self.dropout(x1)

        x2 = F.elu(self.gat2(x1, edge_index))
        x2 = self.norm2(x2)
        x2 = self.dropout(x2)
        x2 = x1 + x2  # Residual Connection

        x3 = F.elu(self.gat3(x2, edge_index))
        x3 = self.norm3(x3)
        x3 = self.dropout(x3)

        x3 = x3 + x2

        x4 = F.elu(self.gat4(x3, edge_index))
        x4 = self.norm4(x4)
        x4 = self.dropout(x4)


        # 🟢 Convert node embeddings to variable-length sequences per graph
        node_embeddings = []
        graph_lengths = []
        for i in range(batch_size):
            node_indices = (batch == i).nonzero(as_tuple=True)[0]  # Get node indices for each graph
            node_embeddings.append(x4[node_indices])
            graph_lengths.append(len(node_indices))

        # 🟢 Pad sequences to the longest graph in the batch
        x_padded = nn.utils.rnn.pad_sequence(node_embeddings, batch_first=True, padding_value=0)

        # 🟢 Pack sequence to handle variable lengths
        x_packed = pack_padded_sequence(x_padded, graph_lengths, batch_first=True, enforce_sorted=False)

        # 🟢 LSTM Layer
        _, (h_n, _) = self.lstm(x_packed)  # Get final hidden state
        h_n = torch.cat((h_n[0], h_n[1]), dim=-1)  # Concatenate bidirectional outputs

        # 🟢 Fully Connected Layers
        x = self.dropout(F.relu(self.fc1(h_n)))
        x = self.fc2(x)

        return x
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, GATConv):
        nn.init.xavier_uniform_(m.lin.weight)
        




# Device setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset
file_path = "/home/es21btech11028/IR2Vec/tryouts/Data_things/processed_graph_data.pt"
dataset, slices = torch.load(file_path)

for data in dataset:
    data.y -= 1  # Convert from 1-104 to 0-103


print("Min class:", min([data.y.min().item() for data in dataset]))
print("Max class:", max([data.y.max().item() for data in dataset]))

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Shuffle dataset before splitting
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Training shuffled in DataLoader
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # No shuffle in testing


# Model, Optimizer & Loss# 🚀 Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AdvancedGAT_LSTM().to(device)
model.apply(init_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
criterion = nn.CrossEntropyLoss()

scaler = torch.amp.GradScaler(device='cuda')  # Mixed Precision for Faster Training
# 🚀 Best Model Saving Setup
best_test_acc = 0.0


# 🚀 Training Function with Progress Bar & Gradient Clipping
def train(epoch):
    model.train()
    total_loss, correct = 0, 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch} [Training]", leave=True)

    for data in loop:
        data = data.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type = 'cuda'):  # Mixed Precision
            out = model(data)
            loss = criterion(out, data.y)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent exploding gradients
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct += (out.argmax(dim=1) == data.y).sum().item()

        loop.set_postfix(loss=loss.item(), acc=correct / len(train_dataset))

    return total_loss / len(train_loader), correct / len(train_dataset)

# 🚀 Evaluation Function
def evaluate():
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            correct += (out.argmax(dim=1) == data.y).sum().item()
    return correct / len(test_dataset)

# 🚀 Run Training
for epoch in range(50):  # Train for more epochs
    train_loss, train_acc = train(epoch)
    test_acc = evaluate()
    model_save_path = f"best_model_{test_acc}.pth"
    # scheduler.step()  # Adjust learning rate
    # Save best model
    # global best_test_acc
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        torch.save(model.state_dict(), model_save_path)
        print(f"✅ Best model saved at epoch {epoch+1} with Test Acc: {test_acc:.4f}")

    print(f"🔥 Epoch {epoch+1}: Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")
    print(f"🔥 Epoch {epoch+1}: Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Test Acc = {test_acc:.4f}")
