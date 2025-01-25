import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging
import os
import json
from torch.cuda.amp import autocast, GradScaler
import argparse

MODEL = 4
MAXNORM = 2        # need to change this to small value otherwise its very high variance 


# Step 1: Configuration Management
def parse_args():
    parser = argparse.ArgumentParser(
        description="Training script for Multiclass Classifier"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="processed_data/",
        help="Directory for processed data",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.00001,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs for training"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=f"model_data_{MODEL}/training_log_{MODEL}.txt",
        help="Log file path",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=f"model_data_{MODEL}/",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--model_architecture_file",
        type=str,
        default=f"model_data_{MODEL}/model_architecture_{MODEL}.json",
        help="File to save model architecture",
    )
    return parser.parse_args()


args = parse_args()

config = {
    "data_dir": args.data_dir,
    "batch_size": args.batch_size,
    "learning_rate": args.learning_rate,
    "epochs": args.epochs,
    "log_file": args.log_file,
    "model_dir": args.model_dir,
    "early_stopping_patience": args.early_stopping_patience,
    "model_architecture_file": args.model_architecture_file,
}

# Create checkpoint directory if it doesn't exist
os.makedirs(config["model_dir"], exist_ok=True)

# Step 2: Setup Logging
logging.basicConfig(
    filename=config["log_file"], level=logging.INFO, format="%(asctime)s %(message)s"
)


def log_message(message):
    logging.info(message)
    print(message)


# Step 3: Load Preprocessed Data
try:
    X_train = np.load(os.path.join(config["data_dir"], "X_train.npy"))
    y_train = np.load(os.path.join(config["data_dir"], "y_train.npy"))
    X_val = np.load(os.path.join(config["data_dir"], "X_val.npy"))
    y_val = np.load(os.path.join(config["data_dir"], "y_val.npy"))
except Exception as e:
    log_message(f"Error loading data: {e}")
    raise

# Apply advanced normalization (e.g., z-score normalization)
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)


class MulticlassClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MulticlassClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256 ,128)
        self.fc5 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)  # Add dropout for regularization
        self.batch_norm1 = nn.BatchNorm1d(1024)  # Add batch normalization
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.batch_norm3 = nn.BatchNorm1d(256)
        self.batch_norm4 = nn.BatchNorm1d(128)
        self.apply(self.initialize_weights)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)  # Outputs logits (before softmax)
        return x

    # Initialize weights
    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


input_size = np.prod(X_train.shape[1:])  # Flatten input dynamically
num_classes = y_train.shape[1]
model = MulticlassClassifier(input_size, num_classes)

# Convert numpy int64 to Python int
model_architecture = {
    "input_size": int(input_size),
    "num_classes": int(num_classes),
    "layers": [
        {"type": "Flatten"},
        {"type": "Linear", "in_features": int(input_size), "out_features": 1024},
        {"type": "BatchNorm1d", "num_features": 1024},
        {"type": "ReLU"},
        {"type": "Dropout", "p": 0.2},
        {"type": "Linear", "in_features": 1024, "out_features": 512},
        {"type": "BatchNorm1d", "num_features": 512},
        {"type": "ReLU"},
        {"type": "Dropout", "p": 0.2},
        {"type": "Linear", "in_features": 512, "out_features": 256},
        {"type": "BatchNorm1d", "num_features": 256},
        {"type": "ReLU"},
        {"type": "Dropout", "p": 0.2},
        {"type": "Linear", "in_features": 256, "out_features": 128},
        {"type": "BatchNorm1d", "num_features": 128},
        {"type": "ReLU"},
        {"type": "Dropout", "p": 0.2},
        {"type": "Linear", "in_features": 128, "out_features": int(num_classes)},
    ],
}

with open(config["model_architecture_file"], "w") as f:
    json.dump(model_architecture, f)

# Step 5: Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
scaler = torch.amp.GradScaler()


# Step 6: Train the Model with Early Stopping and Checkpointing
def train_model(
    model, train_loader, val_loader, criterion, optimizer, config, device="cpu"
):
    model.to(device)
    best_val_loss = float("inf")
    patience_counter = 0

    log_message("Epoch, Train Loss, Val Loss, Val Accuracy, GPU Memory Allocated, GPU Memory Reserved")  # Add header to log file
    for epoch in range(config["epochs"]):
        model.train()  # Set model to training mode
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            # Forward pass with autocast
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.argmax(dim=1))

            # Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAXNORM)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                with torch.amp.autocast("cuda"):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch.argmax(dim=1))
                    val_loss += loss.item()

                    # Calculate accuracy
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == y_batch.argmax(dim=1)).sum().item()

        # Calculate metrics
        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / len(val_dataset)

        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated(device)
            gpu_memory_reserved = torch.cuda.memory_reserved(device)
        else:
            gpu_memory_allocated = 0
            gpu_memory_reserved = 0

        epoch_log = f"{epoch+1}, {train_loss:.4f}, {val_loss:.4f}, {val_accuracy:.4f}, {gpu_memory_allocated}, {gpu_memory_reserved}"

        # Log the results
        epoch_log = f"{epoch+1}, {train_loss:.4f}, {val_loss:.4f}, {val_accuracy:.4f}"
        log_message(epoch_log)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(config["model_dir"], f"best_model.pth"),
            )
            log_message(f"Model saved at epoch {epoch+1}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= config["early_stopping_patience"]:
            log_message("Early stopping triggered")
            break


# Write a new JSON object to a file
def append_json_line(file_path, new_data):
    with open(file_path, "a") as file:
        file.write(json.dumps(new_data) + "\n")


# Read all JSON objects from the file
def read_json_lines(file_path):
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file]


if __name__ == "__main__":
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        log_message(f"Training on GPU: {torch.cuda.get_device_name(device)}")
    else:
        log_message("Training on CPU")
        
    # Load the best model if exists
    best_model_path = os.path.join(config["model_dir"], "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        log_message("Resuming training from the best saved model.")
    else:
        log_message("No saved model found. Starting training from scratch.")

    # Train the model
    train_model(
        model, train_loader, val_loader, criterion, optimizer, config, device=device
    )
    # Save the config dictionary to a file
    config_file_path = os.path.join(
        config["model_dir"], f"training_config_{MODEL}.json"
    )
    append_json_line(config_file_path, config)

    log_message(f"Training configuration saved to {config_file_path}")
