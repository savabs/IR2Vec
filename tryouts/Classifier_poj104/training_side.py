# Step 1: Import Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Step 2: Load Preprocessed Data
data_dir = 'processed_data/'

# Load the data
X_train = np.load(data_dir + 'X_train.npy')  # Training data
y_train = np.load(data_dir + 'y_train.npy')  # Training labels (one-hot encoded)
X_val = np.load(data_dir + 'X_val.npy')      # Validation data
y_val = np.load(data_dir + 'y_val.npy')      # Validation labels (one-hot encoded)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(np.argmax(y_train, axis=1), dtype=torch.long)  # Convert to class indices
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(np.argmax(y_val, axis=1), dtype=torch.long)  # Convert to class indices

# Create DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Step 3: Define the Model
class MulticlassClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MulticlassClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Outputs logits (before softmax)
        x = self.softmax(x)
        return x

# Instantiate the model
input_size = np.prod(X_train.shape[1:])  # Flatten input dynamically
num_classes = y_train.shape[1]
model = MulticlassClassifier(input_size, num_classes)

# Step 4: Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Step 5.1: Setup Logging
log_file = "training_log.txt"

def log_message(message, file=log_file):
    with open(file, "a") as f:
        f.write(message + "\n")

# Modify the `train_model` function to include logging
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20, device='cpu'):
    model.to(device)
    log_message("Epoch, Train Loss, Val Loss, Val Accuracy")  # Add header to log file
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.argmax(dim=1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.argmax(dim=1))
                val_loss += loss.item()

                # Calculate accuracy
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == y_batch).sum().item()

        # Calculate metrics
        train_loss = running_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / len(val_loader.dataset)

        # Log the results
        epoch_log = f"{epoch+1}, {train_loss:.4f}, {val_loss:.4f}, {val_accuracy:.4f}"
        log_message(epoch_log)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            model_path = f"model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

# Step 6: Train the Model
def main():
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20, device=device)

if __name__ == "__main__":
    main()
