import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import os
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import logging

# Path to dataset
DATASET_PATH = "/home/es21btech11028/IR2Vec/tryouts/Data_things/instruction_embeddings"

# Set up logging
logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TransformerChunkEncoder(nn.Module):
    def __init__(
        self, input_dim=300, embed_dim=256, num_heads=8, ff_dim=512, num_layers=2
    ):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        x = self.embedding(x)

        # Identify chunks that are fully padded and replace output with zeros
        fully_padded = mask.all(dim=1)  # (batch_size * num_chunks)
        if fully_padded.any():
            x[fully_padded] = 0

        # Only process non-empty sequences
        non_empty_x = x[~fully_padded]
        non_empty_mask = mask[~fully_padded] if mask is not None else None

        if non_empty_x.shape[0] > 0:
            non_empty_x = self.transformer(
                non_empty_x, src_key_padding_mask=non_empty_mask
            )

        # Ensure the data types match
        non_empty_x = non_empty_x.to(x.dtype)
        x[~fully_padded] = non_empty_x

        # Exclude padding from aggregation
        mask = mask.unsqueeze(-1)
        valid_tokens = x * (~mask)
        agg_output = valid_tokens.sum(dim=1) / (~mask).sum(dim=1)

        return agg_output


class BiLSTMProcessor(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, num_layers=2):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, lengths):
        # Convert lengths to a tensor
        lengths = torch.tensor(lengths, dtype=torch.long)

        # Ensure lengths are sorted in descending order
        lengths, sorted_idx = torch.sort(lengths, descending=True)
        x = x[sorted_idx]

        # Check if lengths are within the valid range
        if lengths[0] > x.size(1):
            raise ValueError(f"Max length {lengths[0]} exceeds input size {x.size(1)}")

        packed_x = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        packed_out, _ = self.bilstm(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # Restore the original order
        _, original_idx = sorted_idx.sort(0, descending=False)
        out = out[original_idx]

        return out


class HybridModel(nn.Module):
    def __init__(
        self,
        input_dim=300,
        embed_dim=512,
        hidden_dim=512,
        num_heads=8,
        ff_dim=512,
        num_layers=2,
        num_classes=104,
    ):
        super().__init__()
        self.transformer_chunk = TransformerChunkEncoder(
            input_dim, embed_dim, num_heads, ff_dim, num_layers
        )
        self.bilstm = BiLSTMProcessor(embed_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, sequences, lengths):
        batch_size = len(sequences)
        max_chunks = max((len(seq) + 47) // 48 for seq in sequences)

        padded_chunks = torch.zeros((batch_size, max_chunks, 48, 300)).to(device)
        masks = torch.ones((batch_size, max_chunks, 48), dtype=torch.bool).to(device)

        for i, seq in enumerate(sequences):
            chunks = []
            for j in range(0, len(seq), 48):
                chunk = seq[j : j + 48]
                chunk_size = chunk.shape[0]

                if chunk_size < 48:
                    pad_amount = 48 - chunk_size
                    padding = torch.zeros((pad_amount, 300)).to(device)
                    chunk = torch.cat([chunk, padding], dim=0)

                chunks.append(chunk)

                masks[i, len(chunks) - 1, :chunk_size] = False

            chunk_tensor = torch.stack(chunks)
            padded_chunks[i, : chunk_tensor.shape[0], :, :] = chunk_tensor

        batch_chunks = padded_chunks.view(-1, 48, 300)
        masks = masks.view(-1, 48)

        chunk_embeddings = self.transformer_chunk(batch_chunks, mask=masks)
        chunk_embeddings = chunk_embeddings.view(batch_size, max_chunks, -1)

        # Update lengths to reflect the number of chunks after transformer processing
        lengths = [min(max_chunks, (length + 47) // 48) for length in lengths]

        lstm_out = self.bilstm(chunk_embeddings, lengths)

        agg_output = torch.mean(lstm_out, dim=1)

        return self.fc(agg_output)


class InstructionDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))  # 104 classes
        self.data = []
        self.labels = []

        for class_idx, class_name in enumerate(
            tqdm(self.classes, desc="Loading Classes")
        ):
            class_path = os.path.join(root_dir, class_name)
            for sample_file in tqdm(
                os.listdir(class_path), desc=f"Loading {class_name}", leave=False
            ):
                sample_path = os.path.join(class_path, sample_file)
                sample_data = np.load(sample_path)  # Load .npy file
                sample_data = torch.tensor(sample_data)  # Convert to tensor
                self.data.append(sample_data)
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def collate_fn(batch):
    # Sort the batch by the sequence lengths in descending order
    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]

    return sequences, lengths, torch.tensor(labels, dtype=torch.long)


def dynamic_batching(dataset, batch_size=32, max_len_diff=10):
    """
    Dynamically batches the dataset based on sequence lengths.
    Sequences with similar lengths are grouped together in a batch, avoiding excessive padding.

    Args:
        dataset: The dataset to be batched
        batch_size: The maximum size of each batch
        max_len_diff: The maximum allowed difference between the shortest and longest sequence in the batch

    Returns:
        A generator that yields batches of sequences
    """
    # Sort the dataset by sequence length
    sorted_data = sorted(dataset, key=lambda x: len(x[0]))

    current_batch = []
    current_lengths = []
    current_labels = []

    for seq, label in tqdm(sorted_data, desc="Creating Batches"):
        # If the current batch is empty, add the sequence
        if not current_batch:
            current_batch.append(seq)
            current_lengths.append(len(seq))
            current_labels.append(label)
            continue

        # Check if the sequence can fit in the current batch
        if len(current_batch) < batch_size:
            # Check if adding this sequence would exceed the max allowed length difference
            if max(current_lengths) - min(current_lengths) <= max_len_diff:
                current_batch.append(seq)
                current_lengths.append(len(seq))
                current_labels.append(label)
            else:
                # If length difference is too high, create a new batch
                yield current_batch, current_lengths, torch.tensor(current_labels)
                current_batch = [seq]
                current_lengths = [len(seq)]
                current_labels = [label]
        else:
            # If batch is full, yield it and start a new batch
            yield current_batch, current_lengths, torch.tensor(current_labels)
            current_batch = [seq]
            current_lengths = [len(seq)]
            current_labels = [label]

    # Yield the last batch
    if current_batch:
        yield current_batch, current_lengths, torch.tensor(current_labels)


# Create DataLoader with dynamic batching
# train_dataset = InstructionDataset(DATASET_PATH)


# Custom DataLoader with dynamic batching
def dynamic_batch_loader(dataset, batch_size=32, max_len_diff=10):
    for batch in dynamic_batching(dataset, batch_size, max_len_diff):
        sequences, lengths, labels = batch
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        yield padded_sequences, lengths, labels


# Create DataLoader
# train_loader = dynamic_batch_loader(train_dataset, batch_size=32, max_len_diff=30)

dataset = InstructionDataset(DATASET_PATH)

# Split into 90% training and 10% validation
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader for training and validation datasets
train_loader = DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
)


# Training code
def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    learning_rate=0.001,
    weight_decay=1e-5,
    max_grad_norm=1.0,
    best_model_path=None,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scaler = torch.amp.GradScaler('cuda')

    best_val_loss = float("inf")

    # Load the best model if a path is provided
    if best_model_path and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Loaded best model from {best_model_path}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for sequences, lengths, labels in tqdm(
            train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"
        ):
            sequences = [seq.to(device) for seq in sequences]
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda"):
                outputs = model(sequences, lengths)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            logger.info(f"Iteration Loss: {loss.item():.4f}")

        logger.info(
            f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}"
        )

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for sequences, lengths, labels in tqdm(val_loader, desc="Validating"):
                sequences = [seq.to(device) for seq in sequences]
                labels = labels.to(device)
                with torch.autocast(device_type="cuda"):
                    outputs = model(sequences, lengths)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        logger.info(f"Validation Loss: {val_loss}, Accuracy: {accuracy}%")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            logger.info(
                "Best model saved with validation loss: {:.4f}".format(best_val_loss)
            )


# Example Usage:
model = HybridModel(num_classes=104).to(device)  # Assuming 104 classes
train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=5,
    learning_rate=0.000001,
    best_model_path="/home/es21btech11028/IR2Vec/tryouts/Classifier_poj104/training_lst_tran/best_model.pth",
)
