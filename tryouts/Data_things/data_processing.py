import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from tqdm import tqdm  # For progress bars

# Set the dataset path
data_dir = '/home/es21btech11028/IR2Vec/tryouts/Data_things/processed_instruction_embeddings/'

# Initialize class names
class_names = [class_name for class_name in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, class_name))]
print(f"Class Names: {class_names}")

def process_and_save_batch(batch_data, batch_labels, batch_file_name, progress_desc):
    """Save processed data batch to disk."""
    data_tensors = [torch.tensor(seq, dtype=torch.float32) for seq in batch_data]
    labels_tensor = torch.tensor(batch_labels, dtype=torch.int64)
    
    # Ensure labels are within bounds
    if any(label >= len(class_names) for label in batch_labels):
        raise ValueError(f"Invalid label found in batch: {batch_labels}")
    
    padded_data = pad_sequence(data_tensors, batch_first=True, padding_value=0.0)
    attention_masks = (padded_data != 0).int()
    
    labels_one_hot = torch.nn.functional.one_hot(labels_tensor, num_classes=len(class_names)).float()
    
    torch.save(
        {'data': padded_data, 'labels': labels_one_hot, 'masks': attention_masks}, 
        batch_file_name
    )
    progress_desc.update(len(batch_data))  # Update progress bar

def load_class_data_in_chunks(class_index, class_name, batch_size=1000):
    """Load .npy files from a class directory in chunks."""
    class_path = os.path.join(data_dir, class_name)
    if not os.path.isdir(class_path):
        return
    
    files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
    num_files = len(files)
    batch_data, batch_labels = [], []
    batch_id = 0

    # Create a progress bar for the current class
    with tqdm(total=num_files, desc=f"Processing class: {class_name}", unit="files") as progress_desc:
        for i, file_name in enumerate(files):
            file_path = os.path.join(class_path, file_name)
            file_data = np.load(file_path)
            batch_data.append(file_data)
            batch_labels.append(class_index)
            
            # Process and save in chunks
            if (i + 1) % batch_size == 0 or i == num_files - 1:
                batch_file_name = f"processed_data/{class_name}_batch_{batch_id}.pt"
                process_and_save_batch(batch_data, batch_labels, batch_file_name, progress_desc)
                batch_data, batch_labels = [], []
                batch_id += 1

# Process each class sequentially
os.makedirs('processed_data', exist_ok=True)
for class_index, class_name in enumerate(class_names):
    load_class_data_in_chunks(class_index, class_name)

# Combine batches for splitting and saving
print("Combining and splitting data...")
data_files = [f for f in os.listdir('processed_data') if f.endswith('.pt')]
all_data, all_labels, all_masks = [], [], []

# Progress bar for combining files
with tqdm(total=len(data_files), desc="Combining batches", unit="files") as progress_desc:
    for file_name in data_files:
        batch = torch.load(os.path.join('processed_data', file_name))
        all_data.append(batch['data'])
        all_labels.append(batch['labels'])
        all_masks.append(batch['masks'])
        progress_desc.update(1)

# Concatenate all data
data_padded = torch.cat(all_data, dim=0)
labels_one_hot = torch.cat(all_labels, dim=0)
attention_masks = torch.cat(all_masks, dim=0)

# Split the dataset into training, validation, and test sets
print("Splitting data into training, validation, and test sets...")
X_train, X_temp, y_train, y_temp, mask_train, mask_temp = train_test_split(
    data_padded, labels_one_hot, attention_masks, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test, mask_val, mask_test = train_test_split(
    X_temp, y_temp, mask_temp, test_size=0.5, random_state=42
)

# Save final processed datasets
torch.save({'X_train': X_train, 'y_train': y_train, 'mask_train': mask_train}, 
           'processed_data/training_data.pt')
torch.save({'X_val': X_val, 'y_val': y_val, 'mask_val': mask_val}, 
           'processed_data/validation_data.pt')
torch.save({'X_test': X_test, 'y_test': y_test, 'mask_test': mask_test}, 
           'processed_data/test_data.pt')

print("Processing complete. Final datasets saved in 'processed_data' folder.")
