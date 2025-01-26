import os
import torch
import numpy as np
from typing import List, Tuple

def process_pt_files(
    input_directory: str, 
    output_directory: str,
    test_split: float = 0.1, 
    val_split: float = 0.1
):
    """
    Process .pt files with variable-length sequences
    
    Args:
        input_directory: Directory containing input .pt files
        output_directory: Directory to save processed files
        test_split: Proportion of data for test set
        val_split: Proportion of data for validation set
    """
    os.makedirs(output_directory, exist_ok=True)
    
    # Collect file paths
    pt_files = sorted([f for f in os.listdir(input_directory) if f.endswith('.pt')])
    
    # Preliminary data collection to understand distributions
    all_data_lengths = []
    all_data, all_labels, all_masks = [], [], []
    
    for filename in pt_files:
        file_path = os.path.join(input_directory, filename)
        file_data = torch.load(file_path)
        
        all_data.append(file_data['data'])
        all_labels.append(file_data['labels'])
        all_masks.append(file_data['masks'])
        
        # Track sequence lengths
        all_data_lengths.append(file_data['data'].shape[1])
    
    # Determine padding length (e.g., 95th percentile)
    max_length = 512
    
    # Pad and combine data
    padded_data, padded_labels, padded_masks = [], [], []
    
    for data, labels, masks in zip(all_data, all_labels, all_masks):
        batch_size, seq_length, feature_dim = data.shape
        
        # Create padded tensors
        padded_batch_data = torch.zeros(batch_size, max_length, feature_dim)
        padded_batch_masks = torch.zeros(batch_size, max_length, dtype=torch.bool)
        
        for i in range(batch_size):
            actual_length = min(seq_length, max_length)
            padded_batch_data[i, :actual_length] = data[i, :actual_length]
            padded_batch_masks[i, :actual_length] = 1
        
        padded_data.append(padded_batch_data)
        padded_labels.append(labels)
        padded_masks.append(padded_batch_masks)
    
    # Concatenate all padded data
    final_data = torch.cat(padded_data)
    final_labels = torch.cat(padded_labels)
    final_masks = torch.cat(padded_masks)
    
    # Shuffle data
    indices = torch.randperm(final_data.shape[0])
    shuffled_data = final_data[indices]
    shuffled_labels = final_labels[indices]
    shuffled_masks = final_masks[indices]
    
    # Split data
    total_samples = shuffled_data.shape[0]
    test_end = int(total_samples * test_split)
    val_end = test_end + int(total_samples * val_split)
    
    # Save datasets
    datasets = {
        'train': (shuffled_data[val_end:], shuffled_labels[val_end:], shuffled_masks[val_end:]),
        'val': (shuffled_data[test_end:val_end], shuffled_labels[test_end:val_end], shuffled_masks[test_end:val_end]),
        'test': (shuffled_data[:test_end], shuffled_labels[:test_end], shuffled_masks[:test_end])
    }
    
    # Save each dataset
    for split, (data, labels, masks) in datasets.items():
        output_file = os.path.join(output_directory, f'{split}_data.pt')
        torch.save({
            'data': data,
            'labels': labels,
            'masks': masks
        }, output_file)
        print(f"Saved {split} data: {data.shape}")

# Example usage
process_pt_files('/home/es21btech11028/IR2Vec/tryouts/Data_things/processed_data', '/home/es21btech11028/IR2Vec/tryouts/Data_things/train_val_test_instrcution')