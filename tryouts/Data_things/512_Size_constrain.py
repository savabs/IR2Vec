import os
import numpy as np

def downsample_to_fixed_length(vectors, target_length=512):
    """
    Downsample a list of vectors to a fixed length using average pooling.
    
    Parameters:
        vectors (np.ndarray): A 2D array of shape (sequence_length, embedding_dim).
        target_length (int): The target sequence length after downsampling.
    
    Returns:
        np.ndarray: A 2D array of shape (target_length, embedding_dim).
    """
    sequence_length, embedding_dim = vectors.shape
    # if sequence_length <= target_length:
    #     # If the sequence length is already <= target, return as-is (with optional padding)
    #     return vectors

    # Compute chunk size
    chunk_size = sequence_length / target_length + 1
    
    # Perform average pooling
    downsampled = []
    for i in range(target_length):
        if sequence_length <= target_length : 
            downsampled.extend(vectors[int(i*chunk_size):-1])
            break
        start_idx = int(i * chunk_size)
        end_idx = int((i + 1) * chunk_size)
        chunk = vectors[start_idx:end_idx]
        downsampled.append(chunk.mean(axis=0))
        sequence_length -= chunk_size - 1

    return np.array(downsampled)

# Directory containing the instruction vectors
data_dir = 'instruction_embeddings/'

# Directory to save the processed instruction vectors
processed_dir = 'processed_instruction_embeddings/'
os.makedirs(processed_dir, exist_ok=True)

# Process each program file
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    processed_class_path = os.path.join(processed_dir, class_name)
    os.makedirs(processed_class_path, exist_ok=True)
    
    # Skip non-directory files
    if not os.path.isdir(class_path):
        continue
    
    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)
        
        if file_name.endswith('.npy'):
            # Load the instruction vectors
            instruction_vectors = np.load(file_path)
            
            # Downsample if necessary
            downsampled_vectors = downsample_to_fixed_length(instruction_vectors, target_length=512)
            
            # Save the processed vectors
            save_path = os.path.join(processed_class_path, file_name)
            np.save(save_path, downsampled_vectors)

print(f"Processing complete. Downsampled instruction vectors saved in '{processed_dir}' directory.")
