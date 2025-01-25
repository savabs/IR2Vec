import os
import numpy as np
import matplotlib.pyplot as plt

# Set the dataset directory
data_dir = '/home/es21btech11028/IR2Vec/tryouts/Data_things/processed_instruction_embeddings'

# Initialize a list to store lengths
lengths = []

# Loop through each class directory
for class_name in os.listdir(data_dir):
    class_path = os.path.join(data_dir, class_name)
    
    # Skip non-directory files
    if not os.path.isdir(class_path):
        continue
    
    # Loop through each file in the class directory
    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)
        
        if file_name.endswith('.npy'):
            # Load the .npy file
            instruction_vectors = np.load(file_path)
            # Append the length of the instruction list
            lengths.append(len(instruction_vectors))

# Convert lengths to a NumPy array for easier statistics computation
lengths = np.array(lengths)

# Compute statistics
max_length = lengths.max()
min_length = lengths.min()
mean_length = lengths.mean()
median_length = np.median(lengths)
std_dev = lengths.std()
percentiles = np.percentile(lengths, [75, 90, 99])  # 25th, 50th (median), and 75th percentiles

# Print statistics
print(f"Number of programs: {len(lengths)}")
print(f"Maximum length: {max_length}")
print(f"Minimum length: {min_length}")
print(f"Mean length: {mean_length:.2f}")
print(f"Median length: {median_length}")
print(f"Standard deviation: {std_dev:.2f}")
print(f"25th Percentile: {percentiles[0]}")
print(f"50th Percentile (Median): {percentiles[1]}")
print(f"75th Percentile: {percentiles[2]}")

# Plot the distribution of lengths
plt.figure(figsize=(10, 6))
plt.hist(lengths, bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Instruction List Lengths")
plt.xlabel("Length of Instruction List")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
