import os
import numpy as np

# Set the dataset path
data_dir = 'instruction_embeddings'

# Initialize variables for analysis
class_details = {}
data_shapes = []

# Loop through each class directory
for class_index, class_name in enumerate(sorted(os.listdir(data_dir))):
    class_path = os.path.join(data_dir, class_name)
    
    # Skip non-directory files
    if not os.path.isdir(class_path):
        continue
    
    print(f"Class {class_index}: {class_name}")
    num_samples = 0
    sample_shapes = []
    
    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)
        
        if file_name.endswith('.npy'):
            # Load the .npy file
            file_data = np.load(file_path)
            
            # Record sample shape
            sample_shapes.append(file_data.shape)
            data_shapes.append(file_data.shape)
            num_samples += 1
    
    # Record details for the class
    class_details[class_name] = {
        "num_samples": num_samples,
        "sample_shapes": sample_shapes,
    }

# Print class-wise details
print("\nDataset Overview:")
for class_name, details in class_details.items():
    print(f"\nClass: {class_name}")
    print(f"  Number of samples: {details['num_samples']}")
    print(f"  Sample shapes: {set(details['sample_shapes'])}")

# Analyze the overall data structure
print("\nGlobal Analysis:")
unique_shapes = set(data_shapes)
print(f"Total samples across all classes: {len(data_shapes)}")
print(f"Unique shapes of samples: {unique_shapes}")

# Check for the most common shape
from collections import Counter
shape_counts = Counter(data_shapes)
most_common_shape = shape_counts.most_common(1)[0]
print(f"Most common shape: {most_common_shape[0]} (Appears {most_common_shape[1]} times)")

# Handle edge cases if required
if len(unique_shapes) > 1:
    print("\nWarning: Samples have varying shapes. Consider padding or truncating for consistency.")
