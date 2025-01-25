import os
import numpy as np
from sklearn.model_selection import train_test_split

# Set the dataset path
data_dir = 'program_embeddings/'

# Initialize lists for data and labels
data = []
labels = []
class_names = []

# Loop through each class directory
for class_index, class_name in enumerate(sorted(os.listdir(data_dir))):
    class_path = os.path.join(data_dir, class_name)
    
    # Skip non-directory files
    if not os.path.isdir(class_path):
        continue
    
    class_names.append(class_name)
    for file_name in os.listdir(class_path):
        file_path = os.path.join(class_path, file_name)
        
        if file_name.endswith('.npy'):
            # Load the .npy file
            file_data = np.load(file_path)
            data.append(file_data)
            labels.append(class_index)

# Convert data and labels to numpy arrays
data = np.array(data)

# Convert labels to one-hot encoding
num_classes = len(class_names)
labels = np.array(labels)
labels_one_hot = np.eye(num_classes)[labels]  # One-hot encoding

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(data, labels_one_hot, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save processed datasets
os.makedirs('processed_data', exist_ok=True)
np.save('processed_data/X_train.npy', X_train)
np.save('processed_data/y_train.npy', y_train)
np.save('processed_data/X_val.npy', X_val)
np.save('processed_data/y_val.npy', y_val)
np.save('processed_data/X_test.npy', X_test)
np.save('processed_data/y_test.npy', y_test)

print("Processing complete. Data and one-hot encoded labels saved in 'processed_data' folder.")
