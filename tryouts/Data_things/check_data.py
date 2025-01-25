import numpy as np
# import matplotlib.pyplot as plt

# Path to preprocessed data
data_dir = 'processed_data/'

# Load the preprocessed data
X_train = np.load(data_dir + 'X_train.npy')
y_train = np.load(data_dir + 'y_train.npy')

# Print dataset information
print(f"Shape of X_train: {X_train.shape}")  # Example: (num_samples, data_shape)
print(f"Shape of y_train: {y_train.shape}")  # Example: (num_samples, num_classes)
print(f"Number of classes: {y_train.shape[1]}")

# Inspect a few examples
def explore_preprocessed_data(X, y, num_samples=5):
    for i in range(num_samples):
        print(f"\n--- Example {i+1} ---")
        print(f"One-hot encoded label: {y[i]}")
        print(f"Class index: {np.argmax(y[i])}")
        
        # Check if the data can be visualized (e.g., images)
        if len(X[i].shape) == 2 or len(X[i].shape) == 3:
            plt.imshow(X[i], cmap='gray' if len(X[i].shape) == 2 else None)
            plt.title(f"Class Index: {np.argmax(y[i])}")
            plt.show()
        else:
            print(f"Data: {X[i]}")
            print("Visualization not available for this data type.")

# Explore a few training samples
explore_preprocessed_data(X_train, y_train, num_samples=5)
