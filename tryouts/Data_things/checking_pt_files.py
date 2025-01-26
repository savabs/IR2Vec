import os
import torch

def inspect_pt_files(directory):
    """
    Inspect the shapes of all .pt files in a given directory
    
    Args:
        directory (str): Path to the directory containing .pt files
    """
    # Iterate through all .pt files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.pt'):
            file_path = os.path.join(directory, filename)
            try:
                # Load the file
                data = torch.load(file_path)
                
                # Print filename and shape information
                print(f"File: {filename}")
                
                # Check if data is a tensor or a dict/list containing tensors
                if isinstance(data, torch.Tensor):
                    print(f"Shape: {data.shape}")
                elif isinstance(data, (dict, list)):
                    # If it's a dict or list, try to find tensor(s)
                    def print_tensor_shapes(obj):
                        if isinstance(obj, torch.Tensor):
                            print(f"Tensor Shape: {obj.shape}")
                        elif isinstance(obj, dict):
                            for key, value in obj.items():
                                print(f"Key: {key}")
                                print_tensor_shapes(value)
                        elif isinstance(obj, list):
                            for item in obj:
                                print_tensor_shapes(item)
                    
                    print_tensor_shapes(data)
                else:
                    print("Unable to determine tensor shape")
                
                print("\n")
            
            except Exception as e:
                print(f"Error processing {filename}: {e}\n")

# Example usage
inspect_pt_files('/home/es21btech11028/IR2Vec/tryouts/Data_things/train_val_test_instrcution')