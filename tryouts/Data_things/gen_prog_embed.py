import os
import numpy as np
import ir2vec

# Base directory where the .ll files are stored
LLVM_OUTPUT_BASE_DIR = "/home/es21btech11028/IR2Vec/tryouts/Data_things/data"

# Base directory where the program embeddings will be saved
INSTRUCTION_EMBEDDINGS_BASE_DIR = "instruction_embeddings"

# Create the base directory for embeddings if it doesn't exist
os.makedirs(INSTRUCTION_EMBEDDINGS_BASE_DIR, exist_ok=True)

# Iterate through the directory structure of the LLVM output
for root, dirs, files in os.walk(LLVM_OUTPUT_BASE_DIR):
    for file in files:
        if file.endswith(".ll"):
            # Full path of the .ll file
            ll_file_path = os.path.join(root, file)

            # Determine the relative path from LLVM_OUTPUT_BASE_DIR
            relative_path = os.path.relpath(root, LLVM_OUTPUT_BASE_DIR)

            # Create the corresponding directory structure in PROGRAM_EMBEDDINGS_BASE_DIR
            embedding_output_dir = os.path.join(INSTRUCTION_EMBEDDINGS_BASE_DIR, relative_path)
            os.makedirs(embedding_output_dir, exist_ok=True)

            # Define the output .npy file path
            embedding_file_path = os.path.join(embedding_output_dir, f"{file}_program_embedding.npy")

            # Process the .ll file to generate the program embedding
            print(f"Processing {ll_file_path}...")
            try:
                # Initialize IR2Vec
                initObj = ir2vec.initEmbedding(ll_file_path, "fa", "p")

                # Get the program vector
                # progVector = initObj.getProgramVector()
                instruction_matrix = initObj.getInstructionVectors()

                # Save the program vector as a .npy file
                np.save(embedding_file_path, instruction_matrix)
                print(f"Saved program embedding to {embedding_file_path}")

            except Exception as e:
                print(f"Error processing {ll_file_path}: {e}")
