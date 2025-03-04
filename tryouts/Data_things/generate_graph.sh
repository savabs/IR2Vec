#!/bin/bash
# Directory containing your input .ll files (organized by class)
INPUT_ROOT="/home/es21btech11028/IR2Vec/tryouts/Data_things/data"
# Output directory for the generated graph CSV files
OUTPUT_ROOT="/home/es21btech11028/IR2Vec/tryouts/GraphOutputs"
# Path to the IR2Vec executable
IR2VEC_EXEC="/home/es21btech11028/IITH_ir2vec/llvm_env/IR2Vec/build/bin/ir2vec"

# Create output root if it doesn't exist
mkdir -p "$OUTPUT_ROOT"

# Iterate over each class folder
for class_folder in "$INPUT_ROOT"/*; do
    class_name=$(basename "$class_folder")
    # Create corresponding output folder for this class
    output_class_folder="$OUTPUT_ROOT/$class_name"
    mkdir -p "$output_class_folder"
    
    # Iterate over each .ll file in the class folder
    for ll_file in "$class_folder"/*.ll; do
        # If no .ll files exist, skip
        if [ ! -f "$ll_file" ]; then
            continue
        fi
        # Extract the base name of the input file (without extension)
        base_name=$(basename "$ll_file" .ll)
        
        echo "Processing $ll_file ..."
        # Run IR2Vec on the input file (it will generate cfg_graph.csv in the current directory)
        "$IR2VEC_EXEC" --sym --level=f -o cfg_graph.csv "$ll_file"
        
        # Rename the generated file to have the same base name as the input file
        mv cfg_graph.csv "$output_class_folder/${base_name}.csv"
        echo "Saved graph as $output_class_folder/${base_name}.csv"
    done
done

echo "All files processed."
