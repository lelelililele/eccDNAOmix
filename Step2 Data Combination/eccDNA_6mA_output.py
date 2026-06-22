import numpy as np
import argparse

def merge_npz_files(input_files, output_file):
    """
    Merges arrays from multiple NPZ files into a 3D array and generates a corresponding mask array.
    
    Parameters:
        input_files (list): A list containing the paths of the NPZ files to merge.
        output_file (str): The output NPZ file path.
    """
    # Read all NPZ files and collect all arrays
    all_arrays = []
    max_n = 0
    M = None
    
    for file in input_files:
        with np.load(file) as data:
            # Assume the array keys in each NPZ file are 'arr_0', 'arr_1', etc.
            arrays = [data[key] for key in data.keys()]
            all_arrays.extend(arrays)
            
            # Update the maximum n value and feature count M
            current_max_n = max(arr.shape[0] for arr in arrays)
            if current_max_n > max_n:
                max_n = current_max_n
            if M is None:
                M = arrays[0].shape[1]
    
    # Calculate the total number of arrays
    total_arrays = len(all_arrays)
    
    # Initialize the result array and mask array
    result = np.zeros((total_arrays, max_n, M), dtype=np.float32)
    mask = np.zeros((total_arrays, max_n), dtype=np.float32)
    
    # Fill data and mask
    for i, arr in enumerate(all_arrays):
        n_i = arr.shape[0]
        result[i, :n_i, :] = arr
        mask[i, :n_i] = 1  # Mark valid data positions as 1
    
    # Save the merged result and mask arrays
    np.savez_compressed(output_file, OPDNA6mA=result, OPDNA6mAmask=mask)

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Merge multiple 6mA NPZ files into a single NPZ file.')
    parser.add_argument('-i', '--input', nargs='+', required=True, 
                        help='Input NPZ files to merge (space separated)')
    parser.add_argument('-o', '--output', required=True,
                        help='Output NPZ file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Call the merge function
    merge_npz_files(args.input, args.output)
    print(f"Successfully merged {len(args.input)} 6mA files into {args.output}")

if __name__ == "__main__":
    main()
