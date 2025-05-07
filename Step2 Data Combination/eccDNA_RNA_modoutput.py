import numpy as np
import argparse
import os
import sys

def merge_npz_files(input_files, output_file):
    """
    Merge multiple RNA modification NPZ files into a single 3D array with mask.
    
    Args:
        input_files (list): List of NPZ file paths to merge
        output_file (str): Output NPZ file path
    """
    # Verify input files exist
    for file in input_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Input file not found: {file}")
    
    # Read all NPZ files and collect arrays
    all_arrays = []
    max_n = 0
    M = None
    
    for file in input_files:
        with np.load(file) as data:
            # Get all arrays from the file
            arrays = [data[key] for key in data.keys()]
            all_arrays.extend(arrays)
            
            # Update maximum sequence length and feature dimension
            current_max_n = max(arr.shape[0] for arr in arrays)
            if current_max_n > max_n:
                max_n = current_max_n
            if M is None:
                M = arrays[0].shape[1]  # Feature dimension
    
    # Initialize result and mask arrays
    total_arrays = len(all_arrays)
    result = np.zeros((total_arrays, max_n, M), dtype=np.float32)
    mask = np.zeros((total_arrays, max_n), dtype=np.float32)
    
    # Fill data and mask
    for i, arr in enumerate(all_arrays):
        n_i = arr.shape[0]
        result[i, :n_i, :] = arr
        mask[i, :n_i] = 1  # Mark valid positions
    
    # Save merged arrays
    np.savez_compressed(output_file, RNAmod=result, RNAmodmask=mask)

def main():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(
        description='Merge multiple RNA modification NPZ files into a single file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-i', '--input', nargs='+', required=True,
                       help='Input NPZ files (space separated)')
    parser.add_argument('-o', '--output', required=True,
                       help='Output NPZ file path')
    
    args = parser.parse_args()
    
    try:
        # Process files
        print(f"Processing {len(args.input)} RNA modification files...")
        merge_npz_files(args.input, args.output)
        
        # Verify output
        with np.load(args.output) as data:
            print("\nMerge successful!")
            print(f"Output file: {args.output}")
            print(f"RNAmod shape: {data['RNAmod'].shape}")
            print(f"RNAmodmask shape: {data['RNAmodmask'].shape}")
            print(f"Total samples merged: {data['RNAmod'].shape[0]}")
            print(f"Max sequence length: {data['RNAmod'].shape[1]}")
            print(f"Feature dimension: {data['RNAmod'].shape[2]}")
            
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()