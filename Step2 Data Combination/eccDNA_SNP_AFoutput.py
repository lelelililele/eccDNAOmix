import numpy as np
import argparse
import os
import sys
from typing import List

def validate_input_files(input_files: List[str]) -> None:
    """Validate that all input files exist"""
    missing_files = [f for f in input_files if not os.path.exists(f)]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} input files not found:\n" + 
            "\n".join(missing_files)
        )

def merge_npz_files(input_files: List[str], output_file: str) -> None:
    """
    Merge multiple SNP allele frequency NPZ files into a single 3D array with mask
    
    Args:
        input_files: List of NPZ file paths to merge
        output_file: Output NPZ file path
    """
    validate_input_files(input_files)
    
    # Read all NPZ files and collect arrays
    all_arrays = []
    max_length = 0  # Maximum sequence length
    num_features = None  # Number of features per position
    
    for file in input_files:
        with np.load(file) as data:
            # Get all arrays from the file
            arrays = [data[key] for key in data.files]
            all_arrays.extend(arrays)
            
            # Update maximum sequence length and feature dimension
            current_max = max(arr.shape[0] for arr in arrays)
            max_length = max(max_length, current_max)
            
            if num_features is None:
                num_features = arrays[0].shape[1]  # Feature dimension
    
    # Initialize result and mask arrays
    num_samples = len(all_arrays)
    merged_data = np.zeros((num_samples, max_length, num_features), dtype=np.float32)
    mask = np.zeros((num_samples, max_length), dtype=np.float32)
    
    # Fill data and mask
    for idx, arr in enumerate(all_arrays):
        seq_length = arr.shape[0]
        merged_data[idx, :seq_length, :] = arr
        mask[idx, :seq_length] = 1  # Mark valid positions
    
    # Save merged arrays with compression
    np.savez_compressed(
        output_file,
        DNASNP=merged_data,
        DNASNPmask=mask
    )

def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple SNP allele frequency NPZ files into one",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--input',
        nargs='+',
        required=True,
        help='Input NPZ files (space separated)'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output NPZ file path'
    )
    
    args = parser.parse_args()
    
    try:
        print(f"Merging {len(args.input)} SNP allele frequency files...")
        merge_npz_files(args.input, args.output)
        
        # Verify and report output
        with np.load(args.output) as data:
            print("\nMerge successful!")
            print(f"Output file: {args.output}")
            print(f"DNASNP shape: {data['DNASNP'].shape}")
            print(f"DNASNPmask shape: {data['DNASNPmask'].shape}")
            print(f"Total samples: {data['DNASNP'].shape[0]}")
            print(f"Max sequence length: {data['DNASNP'].shape[1]}")
            print(f"Features per position: {data['DNASNP'].shape[2]}")
            
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()