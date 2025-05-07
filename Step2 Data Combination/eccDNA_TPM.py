import numpy as np
import argparse
import os
import sys
from typing import List, Tuple

def validate_inputs(input_files: List[str]) -> None:
    """Verify all input files exist and are accessible"""
    missing_files = []
    for f in input_files:
        if not os.path.exists(f):
            missing_files.append(f)
    if missing_files:
        raise FileNotFoundError(
            f"Missing {len(missing_files)} input files:\n" + 
            "\n".join(f"- {f}" for f in missing_files)
        )

def load_and_validate_shapes(input_files: List[str]) -> Tuple[List[np.ndarray], int, int]:
    """
    Load arrays from NPZ files and validate consistent feature dimensions
    
    Returns:
        Tuple of (all_arrays, max_sequence_length, num_features)
    """
    all_arrays = []
    max_len = 0
    num_features = None
    
    for file in input_files:
        with np.load(file) as data:
            arrays = [data[key] for key in data.files]
            all_arrays.extend(arrays)
            
            # Track maximum sequence length
            current_max = max(arr.shape[0] for arr in arrays)
            max_len = max(max_len, current_max)
            
            # Validate consistent feature dimensions
            if num_features is None:
                num_features = arrays[0].shape[1]
            for arr in arrays:
                if arr.shape[1] != num_features:
                    raise ValueError(
                        f"Inconsistent feature dimensions in {file}: "
                        f"expected {num_features}, got {arr.shape[1]}"
                    )
    
    return all_arrays, max_len, num_features

def merge_tpm_data(input_files: List[str], output_file: str) -> None:
    """
    Merge TPM data from multiple NPZ files into a single compressed file
    
    Args:
        input_files: List of input NPZ file paths
        output_file: Path for output NPZ file
    """
    validate_inputs(input_files)
    all_arrays, max_len, num_features = load_and_validate_shapes(input_files)
    
    # Initialize merged arrays
    num_samples = len(all_arrays)
    merged_data = np.zeros((num_samples, max_len, num_features), dtype=np.float32)
    mask = np.zeros((num_samples, max_len), dtype=np.float32)
    
    # Fill data and mask
    for idx, arr in enumerate(all_arrays):
        seq_len = arr.shape[0]
        merged_data[idx, :seq_len, :] = arr
        mask[idx, :seq_len] = 1  # 1 indicates valid data
    
    # Save with compression
    np.savez_compressed(
        output_file,
        RNATPM=merged_data,
        RNATPMmask=mask
    )

def main():
    # Set up command line interface
    parser = argparse.ArgumentParser(
        description="Merge multiple TPM (Transcripts Per Million) NPZ files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-i', '--input',
        nargs='+',
        required=True,
        help='Input NPZ files containing TPM data'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output NPZ file path'
    )
    
    args = parser.parse_args()
    
    try:
        print(f"Processing {len(args.input)} TPM files...")
        merge_tpm_data(args.input, args.output)
        
        # Verify and report results
        with np.load(args.output) as data:
            print("\nSuccessfully merged TPM data:")
            print(f"Output file: {args.output}")
            print(f"RNATPM shape: {data['RNATPM'].shape}")
            print(f"RNATPMmask shape: {data['RNATPMmask'].shape}")
            print(f"Total samples: {data['RNATPM'].shape[0]}")
            print(f"Max sequence length: {data['RNATPM'].shape[1]}")
            print(f"Features per position: {data['RNATPM'].shape[2]}")
            
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()