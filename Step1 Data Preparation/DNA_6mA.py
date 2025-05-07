import gzip
import numpy as np
import argparse
from collections import defaultdict

def parse_bed_file(bed_file):
    """Parse a BED file and return a dictionary of regions."""
    regions = defaultdict(list)
    opener = gzip.open if bed_file.endswith('.gz') else open
    with opener(bed_file, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            name = parts[3] if len(parts) > 3 else None
            strand = parts[5] if len(parts) > 5 else '+'
            regions[chrom].append((start, end, name, strand))
    return regions

def parse_value_file(value_file):
    """Parse the second BED file and return a dictionary of positions and their values."""
    values = defaultdict(dict)
    opener = gzip.open if value_file.endswith('.gz') else open
    with opener(value_file, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            chrom = parts[0]
            pos = int(parts[1])  # 0-based start position
            value = float(parts[10])  # 11th column (0-based index 10)
            values[chrom][pos] = value
    return values

def process_regions(regions, values):
    """Process regions and create numpy arrays based on the values."""
    arrays = []
    for chrom in regions:
        for start, end, name, strand in regions[chrom]:
            length = end - start
            arr = np.zeros((length, 1), dtype=np.float32)
            
            for i in range(length):
                pos = start + i
                if pos in values[chrom]:
                    arr[i, 0] = values[chrom][pos]
            
            arrays.append(arr)
    return arrays

def main():
    parser = argparse.ArgumentParser(description='Process BED files and create NPZ file.')
    parser.add_argument('-a', '--bed_file', required=True, help='First BED file (regions)')
    parser.add_argument('-b', '--value_file', required=True, help='Second BED file (values)')
    parser.add_argument('-o', '--output', required=True, help='Output NPZ file')
    args = parser.parse_args()

    # Parse input files
    regions = parse_bed_file(args.bed_file)
    values = parse_value_file(args.value_file)

    # Process regions and create arrays
    arrays = process_regions(regions, values)

    # Save arrays to NPZ file
    np.savez_compressed(args.output, *arrays)
    print(f"Saved {len(arrays)} arrays to {args.output}")

if __name__ == '__main__':
    main()
