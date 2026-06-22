import sys
import numpy as np

def main():
    if len(sys.argv) != 4:
        print("Usage: python eccDNA_TPM.py <bed_file> <count_file> <output.npz>")
        sys.exit(1)

    bed_file = sys.argv[1]
    count_file = sys.argv[2]
    output_file = sys.argv[3]

    # Build transcript to TPM mapping dictionary
    transcript_to_tpm = {}
    with open(count_file, 'r') as f:
        next(f)  # Skip header row
        for line in f:
            parts = line.strip().split()
            transcript = parts[0]
            tpm = float(parts[3])  # Convert to float
            transcript_to_tpm[transcript] = tpm

    # Process BED file and collect TPM values
    arrays_to_save = []
    with open(bed_file, 'r') as f:
        for line in f:
            chrom, start, end = line.strip().split()[:3]
            key = f"{chrom}_{start}_{end}"
            tpm = transcript_to_tpm.get(key, 0.0)  # Return 0.0 if not found
            # Convert each TPM value to an array of shape (1,1)
            arrays_to_save.append(np.array([[tpm]], dtype=np.float32))

    # Save as compressed NPZ file
    np.savez_compressed(output_file, *arrays_to_save)
    print(f"Save complete. Output file: {output_file}")

if __name__ == "__main__":
    main()
