import sys  
import os  
from Bio import SeqIO  
import numpy as np  

def bed_to_onehot(bed_file, fasta_file, output_npz='output.npz', output_bed='extended.bed', output_fasta='extended_sequences.fasta'):  
    """Extends each entry in the BED file by 100bp upstream and downstream, extracts the sequence, and converts it to one-hot encoding.  
    Outputs the extended BED file, FASTA file, and a large 2D array `encodings` to an NPZ file."""  
    # Load FASTA file into a dictionary, where keys are chromosome names and values are full sequences  
    print("Loading FASTA file...")  
    fasta_sequences = {}  
    for record in SeqIO.parse(fasta_file, "fasta"):  
        chrom = record.id  
        fasta_sequences[chrom] = record.seq  
    print(f"FASTA file loaded, total {len(fasta_sequences)} chromosomes.")  

    # Prepare to save all one-hot encodings  
    all_sequences = []  # Store encoding for each row, shape is (N, M, 4)  

    # Output extended BED entries and FASTA entries  
    extended_bed_entries = []  
    extended_fasta_entries = []  

    # Process BED file line by line  
    print("Starting to process BED file...")  
    with open(bed_file, 'r') as bed:  
        for line in bed:  
            line = line.strip()  
            if not line:  
                continue  

            parts = line.split()  
            if len(parts) < 3:  
                print(f"Skipping invalid BED line: {line}")  
                continue  

            chrom = parts[0]  
            start = int(parts[1])  
            end = int(parts[2])  

            # Check if chromosome exists  
            if chrom not in fasta_sequences:  
                print(f"Warning: Chromosome {chrom} does not exist in FASTA file, skipping entry.")  
                continue  

            seq = fasta_sequences[chrom]  
            chrom_length = len(seq)  

            # Calculate extended positions, note it is 1-based  
            new_start = max(0, start - 100)  
            new_end = min(end + 100, chrom_length)  

            # Ensure the interval is valid  
            if new_start >= new_end:  
                print(f"Warning: Extended interval {chrom}:{new_start}-{new_end} is invalid, skipping entry.")  
                continue  

            # Output extended BED entry  
            extended_bed_entries.append(f"{chrom}\t{new_start}\t{new_end}")  

            # Extract the required sequence fragment (converted to 0-based)  
            extracted_seq = seq[new_start:new_end]  # Convert to 0-based  

            # Convert the extracted sequence to one-hot encoding  
            one_hot = []  
            for base in extracted_seq:  
                if base == 'A':  
                    one_hot.append([1, 0, 0, 0])  
                elif base == 'T':  
                    one_hot.append([0, 1, 0, 0])  
                elif base == 'C':  
                    one_hot.append([0, 0, 1, 0])  
                elif base == 'G':  
                    one_hot.append([0, 0, 0, 1])  
                else:  
                    one_hot.append([0, 0, 0, 0])  
                    print(f"Warning: Unknown base {base} found, encoding as [0, 0, 0, 0] in the sequence.")  

            # Add the row's encoding to the all_sequences list  
            all_sequences.append(np.array(one_hot, dtype=np.int8))  

            # Prepare FASTA entry  
            fasta_id = f"{chrom}_{new_start}_{new_end}"  
            extended_fasta_entries.append(f">{fasta_id}\n{extracted_seq}\n")  

    # Write extended BED entries to file  
    with open(output_bed, 'w') as bed_output:  
        for entry in extended_bed_entries:  
            bed_output.write(f"{entry}\n")  
    print(f"Extended BED entries saved to {output_bed}.")  

    # Write extracted sequences to FASTA file  
    with open(output_fasta, 'w') as fasta_output:  
        for entry in extended_fasta_entries:  
            fasta_output.write(entry)  
    print(f"Extracted sequences saved to {output_fasta}.")  

    # Save all one-hot encodings to NPZ file, suitable for varying entry lengths  
    if all_sequences:  
        # Create a dictionary to save the encoding array of each entry  
        encodings_dict = {}  
        for i, seq_array in enumerate(all_sequences):  
            # Generate a unique key, e.g., using chromosome, start, and end positions  
            parts = extended_bed_entries[i].split('\t')  
            chrom, start, end = parts[0], parts[1], parts[2]  
            key = f"{chrom}_{start}_{end}"  
            encodings_dict[key] = seq_array  

        # Save dictionary to NPZ file  
        np.savez_compressed(output_npz, **encodings_dict)  
        print(f"Complete! One-hot encoding results saved to {output_npz}.")  
    else:  
        print("No encodings generated, perhaps all lines in BED file are invalid.")  
        np.savez_compressed(output_npz)  
        print(f"Created empty NPZ file {output_npz}.")  

if __name__ == "__main__":  
    # Parse command-line arguments  
    if len(sys.argv) != 4:  
        print("Usage: python bed_to_onehot.py <bed_file> <fasta_file> <output_npz>")  
        sys.exit(1)  
    
    bed_file = sys.argv[1]  
    fasta_file = sys.argv[2]  
    output_npz = sys.argv[3]  

    # Generate output filenames  
    base_name = os.path.splitext(os.path.basename(bed_file))[0]  
    output_bed = f"{base_name}_extended.bed"  
    output_fasta = f"{base_name}_extended_sequences.fasta"  

    # Call processing function  
    bed_to_onehot(  
        bed_file=bed_file,  
        fasta_file=fasta_file,  
        output_npz=output_npz,  
        output_bed=output_bed,  
        output_fasta=output_fasta  
    )
