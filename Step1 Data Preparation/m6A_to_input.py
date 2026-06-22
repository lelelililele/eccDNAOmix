import pandas as pd  
import numpy as np  
import argparse  

def read_af_file(af_path):  
    """Reads a file containing values for base positions, returning a dictionary with (chromosome, position) as keys and AF values as values, where NA or unrecorded positions are set to 0."""  
    af_dict = {}  
    with open(af_path, 'r') as af_file:  
        for line in af_file:  
            if not line.strip():  
                continue  
            parts = line.strip().split('\t')  
            if len(parts) != 2:  
                continue  
            # Extract chromosome and position information  
            chrom_pos, af_value = parts[0], parts[1]  
            # Parse chromosome and position  
            parts_chrom_pos = chrom_pos.split(':')  
            if len(parts_chrom_pos) < 2:  
                continue  # Invalid format, skip  
            chrom = parts_chrom_pos[0]  
            position_part = parts_chrom_pos[1]  
            
            # Further split position and base information  
            position_part_split = position_part.split('_')  
            if len(position_part_split) < 1:  
                continue  # Invalid format, skip  
            pos_str = position_part_split[0]  
            if pos_str.isdigit():  
                pos = int(pos_str)  
                # Convert to 0-based  
                pos_0based = pos - 1  
            else:  
                # Invalid position format, skip  
                continue  
             
            # Process AF value  
            if af_value == 'NA':  
                af_value = 0.0  
            else:  
                try:  
                    af_value = float(af_value)  
                except ValueError:  
                    af_value = 0.0  
            af_dict[(chrom, pos_0based)] = af_value  
    return af_dict  

def bed_to_input(bed_path, af_dict, output_path):  
    """Reads a BED file, generates an encoding array for each line, and saves it as an independent NPZ file."""  
    # Read BED file  
    bed_df = pd.read_csv(bed_path, sep='\t', header=None, names=['chr', 'start', 'end'])  
    
    arrays_to_save = []  
    for index, row in bed_df.iterrows():  
        chr_name = row['chr']  
        start = row['start']  
        end = row['end']  
        # Iterate through each position, generate encoding array  
        current_row_enc = []  
        for pos in range(start, end):  
            key = (chr_name, pos)  
            if key in af_dict:  
                af_value = af_dict[key]  
                current_row_enc.append([af_value])  
            else:  
                current_row_enc.append([0.0])  
        # Convert the row's encoding to a numpy array  
        current_row_array = np.array(current_row_enc, dtype=np.float32)  
        # Add to save list  
        arrays_to_save.append(current_row_array)  
    
    # Save all arrays to NPZ file  
    np.savez_compressed(output_path, *arrays_to_save)  

    print(f"Save complete. Output file: {output_path}")  

def main():  
    # Parse command-line arguments  
    parser = argparse.ArgumentParser(description='Converts BED files and corresponding base position values to NPZ format.')  
    parser.add_argument('-b', '--bed', type=str, required=True, help='Input BED file path')  
    parser.add_argument('-a', '--af', type=str, required=True, help='Input file path for base position values')  
    parser.add_argument('-o', '--output', type=str, default='output.npz', help='Output NPZ file path')  
    args = parser.parse_args()  

    af_dict = read_af_file(args.af)  
    bed_to_input(args.bed, af_dict, args.output)  

if __name__ == "__main__":  
    main()
