import pandas as pd  
import csv  
import numpy as np  
import argparse  

def read_vcf(vcf_path):  
    """  
    Reads a VCF file, extracts variant information, and returns a dictionary 
    where keys are (chromosome, position) and values are variant types.  
    """  
    variant_info = {}  
    with open(vcf_path, 'r') as vcf_file:  
        reader = csv.reader(vcf_file, delimiter='\t')  
        for line in reader:  
            if line[0].startswith('#'):  
                continue  # Skip comment lines  
            chrom = line[0]  
            pos = int(line[1])  
            info = line[7]  
            vt = 'SV'  # Default to SNP  
            params = info.split(';')  
            for param in params:  
                if param.startswith('SVTYPE='):  
                    vt = param.split('=')[1]  
                    break
            pos_0based = pos - 1  
            variant_info[(chrom, pos_0based)] = vt  
    return variant_info  

def bed_to_onehot(bed_path, variant_info, output_path):  
    """  
    Reads a BED file, generates one-hot encoding for each row, 
    and saves them into a single NPZ file.  
    """  
    # Read the BED file  
    bed_df = pd.read_csv(bed_path, sep='\t', header=None, names=['chr', 'start', 'end'])  
    
    # Iterate through each row  
    arrays_to_save = []  
    for index, row in bed_df.iterrows():  
        chr_name = row['chr']  
        start = row['start']  
        end = row['end']  
        current_row_enc = []  
        # Iterate through each position  
        for pos in range(start, end):  
            key = (chr_name, pos)  
            if key in variant_info:  
                vt = variant_info[key]  
                if vt == 'BND':  
                    current_row_enc.append([1, 0])  
                elif vt == 'INS':  
                    current_row_enc.append([0, 1])  
                else:  
                    current_row_enc.append([0, 0])  
            else:  
                current_row_enc.append([0, 0])  
        # Convert the row's encoding to a numpy array  
        current_row_array = np.array(current_row_enc, dtype=np.int8)  
        # Prepare to save to arrays_to_save  
        arrays_to_save.append(current_row_array)  
    
    # Save all arrays to the NPZ file  
    np.savez_compressed(output_path, *arrays_to_save)  
    
    print(f"Save complete. Output file: {output_path}")  

def main():  
    # Parse command-line arguments  
    parser = argparse.ArgumentParser(description='Converts mutation information from a VCF file into one-hot encoding and outputs an NPZ file based on a BED file.')  
    parser.add_argument('-b', '--bed', type=str, required=True, help='Input BED file path')  
    parser.add_argument('-v', '--vcf', type=str, required=True, help='Input VCF file path')  
    parser.add_argument('-o', '--output', type=str, default='output.npz', help='Output NPZ file path')  
    args = parser.parse_args()  
    
    variant_info = read_vcf(args.vcf)  
    bed_to_onehot(args.bed, variant_info, args.output)  

if __name__ == "__main__":  
    main()
