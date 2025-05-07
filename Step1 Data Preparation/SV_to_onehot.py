import pandas as pd  
import csv  
import numpy as np  
import argparse  

def read_vcf(vcf_path):  
    """  
    读取VCF文件，并提取变异信息，返回一个字典，键是(染色体, 位置)，值是变异类型。  
    """  
    variant_info = {}  
    with open(vcf_path, 'r') as vcf_file:  
        reader = csv.reader(vcf_file, delimiter='\t')  
        for line in reader:  
            if line[0].startswith('#'):  
                continue  # 跳过注释行  
            chrom = line[0]  
            pos = int(line[1])  
            info = line[7]  
            vt = 'SV'  # 默认为SNP  
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
    读取BED文件，并为每一行生成one-hot编码，保存为单个NPZ文件。  
    """  
    # 读取BED文件  
    bed_df = pd.read_csv(bed_path, sep='\t', header=None, names=['chr', 'start', 'end'])  
    
    # 遍历每一行  
    arrays_to_save = []  
    for index, row in bed_df.iterrows():  
        chr_name = row['chr']  
        start = row['start']  
        end = row['end']  
        current_row_enc = []  
        # 遍历每个位置  
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
        # 将该行的编码转换为numpy数组  
        current_row_array = np.array(current_row_enc, dtype=np.int8)  
        # 准备保存到arrays_to_save  
        arrays_to_save.append(current_row_array)  
    
    # 保存所有数组到NPZ文件  
    np.savez_compressed(output_path, *arrays_to_save)  
    
    print(f"保存完成，输出文件为：{output_path}")  

def main():  
    # 解析命令行参数  
    parser = argparse.ArgumentParser(description='将VCF文件中的突变信息转换为one-hot编码，并根据BED文件输出一个NPZ文件。')  
    parser.add_argument('-b', '--bed', type=str, required=True, help='输入BED文件路径')  
    parser.add_argument('-v', '--vcf', type=str, required=True, help='输入VCF文件路径')  
    parser.add_argument('-o', '--output', type=str, default='output.npz', help='输出NPZ文件路径')  
    args = parser.parse_args()  
    
    variant_info = read_vcf(args.vcf)  
    bed_to_onehot(args.bed, variant_info, args.output)  

if __name__ == "__main__":  
    main() 
