import pandas as pd  
import numpy as np  
import argparse  

def read_af_file(af_path):  
    """读取包含碱基位置对应值的文件，返回一个字典，键是(染色体, 位置)，值是对应的AF值，NA或没有记录的位置设为0。"""  
    af_dict = {}  
    with open(af_path, 'r') as af_file:  
        for line in af_file:  
            if not line.strip():  
                continue  
            parts = line.strip().split('\t')  
            if len(parts) != 2:  
                continue  
            # 提取染色体和位置信息  
            chrom_pos, af_value = parts[0], parts[1]  
            # 解析染色体和位置  
            parts_chrom_pos = chrom_pos.split(':')  
            if len(parts_chrom_pos) < 2:  
                continue  # 无效格式，跳过  
            chrom = parts_chrom_pos[0]  
            position_part = parts_chrom_pos[1]  
            
            # 进一步分割位置和碱基信息  
            position_part_split = position_part.split('_')  
            if len(position_part_split) < 1:  
                continue  # 无效格式，跳过  
            pos_str = position_part_split[0]  
            if pos_str.isdigit():  
                pos = int(pos_str)  
                # 转换为0-based  
                pos_0based = pos - 1  
            else:  
                # 无效的位置格式，跳过  
                continue  
             
            # 处理AF值  
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
    """读取BED文件，并为每一行生成一个编码数组，保存为独立的NPZ文件。"""  
    # 读取BED文件  
    bed_df = pd.read_csv(bed_path, sep='\t', header=None, names=['chr', 'start', 'end'])  
    
    arrays_to_save = []  
    for index, row in bed_df.iterrows():  
        chr_name = row['chr']  
        start = row['start']  
        end = row['end']  
        # 遍历每个位置，生成编码数组  
        current_row_enc = []  
        for pos in range(start, end):  
            key = (chr_name, pos)  
            if key in af_dict:  
                af_value = af_dict[key]  
                current_row_enc.append([af_value])  
            else:  
                current_row_enc.append([0.0])  
        # 将该行的编码转换为numpy数组  
        current_row_array = np.array(current_row_enc, dtype=np.float32)  
        # 添加到保存列表  
        arrays_to_save.append(current_row_array)  
    
    # 保存所有数组到NPZ文件  
    np.savez_compressed(output_path, *arrays_to_save)  

    print(f"保存完成，输出文件为：{output_path}")  

def main():  
    # 解析命令行参数  
    parser = argparse.ArgumentParser(description='将BED文件和碱基位置对应的值转换为NPZ格式。')  
    parser.add_argument('-b', '--bed', type=str, required=True, help='输入BED文件路径')  
    parser.add_argument('-a', '--af', type=str, required=True, help='输入碱基位置对应值的文件路径')  
    parser.add_argument('-o', '--output', type=str, default='output.npz', help='输出NPZ文件路径')  
    args = parser.parse_args()  

    af_dict = read_af_file(args.af)  
    bed_to_input(args.bed, af_dict, args.output)  

if __name__ == "__main__":  
    main()  
