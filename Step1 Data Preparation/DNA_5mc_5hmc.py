import numpy as np  
import argparse  
import gzip  

def parse_arguments():  
    parser = argparse.ArgumentParser(description='Process BED files to generate NPZ output.')  
    parser.add_argument('file1', type=str, help='Path to the first BED file.')  
    parser.add_argument('file2', type=str, help='Path to the second BED file (can be .bed or .bed.gz).')  
    parser.add_argument('output', type=str, help='Path to the output NPZ file.')  
    return parser.parse_args()  

def main():  
    # 解析命令行参数  
    args = parse_arguments()  
    file1 = args.file1  
    file2 = args.file2  
    output_file = args.output  

    # 读取第二个BED文件，构建positions字典  
    positions = {}  

    # 判断文件是否是压缩文件  
    if file2.endswith('.gz'):  
        open_func = gzip.open  
    else:  
        open_func = open  

    with open_func(file2, 'rt') as f:  
        for line in f:  
            line = line.strip()  
            if not line:  
                continue  
            parts = line.split()  
            if len(parts) < 11:  
                continue  # 跳过格式错误的行  
            pos = int(parts[1])  
            type_ = parts[3]  
            value = float(parts[10])  

            if pos not in positions:  
                positions[pos] = {'h': 0, 'm': 0}  
            
            if type_ == 'h':  
                positions[pos]['h'] = value  
            elif type_ == 'm':  
                positions[pos]['m'] = value  

    # 处理第一个文件，生成编码数组  
    arrays = []  

    with open(file1, 'r') as f:  
        for line in f:  
            line = line.strip()  
            if not line:  
                continue  
            parts = line.split()  
            if len(parts) < 3:  
                continue  # 跳过格式错误的行  
            chrom = parts[0]  
            start = int(parts[1])  
            end = int(parts[2])  
            M = end - start + 1  
            current_array = []  
            for pos in range(start, end):  
                h = positions.get(pos, {}).get('h', 0)  
                m = positions.get(pos, {}).get('m', 0)  
                current_array.append([h, m])  
            arrays.append(np.array(current_array))  

    # 保存到NPZ文件  
    output = {}  
    for idx, arr in enumerate(arrays):  
        output[f'array_{idx}'] = arr  

    np.savez_compressed(output_file, **output)  

if __name__ == '__main__':  
    main()  
