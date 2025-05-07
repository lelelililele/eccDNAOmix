import numpy as np
import argparse

def merge_npz_files(input_files, output_file):
    """
    合并多个NPZ文件中的数组到一个三维数组中，并生成对应的掩码数组。
    
    参数:
        input_files (list): 包含要合并的NPZ文件路径的列表。
        output_file (str): 输出的NPZ文件路径。
    """
    # 读取所有NPZ文件并收集所有数组
    all_arrays = []
    max_n = 0
    M = None
    
    for file in input_files:
        with np.load(file) as data:
            # 假设每个NPZ文件中的数组键名为'arr_0', 'arr_1'等
            arrays = [data[key] for key in data.keys()]
            all_arrays.extend(arrays)
            
            # 更新最大n值和特征数量M
            current_max_n = max(arr.shape[0] for arr in arrays)
            if current_max_n > max_n:
                max_n = current_max_n
            if M is None:
                M = arrays[0].shape[1]
    
    # 计算总数组数量
    total_arrays = len(all_arrays)
    
    # 初始化结果数组和掩码数组
    result = np.zeros((total_arrays, max_n, M), dtype=np.float32)
    mask = np.zeros((total_arrays, max_n), dtype=np.float32)
    
    # 填充数据和掩码
    for i, arr in enumerate(all_arrays):
        n_i = arr.shape[0]
        result[i, :n_i, :] = arr
        mask[i, :n_i] = 1  # 标记有效数据位置为1
    
    # 保存合并后的结果和掩码数组
    np.savez_compressed(output_file, OPDNA5mc=result, OPDNA5mcmask=mask)

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Merge multiple NPZ files into a single NPZ file.')
    parser.add_argument('-i', '--input', nargs='+', required=True, 
                        help='Input NPZ files to merge')
    parser.add_argument('-o', '--output', required=True,
                        help='Output NPZ file path')
    
    # 解析参数
    args = parser.parse_args()
    
    # 调用合并函数
    merge_npz_files(args.input, args.output)
    print(f"Successfully merged {len(args.input)} files into {args.output}")

if __name__ == "__main__":
    main()