import sys
import numpy as np

def main():
    if len(sys.argv) != 4:
        print("Usage: python eccDNA_TPM.py <bed_file> <count_file> <output.npz>")
        sys.exit(1)

    bed_file = sys.argv[1]
    count_file = sys.argv[2]
    output_file = sys.argv[3]

    # 构建transcript到TPM的映射字典
    transcript_to_tpm = {}
    with open(count_file, 'r') as f:
        next(f)  # 跳过标题行
        for line in f:
            parts = line.strip().split()
            transcript = parts[0]
            tpm = float(parts[3])  # 转换为浮点数
            transcript_to_tpm[transcript] = tpm

    # 处理BED文件并收集TPM值
    arrays_to_save = []
    with open(bed_file, 'r') as f:
        for line in f:
            chrom, start, end = line.strip().split()[:3]
            key = f"{chrom}_{start}_{end}"
            tpm = transcript_to_tpm.get(key, 0.0)  # 未找到返回0.0
            # 将每个TPM值转换为形状为(1,1)的数组
            arrays_to_save.append(np.array([[tpm]], dtype=np.float32))

    # 保存为压缩的NPZ文件
    np.savez_compressed(output_file, *arrays_to_save)
    print(f"保存完成，输出文件为：{output_file}")

if __name__ == "__main__":
    main()
