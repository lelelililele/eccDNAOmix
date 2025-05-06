import sys  
import os  
from Bio import SeqIO  
import numpy as np  

def bed_to_onehot(bed_file, fasta_file, output_npz='output.npz', output_bed='extended.bed', output_fasta='extended_sequences.fasta'):  
    """将BED文件中的每个条目扩展上下游100bp，提取序列并转换为one-hot编码，  
    输出扩展后的BED文件、FASTA文件和一个大的二维数组`encodings`到NPZ文件中。"""  
    # 加载FASTA文件到字典，键是染色体名，值是完整的序列  
    print("加载FASTA文件...")  
    fasta_sequences = {}  
    for record in SeqIO.parse(fasta_file, "fasta"):  
        chrom = record.id  
        fasta_sequences[chrom] = record.seq  
    print(f"已加载FASTA文件，共 {len(fasta_sequences)} 个染色体.")  

    # 准备保存所有的one-hot编码  
    all_sequences = []  # 存储每行的编码，形状为 (N, M, 4)  

    # 输出扩展后的BED条目和FASTA条目  
    extended_bed_entries = []  
    extended_fasta_entries = []  

    # 逐行处理BED文件  
    print("开始处理BED文件...")  
    with open(bed_file, 'r') as bed:  
        for line in bed:  
            line = line.strip()  
            if not line:  
                continue  

            parts = line.split()  
            if len(parts) < 3:  
                print(f"跳过无效的BED行：{line}")  
                continue  

            chrom = parts[0]  
            start = int(parts[1])  
            end = int(parts[2])  

            # 检查染色体是否存在  
            if chrom not in fasta_sequences:  
                print(f"警告：染色体 {chrom} 不存在于FASTA文件中，跳过该条目。")  
                continue  

            seq = fasta_sequences[chrom]  
            chrom_length = len(seq)  

            # 计算扩展后的位置，注意是1-based  
            new_start = max(0, start - 100)  
            new_end = min(end + 100, chrom_length)  

            # 确保区间有效  
            if new_start >= new_end:  
                print(f"警告：扩展后的区间 {chrom}:{new_start}-{new_end} 无效，跳过该条目。")  
                continue  

            # 输出扩展后的BED条目  
            extended_bed_entries.append(f"{chrom}\t{new_start}\t{new_end}")  

            # 提取所需的序列片段（转换为0-based）  
            extracted_seq = seq[new_start:new_end]  # 转换为0-based  

            # 将提取到的序列转换为one-hot编码  
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
                    print(f"警告：发现未知碱基 {base}，在序列中将其编码为 [0,0,0,0]。")  

            # 将该行的编码添加到all_sequences列表中  
            all_sequences.append(np.array(one_hot, dtype=np.int8))  

            # 准备FASTA条目  
            fasta_id = f"{chrom}_{new_start}_{new_end}"  
            extended_fasta_entries.append(f">{fasta_id}\n{extracted_seq}\n")  

    # 将扩展后的BED条目写入文件  
    with open(output_bed, 'w') as bed_output:  
        for entry in extended_bed_entries:  
            bed_output.write(f"{entry}\n")  
    print(f"已将扩展后的BED条目保存至 {output_bed}。")  

    # 将提取到的序列写入FASTA文件  
    with open(output_fasta, 'w') as fasta_output:  
        for entry in extended_fasta_entries:  
            fasta_output.write(entry)  
    print(f"已将提取的序列保存至 {output_fasta}。")  

    # 将所有one-hot编码保存到NPZ文件，适用于条目长度不同的情况  
    if all_sequences:  
        # 创建一个字典来保存每个条目的编码数组  
        encodings_dict = {}  
        for i, seq_array in enumerate(all_sequences):  
            # 生成唯一的键，例如使用染色体、起始和结束位置  
            parts = extended_bed_entries[i].split('\t')  
            chrom, start, end = parts[0], parts[1], parts[2]  
            key = f"{chrom}_{start}_{end}"  
            encodings_dict[key] = seq_array  

        # 保存字典到NPZ文件  
        np.savez_compressed(output_npz, **encodings_dict)  
        print(f"完成！已将one-hot编码结果保存至 {output_npz}。")  
    else:  
        print("没有编码生成，可能BED文件中所有行都无效。")  
        np.savez_compressed(output_npz)  
        print(f"已创建空的NPZ文件 {output_npz}。")  

if __name__ == "__main__":  
    # 解析命令行参数  
    if len(sys.argv) != 4:  
        print("使用方法: python bed_to_onehot.py <bed_file> <fasta_file> <output_npz>")  
        sys.exit(1)  
    
    bed_file = sys.argv[1]  
    fasta_file = sys.argv[2]  
    output_npz = sys.argv[3]  

    # 生成输出文件名  
    base_name = os.path.splitext(os.path.basename(bed_file))[0]  
    output_bed = f"{base_name}_extended.bed"  
    output_fasta = f"{base_name}_extended_sequences.fasta"  

    # 调用处理函数  
    bed_to_onehot(  
        bed_file=bed_file,  
        fasta_file=fasta_file,  
        output_npz=output_npz,  
        output_bed=output_bed,  
        output_fasta=output_fasta  
    )  
