import os
import numpy as np

base_dirs = {
    # "positive": "split_data/test/positive",
    # "negative": "split_data/test/negative",
    "positive": "split_data/train/positive",
    "negative": "split_data/train/negative"
            }

print("\n=== 检查 split_data/train 各模态数据 shape ===\n")

for label_type, folder in base_dirs.items():
    print(f"\n--- {label_type.upper()} ---")
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.npz'):
            path = os.path.join(folder, filename)
            try:
                data = np.load(path)
                print(f"{filename}:")
                for key in data.files:
                    print(f"  {key} => shape: {data[key].shape}")
            except Exception as e:
                print(f"{filename}: [ERROR] {e}")


# import os
# import numpy as np
# import glob
#
# def list_npz_keys(directory):
#     print(f"\n 正在检查目录: {directory}")
#     npz_files = glob.glob(os.path.join(directory, "*.npz"))
#
#     for file_path in npz_files:
#         try:
#             with np.load(file_path) as data:
#                 keys = data.files
#                 print(f"\n 文件: {os.path.basename(file_path)}")
#                 print(f"  包含键: {keys}")
#         except Exception as e:
#             print(f" 加载失败: {file_path}，错误: {e}")
#
# # 检查 train/negative 和 train/positive
# list_npz_keys("split_data/train/negative")
# list_npz_keys("split_data/train/positive")
#
# # 可选：检查 test 数据
# list_npz_keys("split_data/test/negative")
# list_npz_keys("split_data/test/positive")
