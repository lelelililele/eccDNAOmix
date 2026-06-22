import os
import numpy as np

base_dirs = {
    # "positive": "split_data/test/positive",
    # "negative": "split_data/test/negative",
    "positive": "split_data/train/positive",
    "negative": "split_data/train/negative"
            }

print("\n=== Checking shape of each modality in split_data/train ===\n")

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
