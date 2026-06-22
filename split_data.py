import os
import subprocess

# Base path definition
base_dir = "*/Multimodal_deeplearning"


modality_config = {
    "seq": {
        "pos_script": "eccDNA_onehot_npz_process.py",
        "neg_script": "eccDNA_onehot_npz_processOP.py",
        "folder": "Input_200_1000",
        "pos_suffix": "eccDNA_onehot.npz",
        "neg_suffix": "eccDNA_onehotOP.npz" 
    },
    "snp": {
        "pos_script": "eccDNA_SNP_AFoutput.py",
        "neg_script": "eccDNA_SNP_AFoutputOP.py",
        "folder": "Input_200_1000",
        "pos_suffix": "SNP_AFoutput.npz",
        "neg_suffix": "OP_SNP_AFoutput.npz"
    },
    "variant": {
        "pos_script": "eccDNA_extended_SV.py",
        "neg_script": "eccDNA_extended_SVOP.py",
        "folder": "Input_200_1000",
        "pos_suffix": "extended_SV.npz",
        "neg_suffix": "OP_extended_SV.npz"
    },
    "methylation": {
        "pos_script": "eccDNA_5mc5hmc_output.py",
        "neg_script": "eccDNA_5mc5hmc_outputOP.py",
        "folder": "Input_6mA_5mC",
        "pos_suffix": "5mc5hmc_output.npz",
        "neg_suffix": "OP_5mc5hmc_output.npz"
    },
    "DNA6mA": {
        "pos_script": "eccDNA_6mA_output.py",
        "neg_script": "eccDNA_6mA_outputOP.py",
        "folder": "Input_6mA_5mC",
        "pos_suffix": "6mA_output.npz",
        "neg_suffix": "OP_6mA_output.npz"
    },
    "m6a": {
        "pos_script": "eccDNA_RNA_modoutput.py",
        "neg_script": "eccDNA_RNA_modoutputOP.py",
        "folder": "Input_200_1000",
        "pos_suffix": "RNA_modoutput.npz",
        "neg_suffix": "OP_RNA_modoutput.npz"
    },
    "expression": {
        "pos_script": "eccDNA_TPM.py",
        "neg_script": "eccDNA_TPMOP.py",
        "folder": "Input_200_1000",
        "pos_suffix": "eccDNA_TPM.npz",
        "neg_suffix": "OP_eccDNA_TPM.npz"
    }
}

tissues = ['N', 'T']

def run_command(cmd):
    print(f"  Executing command: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  [Error] Script execution failed: {e}")

def get_file_list(patients, tissue_types, folder, suffix):
    file_list = []
    for p in patients:
        for t in tissue_types:
            filename = f"{t}{p}_{suffix}"
            filepath = os.path.join(base_dir, folder, filename)
            file_list.append(filepath)
    return file_list

def main():
    all_patients = [1, 2, 3, 4, 5]
    
    for test_p in all_patients:
        split_dir = os.path.join(base_dir, f"split_data{test_p}")
        
        # 1. Create directory structure
        subdirs = [
            "test/positive", "test/negative",
            "train/positive", "train/negative"
        ]
        for sub in subdirs:
            os.makedirs(os.path.join(split_dir, sub), exist_ok=True)
            
        print(f"\n========== Generating split set for individual {test_p} (split_data{test_p}) ==========")
        
        # Partition training patients and test patients
        train_patients = [p for p in all_patients if p != test_p]
        test_patients = [test_p] 
        
        for mod_name, config in modality_config.items():
            pos_script = config["pos_script"]
            neg_script = config["neg_script"]
            folder = config["folder"]
            pos_suffix = config["pos_suffix"]
            neg_suffix = config["neg_suffix"]
            
            print(f"\n---> Processing modality: {mod_name}")
            
            # ================= 2. Process Training Set (Train) =================
            # Train Positive -> Call pos_script
            train_pos_inputs = get_file_list(train_patients, tissues, folder, pos_suffix)
            train_pos_out = os.path.join(split_dir, "train", "positive", f"pos_train_{mod_name}.npz")
            cmd_train_pos = f"python {pos_script} -i {' '.join(train_pos_inputs)} -o {train_pos_out}"
            run_command(cmd_train_pos)
            
            # Train Negative -> Call neg_script
            train_neg_inputs = get_file_list(train_patients, tissues, folder, neg_suffix)
            train_neg_out = os.path.join(split_dir, "train", "negative", f"neg_train_{mod_name}.npz")
            cmd_train_neg = f"python {neg_script} -i {' '.join(train_neg_inputs)} -o {train_neg_out}"
            run_command(cmd_train_neg)
            
            # ================= 3. Process Test Set (Test) =================
            # Test Positive -> Call pos_script
            test_pos_inputs = get_file_list(test_patients, tissues, folder, pos_suffix)
            test_pos_out = os.path.join(split_dir, "test", "positive", f"pos_test_{mod_name}.npz")
            cmd_test_pos = f"python {pos_script} -i {' '.join(test_pos_inputs)} -o {test_pos_out}"
            run_command(cmd_test_pos)
            
            # Test Negative -> Call neg_script
            test_neg_inputs = get_file_list(test_patients, tissues, folder, neg_suffix)
            test_neg_out = os.path.join(split_dir, "test", "negative", f"neg_test_{mod_name}.npz")
            cmd_test_neg = f"python {neg_script} -i {' '.join(test_neg_inputs)} -o {test_neg_out}"
            run_command(cmd_test_neg)

if __name__ == "__main__":
    main()
