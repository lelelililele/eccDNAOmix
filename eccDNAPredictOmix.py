import os
import sys
import glob
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import xgboost as xgb
from config import modality_keys, modalities, xgb_modalities, OUTPUT_DIR, FEATURE_SAVE_PATH
from data_utils import generate_xgb_features, transform_xgb_features, MultimodalDataset, collate_fn
from models import DeepMultimodalModel
from train_utils import train_model
from eval_utils import evaluate_modality, final_metrics
#from feature_extract import FeatureExtractor

device = torch.device("cpu")

def print_banner():
    """Display program banner"""
    print("\n" + "="*60)
    print("eccDNA Predictor (Omix version)".center(60))
    print("="*60)
    print(f"Using device: {device}\n")

def load_split_data(data_root, mod):
    """Load data from directory structure"""
    try:
        data_file = glob.glob(f'{data_root}/*{mod}*.npz')[0]
        data = np.load(data_file)
        
        if mod == 'seq':
            features = data['DNASeq']
            mask = data['DNASeqmask']
            
            print(f"\nDebug - Sequence data info:")
            print(f"Data type: {features.dtype}")
            print(f"Shape: {features.shape}")
            
            base_map = {0: 'A', 1: 'T', 2: 'C', 3: 'G'}
            sequences = []
            for sample, sample_mask in zip(features, mask):
                seq = []
                for pos, (position, mask_val) in enumerate(zip(sample, sample_mask)):
                    if mask_val == 0:
                        seq.append('')
                    else:
                        base_idx = np.argmax(position)
                        seq.append(base_map.get(base_idx, 'N'))
                sequences.append(''.join(seq))
            
            print(f"First 50 bases (with mask): {sequences[0][:50]}...")
            print(f"Successfully converted {len(sequences)} sequences")
            
        elif mod == 'snp':
            features = data['DNASNP']
            mask = data['DNASNPmask']
        elif mod == 'variant':
            features = data['DNASV']
            mask = data['DNASVmask']
        elif mod == 'methylation':
            features = data['DNA5mc']
            mask = data['DNA5mcmask']
        elif mod == 'expression':
            features = data['RNATPM']
            mask = data['RNATPMmask']
        elif mod == 'm6a':                  
            features = data['RNAmod']     
            mask = data['RNAmodmask']
        elif mod == 'DNA6mA':                  
            features = data['DNA6mA']    
            mask = data['DNA6mAmask']
        else:
            raise KeyError(f"Unknown modality: {mod}")

        dummy_labels = np.zeros(features.shape[0])
        if mod == 'seq':
            return features, mask, dummy_labels, sequences
        return features, mask, dummy_labels
    except IndexError:
        print(f"[×] Error: No {mod} data file found in {data_root}")
        sys.exit(1)

def check_input_files(data_root, required_modalities):
    """检查所有必需的数据文件是否存在且包含所需字段"""
    missing_files = []
    invalid_files = []
    
    for mod in required_modalities:
        # 检查文件是否存在
        file_pattern = f'{data_root}/*{mod}*.npz'
        matched_files = glob.glob(file_pattern)
        
        if not matched_files:
            missing_files.append(mod)
            continue
            
        # 检查文件内容
        try:
            data = np.load(matched_files[0])
            if mod == 'seq':
                required_keys = ['DNASeq', 'DNASeqmask']
            elif mod == 'DNA6mA':
                required_keys = ['DNA6mA', 'DNA6mAmask']
            elif mod == 'snp':
                required_keys = ['DNASNP', 'DNASNPmask']
            elif mod == 'variant':
                required_keys = ['DNASV', 'DNASVmask']
            elif mod == 'methylation':
                required_keys = ['DNA5mc', 'DNA5mcmask']
            elif mod == 'expression':
                required_keys = ['RNATPM', 'RNATPMmask']
            else:
                required_keys = ['RNAmod', 'RNAmodmask']
            
            for key in required_keys:
                if key not in data:
                    invalid_files.append((mod, f"Missing key: {key}"))
                    break                   
        except Exception as e:
            invalid_files.append((mod, f"Invalid file: {str(e)}"))    
    return missing_files, invalid_files

def load_trained_model(model_path):
    """Load pretrained model"""
    try:
        model = DeepMultimodalModel().to(device)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("[✓] Successfully loaded model:", model_path)
            return model
        else:
            print("[×] Error: Model file not found:", model_path)
            sys.exit(1)
    except Exception as e:
        print(f"[×] Error loading model: {str(e)}")
        sys.exit(1)

def save_predictions(output_dir, predictions, sequences=None):
    """Save prediction results to file with sequences"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "predictions.txt")
    with open(output_file, 'w') as f:
        f.write("SampleID\tSequence\tPrediction\tProbability\n")
        for i, (pred, prob) in enumerate(predictions):
            seq = sequences[i] if sequences else "N/A"
            f.write(f"{i+1}\t{seq}\t{'eccDNA' if pred else 'Non-eccDNA'}\t{prob:.4f}\n")
    print(f"[✓] Predictions saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='eccDNAOmix Prediction Tool')
    parser.add_argument('-i', '--input', required=True, help='Directory containing .npz feature files')
    parser.add_argument('-o', '--output', required=True, help='Output directory path')
    args = parser.parse_args()

    print_banner()
    
    # 检查输入文件
    print("\nChecking input files...")
    missing_files, invalid_files = check_input_files(args.input, modalities)
    
    if missing_files or invalid_files:
        if missing_files:
            print("\n[×] Missing files for modalities:", missing_files)
            print("Expected files like:", [f"*{mod}*.npz" for mod in missing_files])
        if invalid_files:
            print("\n[×] Invalid files:")
            for mod, reason in invalid_files:
                print(f"- {mod}: {reason}")
        sys.exit(1)

    # 加载模型（确保从outputs目录加载）
    model_dir = os.path.join(os.path.dirname(__file__), "outputs")
    model_path = os.path.join(model_dir, "ecc_model_epoch37_auc0.8225.pth")
    model = load_trained_model(model_path)

    # 加载预训练的XGBoost模型和Scaler（从outputs目录）
    xgb_models = {}
    scalers = {}
    for mod in xgb_modalities:
        try:
            # 加载Scaler
            scaler_path = os.path.join(model_dir, f'scaler_{mod}.pkl')
            if not os.path.exists(scaler_path):
                print(f"[×] Error: Scaler file not found at {scaler_path}")
                sys.exit(1)
            scalers[mod] = joblib.load(scaler_path)
            
            # 关键修改：加载XGBoost模型时包装为XGBClassifier
            xgb_path = os.path.join(model_dir, f'xgb_{mod}.model')
            if not os.path.exists(xgb_path):
                print(f"[×] Error: XGBoost model not found at {xgb_path}")
                sys.exit(1)
            
            # 创建XGBClassifier包装器
            xgb_model = xgb.XGBClassifier()
            booster = xgb.Booster()
            booster.load_model(xgb_path)
            xgb_model._Booster = booster
            xgb_models[mod] = xgb_model
            
            print(f"[✓] Loaded {mod} models (with apply support)")
        except Exception as e:
            print(f"[×] Error loading {mod} models: {str(e)}")
            sys.exit(1)

    predict_loaded = {}
    predict_labels = None
    sequences = None

    # 加载和预处理数据
    for mod in modalities:
        try:
            print(f"\nProcessing modality: {mod}")
            if mod == 'seq':
                print("Loading sequence data...")
                features, mask, labels, seq_data = load_split_data(args.input, mod)
                sequences = seq_data
                print(f"Loaded {len(sequences)} sequences")
            else:
                features, mask, labels = load_split_data(args.input, mod)
            
            print(f"Shape - Features: {features.shape}, Mask: {mask.shape}")
            if predict_labels is None:
                predict_labels = labels
            
            # 特殊处理XGBoost模态
            if mod in xgb_modalities:
                # 确保数据是2D的
                features = features.reshape(features.shape[0], -1)
                # 使用预训练的Scaler
                features = scalers[mod].transform(features)
                # 生成XGBoost特征（使用原transform_xgb_features）
                try:
                    xgb_feat = transform_xgb_features(xgb_models[mod], features)
                    # 转换为3D (batch, 1, features)
                    features = xgb_feat[:, np.newaxis, :]
                    print(f"Processed {mod} features shape: {features.shape}")
                except Exception as e:
                    print(f"[×] Error transforming {mod} features: {str(e)}")
                    sys.exit(1)
            
            predict_loaded[mod] = (features, mask)
            
        except Exception as e:
            print(f"Error processing {mod}: {str(e)}")
            sys.exit(1)

    # 创建数据集（确保只使用成功加载的模态）
    available_modalities = list(predict_loaded.keys())
    predict_data = {mod: predict_loaded[mod][0] for mod in available_modalities}
    predict_mask = {mod: predict_loaded[mod][1] for mod in available_modalities}
    predict_set = MultimodalDataset(predict_data, predict_mask, predict_labels, augment=False)

    # 运行预测（使用较小的batch_size防止内存不足）
    print("\nRunning predictions...")
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in DataLoader(predict_set, batch_size=100, collate_fn=collate_fn):
            # 确保只使用实际存在的模态
            inputs = {
                'data': {mod: batch['data'][mod].to(device) for mod in available_modalities},
                'mask': {mod: batch['mask'][mod].to(device) for mod in available_modalities}
            }
            outputs = model(inputs).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            predictions.extend(zip(preds, probs))
            
            # 调试输出
            print(f"Batch predictions - eccDNA: {(preds == 1).sum()}, Non-eccDNA: {(preds == 0).sum()}")
            print(f"Probability range: {probs.min():.4f}-{probs.max():.4f}")

    # 保存结果
    save_predictions(args.output, predictions, sequences)
    print("\n[✓] Prediction completed!")

if __name__ == "__main__":
    main()
