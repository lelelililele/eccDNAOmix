#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
eccDNAPredict.py - eccDNA Prediction Script (Optimized Dynamic Threshold Version)
Usage: python eccDNAPredict.py -i input.txt -o output_dir
"""

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
from config import modality_keys, modalities, xgb_modalities, OUTPUT_DIR, FEATURE_SAVE_PATH
from data_utils import generate_xgb_features, transform_xgb_features, MultimodalDataset, collate_fn
from models import DeepMultimodalModel
from train_utils import train_model
from eval_utils import evaluate_modality, final_metrics
from scipy.stats import spearmanr
from sklearn.manifold import MDS  
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import umap
from sklearn.metrics import pairwise_distances


# Device configuration (Prediction default to CPU, change to "cuda" if GPU is preferred)
device = torch.device("cpu")

def print_banner():
    """Display program banner"""
    print("\n" + "="*60)
    print("eccDNA Predictor (DNA Sequence Only)".center(60))
    print("="*60)
    print(f"Using device: {device}\n")

def check_sequence_length(sequences, max_len=1200):
    """Validate input sequence lengths"""
    too_long = [i+1 for i, seq in enumerate(sequences) if len(seq) > max_len]
    if too_long:
        print(f"Warning: Found {len(too_long)} sequences exceeding {max_len}bp")
        print("The following lines will be truncated:", too_long)
    return [seq[:max_len] for seq in sequences]

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

def generate_multimodal_dataset(input_file, labels=None, max_seq_len=1199):
    """Generate multimodal dataset from DNA sequence file"""
    
    def _load_sequences(file_path):
        with open(file_path) as f:
            sequences = [line.strip().upper() for line in f if line.strip()]
        return check_sequence_length(sequences)
    
    def _dna_to_onehot(sequence):
        onehot = np.zeros((max_seq_len, 4), dtype=np.float32)
        for i, base in enumerate(sequence[:max_seq_len]):
            if base == 'A': onehot[i] = [1,0,0,0]
            elif base == 'T': onehot[i] = [0,1,0,0]
            elif base == 'C': onehot[i] = [0,0,1,0]
            elif base == 'G': onehot[i] = [0,0,0,1]
        return onehot
    
    def _generate_mask(sequence):
        mask = np.zeros(max_seq_len, dtype=np.float32)
        mask[:min(len(sequence), max_seq_len)] = 1
        return mask
    
    def _create_empty_modality(modality_type):
        if modality_type in ['methylation', 'expression', 'DNA6mA']:
            return {
                f"{modality_type}_data": np.zeros((1, 32), dtype=np.float32),
                f"{modality_type}_mask": np.zeros(1 if modality_type == 'expression' else max_seq_len, dtype=np.float32)
            }
        else:
            dim = {'seq':4, 'snp':1, 'variant':2, 'm6a':1, 
                  'methylation':32, 'expression':32, 'DNA6mA':32}[modality_type]
            return {
                f"{modality_type}_data": np.zeros((max_seq_len, dim), dtype=np.float32),
                f"{modality_type}_mask": np.zeros(max_seq_len, dtype=np.float32)
            }
    
    sequences = _load_sequences(input_file)
    
    if labels is None:
        labels = np.random.randint(0, 2, len(sequences)).tolist()
    elif len(labels) != len(sequences):
        print("[×] Error: Number of sequences and labels mismatch")
        sys.exit(1)
    
    dataset = []
    for seq, label in zip(sequences, labels):
        sample = {
            'seq_data': _dna_to_onehot(seq),
            'seq_mask': _generate_mask(seq),
            'label': int(label)
        }
        for mod in ['snp', 'variant', 'methylation', 'expression', 'm6a', 'DNA6mA']:
            sample.update(_create_empty_modality(mod))
        dataset.append(sample)
    
    print(f"[✓] Successfully generated {len(dataset)} samples")
    return dataset

def save_predictions(output_dir, predictions, sequences):
    """Save prediction results to file with sequences"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "predictions.txt")
    with open(output_file, 'w') as f:
        f.write("SequenceID\tSequence\tPrediction\tScore\n")
        for i, ((pred, prob), seq) in enumerate(zip(predictions, sequences)):
            f.write(f"{i+1}\t{seq}\t{'eccDNA' if pred else 'Non-eccDNA'}\t{prob:.4f}\n")
    print(f"[✓] Predictions saved to: {output_file}")

def main():
    # Command line argument parsing
    parser = argparse.ArgumentParser(description='eccDNA Prediction Tool')
    parser.add_argument('-i', '--input', required=True, help='Input sequence file path')
    parser.add_argument('-o', '--output', required=True, help='Output directory path')
    args = parser.parse_args()

    print_banner()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"[×] Error: Input file not found - {args.input}")
        sys.exit(1)

    # 加载最新训练得到的模型权重
    model_path = os.path.join(os.path.dirname(__abspath__ if '__file__' in locals() else os.getcwd()), "outputs", "ecc_model_epoch56_auc0.8437.pth")
    if not os.path.exists(model_path):
        # 兼容相对路径结构
        model_path = os.path.join(os.path.dirname(__file__), "outputs", "ecc_model_epoch56_auc0.8437.pth")
    model = load_trained_model(model_path)

    # Generate dataset
    print("\nProcessing input sequences...")
    dataset = generate_multimodal_dataset(
        input_file=args.input,
        labels=None,
        max_seq_len=1199
    )

    # Run prediction
    print("\nRunning predictions...")
    model.eval()
    predictions = []
    
    # 硬编码写入严格在训练集上推导的单模态决策阈值
    optimal_seq_threshold = 0.2143
    print(f"[➔] Applying sequence-specific dynamic threshold: {optimal_seq_threshold:.4f}")

    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=100, collate_fn=collate_fn):
            inputs = {
                'data': {mod: batch['data'][mod].to(device) for mod in modalities},
                'mask': {mod: batch['mask'][mod].to(device) for mod in modalities}
            }
            outputs = model(inputs).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # 使用最优阈值进行二元切分
            preds = (probs >= optimal_seq_threshold).astype(int)
            predictions.extend(zip(preds, probs))

    # Save results
    with open(args.input) as f:
        input_sequences = [line.strip() for line in f if line.strip()]

    # Save results with sequences
    save_predictions(args.output, predictions, input_sequences)
    print("\nPrediction completed successfully!")

if __name__ == "__main__":
    main()
