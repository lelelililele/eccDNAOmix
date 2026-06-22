import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import logging
import sys

from config import modality_keys, modalities, xgb_modalities, OUTPUT_DIR
from data_utils import generate_xgb_features, transform_xgb_features, MultimodalDataset, collate_fn
from models import DeepMultimodalModel

# ==========================================
# Logging configuration: Output to both console and file
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "10fold_cv_training_log.txt")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[
                        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
                        logging.StreamHandler(sys.stdout)
                    ])

def print_log(message):
    logging.info(message)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_log(f"\nUsing device: {device}")

# General data loading function
def load_split_data(mod, split='train'):
    base_dir = f'split_data/{split}'
    neg_file = glob.glob(f'{base_dir}/negative/*{mod}*.npz')[0]
    pos_file = glob.glob(f'{base_dir}/positive/*{mod}*.npz')[0]
    keys = modality_keys[mod]

    neg_data = np.load(neg_file)
    pos_data = np.load(pos_file)

    neg_feat = neg_data[keys['neg_data_key']]
    neg_mask = neg_data[keys['neg_mask_key']]
    pos_feat = pos_data[keys['pos_data_key']]
    pos_mask = pos_data[keys['pos_mask_key']]

    features = np.concatenate([neg_feat, pos_feat], axis=0)
    masks = np.concatenate([neg_mask, pos_mask], axis=0)
    labels = np.concatenate([np.zeros(len(neg_feat)), np.ones(len(pos_feat))]).astype(np.int64)

    return features, masks, labels

# ==========================================
# Step 1: Load all data and merge into a complete set
# ==========================================
full_data = {}
full_masks = {}
full_labels = None

print_log("Loading and merging datasets...")
for mod in modalities:
    train_f, train_m, train_l = load_split_data(mod, split='train')
    test_f, test_m, test_l = load_split_data(mod, split='test')
    
    features = np.concatenate([train_f, test_f], axis=0)
    masks = np.concatenate([train_m, test_m], axis=0)
    labels = np.concatenate([train_l, test_l], axis=0)
    
    full_data[mod] = features
    full_masks[mod] = masks
    
    if full_labels is None:
        full_labels = labels
    else:
        assert np.all(full_labels == labels), f"{mod} complete set labels are inconsistent"

# ==========================================
# Step 2: Set up 10-fold cross-validation
# ==========================================
n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

cv_metrics = {'auc': [], 'acc': [], 'pre': [], 'rec': [], 'f1': [], 'thresholds': []}

print_log(f"\nStarting {n_splits}-Fold Cross Validation...")

for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(full_labels)), full_labels)):
    print_log(f"\n{'='*20} Fold {fold + 1}/{n_splits} {'='*20}")
    
    train_loaded = {}
    val_loaded = {}
    train_labels = full_labels[train_idx]
    val_labels = full_labels[val_idx]
    
    xgb_models = {}

    # Extract and preprocess current fold data
    for mod in modalities:
        X_train = full_data[mod][train_idx]
        M_train = full_masks[mod][train_idx]
        X_val = full_data[mod][val_idx]
        M_val = full_masks[mod][val_idx]
        
        if mod in xgb_modalities:
            scaler = StandardScaler()
            X_train_flat = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
            X_val_flat = scaler.transform(X_val.reshape(X_val.shape[0], -1))
            
            xgb_feat_train, _, xgb_model = generate_xgb_features(mod, X_train_flat, X_train_flat, train_labels)
            xgb_feat_train = xgb_feat_train[:, np.newaxis, :]
            
            xgb_feat_val = transform_xgb_features(xgb_model, X_val_flat)
            xgb_feat_val = xgb_feat_val[:, np.newaxis, :]
            
            xgb_models[mod] = xgb_model
            train_loaded[mod] = (xgb_feat_train, M_train)
            val_loaded[mod] = (xgb_feat_val, M_val)
        else:
            train_loaded[mod] = (X_train, M_train)
            val_loaded[mod] = (X_val, M_val)

    # Build DataLoader
    train_dataset_data = {mod: train_loaded[mod][0] for mod in modalities}
    train_dataset_mask = {mod: train_loaded[mod][1] for mod in modalities}
    train_set = MultimodalDataset(train_dataset_data, train_dataset_mask, train_labels, augment=True)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)

    val_dataset_data = {mod: val_loaded[mod][0] for mod in modalities}
    val_dataset_mask = {mod: val_loaded[mod][1] for mod in modalities}
    val_set = MultimodalDataset(val_dataset_data, val_dataset_mask, val_labels, augment=False)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Initialize model, loss function and optimizer
    model = DeepMultimodalModel().to(device)
    pos_weight = torch.tensor([(train_labels == 0).sum() / (train_labels == 1).sum()]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=4e-6, weight_decay=4e-1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=False)

    # Set training parameters
    num_epochs = 60
    best_model_path = os.path.join(OUTPUT_DIR, f"ecc_model_fold_{fold+1}_best.pth")
    epoch_metrics = []

    print_log("\nStart Training...")

    # ==========================================
    # Embedded custom training loop (monitoring 5 major metrics per round)
    # ==========================================
    for epoch in range(num_epochs):
        # --- Training phase ---
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {
                'data': {k: v.to(device) for k, v in batch['data'].items()},
                'mask': {k: v.to(device) for k, v in batch['mask'].items()}
            }
            labels = batch['labels'].to(device).float()
            
            outputs = model(inputs).squeeze()
            if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
                
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            train_preds.extend(probs)
            train_targets.extend(labels.cpu().numpy())

        train_loss /= len(train_loader.dataset)
        train_auc = roc_auc_score(train_targets, train_preds)
        
        # Calculate dynamic optimal threshold for current Epoch's training set (only used for log printing monitoring)
        fpr_t, tpr_t, thresholds_t = roc_curve(train_targets, train_preds)
        epoch_thresh = thresholds_t[np.argmax(tpr_t - fpr_t)]
        
        train_bin = (np.array(train_preds) >= epoch_thresh).astype(int)
        train_acc = accuracy_score(train_targets, train_bin)
        train_pre = precision_score(train_targets, train_bin, zero_division=0)
        train_rec = recall_score(train_targets, train_bin, zero_division=0)
        train_f1 = f1_score(train_targets, train_bin, zero_division=0)

        # --- Validation phase ---
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {
                    'data': {k: v.to(device) for k, v in batch['data'].items()},
                    'mask': {k: v.to(device) for k, v in batch['mask'].items()}
                }
                labels = batch['labels'].to(device).float()
                
                outputs = model(inputs).squeeze()
                if outputs.dim() == 0: outputs = outputs.unsqueeze(0)

                loss = loss_fn(outputs, labels)
                val_loss += loss.item() * len(labels)
                probs = torch.sigmoid(outputs).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_auc = roc_auc_score(val_targets, val_preds)
        
        # Apply the threshold calculated from the current Epoch training set to the validation set for print monitoring
        val_bin = (np.array(val_preds) >= epoch_thresh).astype(int)
        val_acc = accuracy_score(val_targets, val_bin)
        val_pre = precision_score(val_targets, val_bin, zero_division=0)
        val_rec = recall_score(val_targets, val_bin, zero_division=0)
        val_f1 = f1_score(val_targets, val_bin, zero_division=0)

        # Record data for each round to find the most stable interval
        epoch_metrics.append({
            'epoch': epoch + 1,
            'val_loss': val_loss,
            'val_auc': val_auc,
            'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}
        })

        # Print all 5 major metrics, synchronously write to txt log file
        print_log(f"Epoch {epoch+1}/{num_epochs} (Epoch Thresh: {epoch_thresh:.4f})")
        print_log(f"  Train -> Loss: {train_loss:.4f} | AUC: {train_auc:.4f} | Acc: {train_acc:.4f} | Pre: {train_pre:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}")
        print_log(f"  Val   -> Loss: {val_loss:.4f} | AUC: {val_auc:.4f} | Acc: {val_acc:.4f} | Pre: {val_pre:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")
        print_log("-" * 60)
        
        scheduler.step(val_auc)

    # Select the model with the largest AUC from the most stable Val Loss interval as the best model for this fold
    window_size = 5
    best_state_dict = None
    max_target_auc = 0.0

    if len(epoch_metrics) >= window_size:
        stable_windows = []
        for i in range(len(epoch_metrics) - window_size + 1):
            window_losses = [m['val_loss'] for m in epoch_metrics[i:i+window_size]]
            variance = np.var(window_losses)
            stable_windows.append((variance, i, i+window_size))
        
        stable_windows.sort(key=lambda x: x[0])
        top_stable_windows = stable_windows[:3]

        for _, start, end in top_stable_windows:
            for m in epoch_metrics[start:end]:
                if m['val_auc'] > max_target_auc:
                    max_target_auc = m['val_auc']
                    best_state_dict = m['state_dict']
                    best_epoch = m['epoch']
    else:
        best_metric = max(epoch_metrics, key=lambda x: x['val_auc'])
        best_state_dict = best_metric['state_dict']
        best_epoch = best_metric['epoch']

    torch.save(best_state_dict, best_model_path)
    print_log(f" Best model saved (selected from Epoch {best_epoch}): {best_model_path}")

    # ==========================================
    # Model evaluation (dynamic threshold logic to prevent data leakage)
    # ==========================================
    print_log(f"\n====== Calculating dynamic threshold without data leakage for Fold {fold + 1} ======")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    
    # 1. Construct a pure training set Loader without data augmentation for this fold
    train_eval_set = MultimodalDataset(train_dataset_data, train_dataset_mask, train_labels, augment=False)
    train_eval_loader = DataLoader(train_eval_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 2. Collect probability distribution on the pure training set and calculate the best threshold
    train_eval_probs, train_eval_labels = [], []
    with torch.no_grad():
        for batch in train_eval_loader:
            inputs = {
                'data': {k: v.to(device) for k, v in batch['data'].items()},
                'mask': {k: v.to(device) for k, v in batch['mask'].items()}
            }
            outputs = model(inputs).squeeze()
            if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            train_eval_probs.extend(probs)
            train_eval_labels.extend(batch['labels'].numpy())

    fpr, tpr, thresholds = roc_curve(train_eval_labels, train_eval_probs)
    best_idx = np.argmax(tpr - fpr)
    fold_opt_thresh = thresholds[best_idx]
    print_log(f"Best threshold derived from current fold training set: {fold_opt_thresh:.4f}")

    # 3. Strictly apply this best threshold to the validation set for final scoring
    final_preds, final_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs = {
                'data': {k: v.to(device) for k, v in batch['data'].items()},
                'mask': {k: v.to(device) for k, v in batch['mask'].items()}
            }
            outputs = model(inputs).squeeze()
            if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
                
            probs = torch.sigmoid(outputs).cpu().numpy()
            final_preds.extend(probs)
            final_labels.extend(batch['labels'].numpy())
            
    final_preds = np.array(final_preds).squeeze()
    final_labels = np.array(final_labels)
    
    auc = roc_auc_score(final_labels, final_preds)
    # Replace hardcoded 0.5, use dynamic threshold
    preds_binary = (final_preds >= fold_opt_thresh).astype(int)
    acc = accuracy_score(final_labels, preds_binary)
    pre = precision_score(final_labels, preds_binary, zero_division=0)
    rec = recall_score(final_labels, preds_binary, zero_division=0)
    f1 = f1_score(final_labels, preds_binary, zero_division=0)

    print_log(f"Fold {fold+1} final score -> AUC: {auc:.4f}, ACC: {acc:.4f}, PRE: {pre:.4f}, REC: {rec:.4f}, F1: {f1:.4f}")
    
    cv_metrics['auc'].append(auc)
    cv_metrics['acc'].append(acc)
    cv_metrics['pre'].append(pre)
    cv_metrics['rec'].append(rec)
    cv_metrics['f1'].append(f1)
    cv_metrics['thresholds'].append(fold_opt_thresh)

# ==========================================
# Step 3: Output final statistical results of 10-fold cross-validation
# ==========================================
print_log("\n" + "="*40)
print_log("10-Fold Cross Validation Final Results:")
print_log("="*40)
print_log(f"AUC:           {np.mean(cv_metrics['auc']):.4f} ± {np.std(cv_metrics['auc']):.4f}")
print_log(f"Accuracy:      {np.mean(cv_metrics['acc']):.4f} ± {np.std(cv_metrics['acc']):.4f}")
print_log(f"Precision:     {np.mean(cv_metrics['pre']):.4f} ± {np.std(cv_metrics['pre']):.4f}")
print_log(f"Recall:        {np.mean(cv_metrics['rec']):.4f} ± {np.std(cv_metrics['rec']):.4f}")
print_log(f"F1 Score:      {np.mean(cv_metrics['f1']):.4f} ± {np.std(cv_metrics['f1']):.4f}")
print_log(f"Avg Threshold: {np.mean(cv_metrics['thresholds']):.4f}")
print_log("="*40)
