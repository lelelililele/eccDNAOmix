import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve

from config import modality_keys, modalities, lr_modalities, OUTPUT_DIR
from data_utils import generate_lr_features, transform_lr_features, MultimodalDataset, collate_fn
from models import BaselineLRMLPModel
from eval_utils import evaluate_modality, final_metrics, find_optimal_threshold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

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
# 1. Data Preparation (Training Set) - Replaced with LR
# ==========================================
train_loaded = {}
train_labels = None
for mod in modalities:
    data, mask, labels = load_split_data(mod, split='train')
    if mod in lr_modalities:
        data = data.reshape(data.shape[0], -1)
        data = StandardScaler().fit_transform(data)
    train_loaded[mod] = (data, mask)
    if train_labels is None:
        train_labels = labels
    else:
        assert np.all(train_labels == labels), f"{mod} train labels are inconsistent"

# Training set LR feature processing
lr_indices_dict = {}
for mod in lr_modalities:
    X = train_loaded[mod][0]
    mask = train_loaded[mod][1]
    lr_feat, _, top_idx = generate_lr_features(X, X, train_labels, n_features=32)
    if lr_feat.shape[1] < 32:
        lr_feat = np.pad(lr_feat, ((0, 0), (0, 32 - lr_feat.shape[1])), mode='constant')     
    lr_indices_dict[mod] = top_idx
    lr_feat = lr_feat[:, np.newaxis, :]
    train_loaded[mod] = (lr_feat, mask)

train_data = {mod: train_loaded[mod][0] for mod in modalities}
train_mask = {mod: train_loaded[mod][1] for mod in modalities}
train_set = MultimodalDataset(train_data, train_mask, train_labels, augment=True)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)

# ==========================================
# 2. Data Preparation (Test Set) - Replaced with LR
# ==========================================
test_loaded = {}
test_labels = None
for mod in modalities:
    data, mask, labels = load_split_data(mod, split='test')
    if mod in lr_modalities:
        data = data.reshape(data.shape[0], -1)
        data = StandardScaler().fit_transform(data)
    test_loaded[mod] = (data, mask)
    if test_labels is None:
        test_labels = labels

# Test set LR feature processing
for mod in lr_modalities:
    X = test_loaded[mod][0]
    mask = test_loaded[mod][1]
    lr_feat = transform_lr_features(X, lr_indices_dict[mod])
        if lr_feat.shape[1] < 32:
        lr_feat = np.pad(lr_feat, ((0, 0), (0, 32 - lr_feat.shape[1])), mode='constant')  
    lr_feat = lr_feat[:, np.newaxis, :]
    test_loaded[mod] = (lr_feat, mask)

test_data = {mod: test_loaded[mod][0] for mod in modalities}
test_mask = {mod: test_loaded[mod][1] for mod in modalities}
test_set = MultimodalDataset(test_data, test_mask, test_labels, augment=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

# ==========================================
# Define model wrapper
# ==========================================
class BaselineModelWrapper(BaselineLRMLPModel):
    def forward(self, x):
        for k in lr_modalities:
            if x['data'][k].dim() > 2:
                x['data'][k] = x['data'][k].view(x['data'][k].size(0), -1)
        return super().forward(x)

model = BaselineModelWrapper().to(device)
pos_weight = torch.tensor([(train_labels == 0).sum() / (train_labels == 1).sum()]).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=4e-6, weight_decay=4e-1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)

num_epochs = 60
patience = 10
best_val_auc = 0.0
patience_counter = 0
base_model_name = "baseline_lrmlp_model"
best_model_filename = ""

print("\nStarting the Baseline (LR+MLP) training process...")

# ==========================================
# 3. Training and validation loop
# ==========================================
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_probs, train_targets = [], []
    
    for batch in train_loader:
        optimizer.zero_grad()
        inputs = {
            'data': {k: v.to(device) for k, v in batch['data'].items()},
            'mask': {k: v.to(device) for k, v in batch['mask'].items()}
        }
        labels = batch['labels'].to(device).float()
        
        outputs = model(inputs).squeeze()
        if outputs.dim() == 0: 
            outputs = outputs.unsqueeze(0)
            
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * len(labels)
        
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        train_probs.extend(probs)
        train_targets.extend(labels.cpu().numpy())
        
    train_loss /= len(train_loader.dataset)
    train_auc = roc_auc_score(train_targets, train_probs)
    
    fpr_t, tpr_t, thresholds_t = roc_curve(train_targets, train_probs)
    best_thresh_train = thresholds_t[np.argmax(tpr_t - fpr_t)]
    train_preds = (np.array(train_probs) > best_thresh_train).astype(int)
    
    train_acc = accuracy_score(train_targets, train_preds)
    train_pre = precision_score(train_targets, train_preds, zero_division=0)
    train_rec = recall_score(train_targets, train_preds, zero_division=0)
    train_f1 = f1_score(train_targets, train_preds, zero_division=0)

    model.eval()
    val_loss = 0.0
    val_probs, val_targets = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                'data': {k: v.to(device) for k, v in batch['data'].items()},
                'mask': {k: v.to(device) for k, v in batch['mask'].items()}
            }
            labels = batch['labels'].to(device).float()
            
            outputs = model(inputs).squeeze()
            if outputs.dim() == 0: 
                outputs = outputs.unsqueeze(0)
                
            loss = loss_fn(outputs, labels)
            val_loss += loss.item() * len(labels)
            
            probs = torch.sigmoid(outputs).cpu().numpy()
            val_probs.extend(probs)
            val_targets.extend(labels.cpu().numpy())
            
    val_loss /= len(test_loader.dataset)
    val_auc = roc_auc_score(val_targets, val_probs)
    
    val_preds = (np.array(val_probs) > best_thresh_train).astype(int)
    
    val_acc = accuracy_score(val_targets, val_preds)
    val_pre = precision_score(val_targets, val_preds, zero_division=0)
    val_rec = recall_score(val_targets, val_preds, zero_division=0)
    val_f1 = f1_score(val_targets, val_preds, zero_division=0)

    print(f"Epoch {epoch+1}/{num_epochs} (Dyn. Threshold: {best_thresh_train:.4f})")
    print(f"  Train -> Loss: {train_loss:.4f} | AUC: {train_auc:.4f} | Acc: {train_acc:.4f} | Pre: {train_pre:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}")
    print(f"  Val   -> Loss: {val_loss:.4f} | AUC: {val_auc:.4f} | Acc: {val_acc:.4f} | Pre: {val_pre:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")

    scheduler.step(val_auc)
    
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        
        old_models = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(base_model_name) and f.endswith('.pth')]
        for f in old_models:
            os.remove(os.path.join(OUTPUT_DIR, f))
            
        best_model_filename = f"{base_model_name}_epoch{epoch+1}_auc{best_val_auc:.4f}.pth"
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, best_model_filename))
        print(f"  New best Baseline model saved: {best_model_filename}")
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"\n[Early Stopping] Validation AUC has not improved for {patience} consecutive epochs. Stopping training.")
        break
        
    print("-" * 60)

# ==========================================
# 4. Post-training evaluation
# ==========================================
if best_model_filename:
    print(f"\nLoading optimal Baseline weights for final evaluation: {best_model_filename}")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, best_model_filename), map_location=device))

train_eval_set = MultimodalDataset(train_data, train_mask, train_labels, augment=False)
train_eval_loader = DataLoader(train_eval_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

print("\n====== Baseline Multimodal Final Evaluation (Test Set, using dynamic threshold) ======")
multi_opt_thresh = find_optimal_threshold(model, "all", train_eval_loader, modalities, device)
acc, pre, rec, f1, auc = final_metrics(model, test_loader, modalities, device, threshold=multi_opt_thresh)

final_metrics_dict = {
    'auc': [auc], 'acc': [acc], 'pre': [pre], 'rec': [rec], 'f1': [f1]
}

print("\n" + "="*40)
print("BASELINE (LR + MLP) Final Metrics:")
print(final_metrics_dict)
print("="*40 + "\n")
