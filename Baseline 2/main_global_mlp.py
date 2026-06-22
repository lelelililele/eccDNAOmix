import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from config import modality_keys, modalities, xgb_modalities, OUTPUT_DIR
from data_utils import MultimodalDataset, collate_fn
from models import GlobalMLPBaselineModel
from eval_utils import evaluate_modality, final_metrics 

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
# 1. Data Preparation (Training Set) - Pure End-to-End, without XGB/LR
# ==========================================
train_loaded = {}
train_labels = None
scaler_dict = {}

for mod in modalities:
    data, mask, labels = load_split_data(mod, split='train')
    
    if mod in xgb_modalities:
        scaler = StandardScaler()
        data = data.reshape(data.shape[0], -1)
        data = scaler.fit_transform(data)
        scaler_dict[mod] = scaler
        
    train_loaded[mod] = (data, mask)
    if train_labels is None:
        train_labels = labels

train_data = {mod: train_loaded[mod][0] for mod in modalities}
train_mask = {mod: train_loaded[mod][1] for mod in modalities}
train_set = MultimodalDataset(train_data, train_mask, train_labels, augment=True)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)

# ==========================================
# 2. Data Preparation (Test Set)
# ==========================================
test_loaded = {}
test_labels = None
for mod in modalities:
    data, mask, labels = load_split_data(mod, split='test')
    
    if mod in xgb_modalities:
        data = data.reshape(data.shape[0], -1)
        data = scaler_dict[mod].transform(data)
        
    test_loaded[mod] = (data, mask)
    if test_labels is None:
        test_labels = labels

test_data = {mod: test_loaded[mod][0] for mod in modalities}
test_mask = {mod: test_loaded[mod][1] for mod in modalities}
test_set = MultimodalDataset(test_data, test_mask, test_labels, augment=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

# ==========================================
# 3. Initialize Global MLP Model
# ==========================================
model = GlobalMLPBaselineModel().to(device)

model.train()
for dummy_batch in train_loader:
    dummy_inputs = {
        'data': {k: v.to(device) for k, v in dummy_batch['data'].items()},
        'mask': {k: v.to(device) for k, v in dummy_batch['mask'].items()}
    }
    _ = model(dummy_inputs)
    break

pos_weight = torch.tensor([(train_labels == 0).sum() / (train_labels == 1).sum()]).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=4e-6, weight_decay=4e-1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)

num_epochs = 60
patience = 10
best_val_auc = 0.0
patience_counter = 0
base_model_name = "baseline_global_mlp"
best_model_filename = ""

print("\nStarting the Global End-to-End MLP training process...")

# ==========================================
# 4. Training and validation loop
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
    train_preds = (np.array(train_probs) > 0.5).astype(int)    
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
    val_preds = (np.array(val_probs) > 0.5).astype(int)  
    val_acc = accuracy_score(val_targets, val_preds)
    val_pre = precision_score(val_targets, val_preds, zero_division=0)
    val_rec = recall_score(val_targets, val_preds, zero_division=0)
    val_f1 = f1_score(val_targets, val_preds, zero_division=0)

    # Print all metrics including Pre and Rec
    print(f"Epoch {epoch+1}/{num_epochs}")
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
        print(f"  New best Global MLP saved: {best_model_filename}")
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"\n[Early Stopping] Validation AUC has not improved. Stopping training.")
        break
        
    print("-" * 60)

# ==========================================
# 5. Post-training evaluation
# ==========================================
if best_model_filename:
    print(f"\nLoading optimal Global MLP weights for final evaluation: {best_model_filename}")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, best_model_filename), map_location=device))

print("\n====== Global MLP Final Evaluation ======")
acc, pre, rec, f1, auc = final_metrics(model, test_loader, modalities, device, threshold=0.5)

final_metrics_dict = {
    'auc': [auc], 'acc': [acc], 'pre': [pre], 'rec': [rec], 'f1': [f1]
}

print("\n" + "="*40)
print("GLOBAL E2E MLP Final Metrics:")
print(final_metrics_dict)
print("="*40 + "\n")
