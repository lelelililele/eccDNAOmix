import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold  # 引入分层交叉验证
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score # 用于收集指标

from config import modality_keys, modalities, xgb_modalities, OUTPUT_DIR, FEATURE_SAVE_PATH
from data_utils import generate_xgb_features, transform_xgb_features, MultimodalDataset, collate_fn
from models import DeepMultimodalModel
from train_utils import train_model
from eval_utils import evaluate_modality, final_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# 通用数据加载函数保持不变
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
# 第一步：加载所有数据并合并为全集
# ==========================================
full_data = {}
full_masks = {}
full_labels = None

print("Loading and merging datasets...")
for mod in modalities:
    train_f, train_m, train_l = load_split_data(mod, split='train')
    test_f, test_m, test_l = load_split_data(mod, split='test')
    
    # 合并 train 和 test
    features = np.concatenate([train_f, test_f], axis=0)
    masks = np.concatenate([train_m, test_m], axis=0)
    labels = np.concatenate([train_l, test_l], axis=0)
    
    full_data[mod] = features
    full_masks[mod] = masks
    
    if full_labels is None:
        full_labels = labels
    else:
        assert np.all(full_labels == labels), f"{mod} 全集标签不一致"

# ==========================================
# 第二步：设置 10 折交叉验证
# ==========================================
n_splits = 10
# 使用 StratifiedKFold 保证每一折正负样本比例一致
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 用于存储每一折的综合评估指标
cv_metrics = {'auc': [], 'acc': [], 'pre': [], 'rec': [], 'f1': []}

print(f"\nStarting {n_splits}-Fold Cross Validation...")

# 针对 full_labels 进行索引划分
for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(full_labels)), full_labels)):
    print(f"\n{'='*20} Fold {fold + 1}/{n_splits} {'='*20}")
    
    train_loaded = {}
    val_loaded = {}
    train_labels = full_labels[train_idx]
    val_labels = full_labels[val_idx]
    
    xgb_models = {}

    # 1. 针对当前折提取数据并进行预处理
    for mod in modalities:
        # 获取当前折的 train 和 val 原始数据
        X_train = full_data[mod][train_idx]
        M_train = full_masks[mod][train_idx]
        X_val = full_data[mod][val_idx]
        M_val = full_masks[mod][val_idx]
        
        # 如果是需要 XGBoost 处理的模态
        if mod in xgb_modalities:
            orig_train_shape = X_train.shape
            orig_val_shape = X_val.shape
            
            # (1) 展平数据并标准化 (注意：scaler 必须仅用 train 数据 fit)
            scaler = StandardScaler()
            X_train_flat = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1))
            X_val_flat = scaler.transform(X_val.reshape(X_val.shape[0], -1))
            
            # (2) 训练 XGBoost 模型并提取特征 (XGBoost 也只用 train 数据训练)
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

    # 2. 构建 Dataset 和 DataLoader
    train_dataset_data = {mod: train_loaded[mod][0] for mod in modalities}
    train_dataset_mask = {mod: train_loaded[mod][1] for mod in modalities}
    train_set = MultimodalDataset(train_dataset_data, train_dataset_mask, train_labels, augment=True)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)

    val_dataset_data = {mod: val_loaded[mod][0] for mod in modalities}
    val_dataset_mask = {mod: val_loaded[mod][1] for mod in modalities}
    val_set = MultimodalDataset(val_dataset_data, val_dataset_mask, val_labels, augment=False)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 3. 初始化模型（非常重要：每一折都要重新初始化模型，避免权重继承！）
    model = DeepMultimodalModel().to(device)
    pos_weight = torch.tensor([(train_labels == 0).sum() / (train_labels == 1).sum()]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=4e-6, weight_decay=4e-1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)

    # 4. 模型训练 (为每一折保存不同的权重文件)
    history = train_model(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=60,
        patience=10,
        base_model_name=f"ecc_model_fold_{fold+1}"  # 避免覆盖
    )

# 5. 模型验证与评估
    print(f"\n====== 评估 Fold {fold + 1} ======")
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            # 1. 正确提取字典中的内容
            inputs = {
                'data': {k: v.to(device) for k, v in batch['data'].items()},
                'mask': {k: v.to(device) for k, v in batch['mask'].items()}
            }
            
            # 2. 模型只接收一个打包好的 inputs 字典
            outputs = model(inputs).squeeze()
            
            # 防止最后一个 batch size 为 1 时，squeeze() 将 1D tensor 变成了 0D 标量而导致报错
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(batch['labels'].numpy())
            
    all_preds = np.array(all_preds).squeeze()
    all_labels = np.array(all_labels)
    
    auc = roc_auc_score(all_labels, all_preds)
    preds_binary = (all_preds >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds_binary)
    pre = precision_score(all_labels, preds_binary)
    rec = recall_score(all_labels, preds_binary)
    f1 = f1_score(all_labels, preds_binary)

    # 打印并记录当前折的结果
    print(f"Fold {fold+1} Results -> AUC: {auc:.4f}, ACC: {acc:.4f}, PRE: {pre:.4f}, REC: {rec:.4f}, F1: {f1:.4f}")
    cv_metrics['auc'].append(auc)
    cv_metrics['acc'].append(acc)
    cv_metrics['pre'].append(pre)
    cv_metrics['rec'].append(rec)
    cv_metrics['f1'].append(f1)

# ==========================================
# 第三步：输出 10 折交叉验证的最终统计结果
# ==========================================
print("\n" + "="*40)
print("10-Fold Cross Validation Final Results:")
print("="*40)
print(f"AUC:       {np.mean(cv_metrics['auc']):.4f} ± {np.std(cv_metrics['auc']):.4f}")
print(f"Accuracy:  {np.mean(cv_metrics['acc']):.4f} ± {np.std(cv_metrics['acc']):.4f}")
print(f"Precision: {np.mean(cv_metrics['pre']):.4f} ± {np.std(cv_metrics['pre']):.4f}")
print(f"Recall:    {np.mean(cv_metrics['rec']):.4f} ± {np.std(cv_metrics['rec']):.4f}")
print(f"F1 Score:  {np.mean(cv_metrics['f1']):.4f}  ± {np.std(cv_metrics['f1']):.4f}")
print("="*40)
