import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve

from config import modality_keys, modalities, xgb_modalities, OUTPUT_DIR, FEATURE_SAVE_PATH
from data_utils import generate_xgb_features, transform_xgb_features, MultimodalDataset, collate_fn
from models import DeepMultimodalModel
# 导入新增的 find_optimal_threshold
from eval_utils import evaluate_modality, final_metrics, find_optimal_threshold

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# 通用数据加载函数
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

# 加载训练集
train_loaded = {}
train_labels = None
for mod in modalities:
    data, mask, labels = load_split_data(mod, split='train')
    if mod in xgb_modalities:
        data = data.reshape(data.shape[0], -1)
        data = StandardScaler().fit_transform(data)
    train_loaded[mod] = (data, mask)
    if train_labels is None:
        train_labels = labels
    else:
        assert np.all(train_labels == labels), f"{mod} train 标签不一致"

# 训练集 XGBoost 特征处理
xgb_models = {}
for mod in xgb_modalities:
    X = train_loaded[mod][0]
    mask = train_loaded[mod][1]
    xgb_feat, _, xgb_model = generate_xgb_features(mod, X, X, train_labels)
    xgb_feat = xgb_feat[:, np.newaxis, :]
    xgb_models[mod] = xgb_model
    train_loaded[mod] = (xgb_feat, mask)

train_data = {mod: train_loaded[mod][0] for mod in modalities}
train_mask = {mod: train_loaded[mod][1] for mod in modalities}
train_set = MultimodalDataset(train_data, train_mask, train_labels, augment=True)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)

# 加载测试集
test_loaded = {}
test_labels = None
for mod in modalities:
    data, mask, labels = load_split_data(mod, split='test')
    if mod in xgb_modalities:
        data = data.reshape(data.shape[0], -1)
        data = StandardScaler().fit_transform(data)
    test_loaded[mod] = (data, mask)
    if test_labels is None:
        test_labels = labels
    else:
        assert np.all(test_labels == labels), f"{mod} test 标签不一致"

# 测试集 XGBoost 特征处理
for mod in xgb_modalities:
    X = test_loaded[mod][0]
    mask = test_loaded[mod][1]
    xgb_feat = transform_xgb_features(xgb_models[mod], X)
    xgb_feat = xgb_feat[:, np.newaxis, :]
    test_loaded[mod] = (xgb_feat, mask)

test_data = {mod: test_loaded[mod][0] for mod in modalities}
test_mask = {mod: test_loaded[mod][1] for mod in modalities}
test_set = MultimodalDataset(test_data, test_mask, test_labels, augment=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 初始化模型、损失函数和优化器
model = DeepMultimodalModel().to(device)
pos_weight = torch.tensor([(train_labels == 0).sum() / (train_labels == 1).sum()]).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(), lr=4e-6, weight_decay=4e-1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)

# 训练控制参数
num_epochs = 60
patience = 10
best_val_auc = 0.0
patience_counter = 0
base_model_name = "ecc_model"
best_model_filename = ""

print("\n开始执行内嵌多指标训练流程...")

# ==========================================
# 内嵌自定义训练与验证循环
# ==========================================
for epoch in range(num_epochs):
    # ------------------ 训练阶段 ------------------
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
    
    # 为训练集监控计算动态最优阈值 (仅用于终端打印展示，不影响模型权重)
    fpr_t, tpr_t, thresholds_t = roc_curve(train_targets, train_probs)
    best_thresh_train = thresholds_t[np.argmax(tpr_t - fpr_t)]
    train_preds = (np.array(train_probs) > best_thresh_train).astype(int)
    
    train_acc = accuracy_score(train_targets, train_preds)
    train_pre = precision_score(train_targets, train_preds, zero_division=0)
    train_rec = recall_score(train_targets, train_preds, zero_division=0)
    train_f1 = f1_score(train_targets, train_preds, zero_division=0)

    # ------------------ 验证阶段 (测试集) ------------------
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
    
    # 验证集同样使用训练集找到的最佳阈值来打印监控
    val_preds = (np.array(val_probs) > best_thresh_train).astype(int)
    
    val_acc = accuracy_score(val_targets, val_preds)
    val_pre = precision_score(val_targets, val_preds, zero_division=0)
    val_rec = recall_score(val_targets, val_preds, zero_division=0)
    val_f1 = f1_score(val_targets, val_preds, zero_division=0)

    # ------------------ 逐行打印 6 大指标 ------------------
    print(f"Epoch {epoch+1}/{num_epochs} (Dyn. Threshold: {best_thresh_train:.4f})")
    print(f"  Train -> Loss: {train_loss:.4f} | AUC: {train_auc:.4f} | Acc: {train_acc:.4f} | Pre: {train_pre:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f}")
    print(f"  Val   -> Loss: {val_loss:.4f} | AUC: {val_auc:.4f} | Acc: {val_acc:.4f} | Pre: {val_pre:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")

    # 学习率调度器更新
    scheduler.step(val_auc)
    
    # 保存最佳权重与 Early Stopping 控制
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        
        # 清除历史生成的旧权重文件，避免硬盘冗余
        old_models = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(base_model_name) and f.endswith('.pth')]
        for f in old_models:
            os.remove(os.path.join(OUTPUT_DIR, f))
            
        best_model_filename = f"{base_model_name}_epoch{epoch+1}_auc{best_val_auc:.4f}.pth"
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, best_model_filename))
        print(f"  新最佳模型已保存: {best_model_filename}")
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"\n[Early Stopping] 验证集 AUC 连续 {patience} 个 Epoch 未提升，停止训练。")
        break
        
    print("-" * 60)

# ==========================================
# 训练完成后评估 (加载刚才保存的最优权重)
# ==========================================
if best_model_filename:
    print(f"\n加载当前训练中最优权重进行代表性评估: {best_model_filename}")
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, best_model_filename), map_location=device))

# ！！！防泄露核心操作：构建一个无数据增强的纯净训练集 DataLoader 用于计算最优阈值
train_eval_set = MultimodalDataset(train_data, train_mask, train_labels, augment=False)
train_eval_loader = DataLoader(train_eval_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 单模态评估
print("\n====== 单模态评估（测试集，使用动态阈值） ======")
for mod in modalities:
    # 1. 严格在训练集上寻找该模态的最佳阈值
    opt_thresh = find_optimal_threshold(model, mod, train_eval_loader, modalities, device)
    # 2. 将此阈值应用于测试集
    evaluate_modality(model, mod, test_loader, modalities, device, threshold=opt_thresh)

# 多模态最终评估 + 生成 ROC 曲线
print("\n====== 多模态最终评估（测试集，使用动态阈值） ======")
# 1. 严格在训练集上寻找多模态的最佳阈值
multi_opt_thresh = find_optimal_threshold(model, "all", train_eval_loader, modalities, device)
# 2. 将此阈值应用于测试集
acc, pre, rec, f1, auc = final_metrics(model, test_loader, modalities, device, threshold=multi_opt_thresh)

# 构建最终的输出结果字典格式
final_metrics_dict = {
    'auc': [auc],
    'acc': [acc],
    'pre': [pre],
    'rec': [rec],
    'f1': [f1]
}

print("\n" + "="*40)
print("Multimodal Final Metrics Dictionary:")
print(final_metrics_dict)
print("="*40 + "\n")
