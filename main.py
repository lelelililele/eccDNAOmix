import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from config import modality_keys, modalities, xgb_modalities, OUTPUT_DIR, FEATURE_SAVE_PATH
from data_utils import generate_xgb_features, transform_xgb_features, MultimodalDataset, collate_fn
from models import DeepMultimodalModel
from train_utils import train_model
from eval_utils import evaluate_modality, final_metrics



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

#  训练集 XGBoost 特征处理
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

#  加载测试集
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

# 初始化模型
model = DeepMultimodalModel().to(device)
pos_weight = torch.tensor([(train_labels == 0).sum() / (train_labels == 1).sum()]).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.AdamW(model.parameters(),lr=4e-6, weight_decay=4e-1)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)

#  模型训练
history = train_model(
    model=model,
    device=device,
    train_loader=train_loader,
    val_loader=test_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=60,
    patience=10,
    base_model_name="ecc_model"
)

#单模态评估
print("\n====== 单模态评估（测试集） ======")
#for mod in modalities:
#    auc = evaluate_modality(model, mod, test_loader, modalities, device)
#    print(f"{mod} AUC: {auc:.4f}")
for mod in modalities:
    # 接收 5 个值，我们只用第一个 auc 来打印，其余的会自动在函数内打印
    auc, acc, pre, rec, f1 = evaluate_modality(model, mod, test_loader, modalities, device)

#多模态评估 + ROC
final_metrics(model, test_loader, modalities, device)


