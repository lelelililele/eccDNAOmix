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
from feature_extract import FeatureExtractor
from umap_vis import visualize_umap
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.manifold import MDS  # 新增导入
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from sklearn.decomposition import PCA

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
#pos_weight = torch.tensor([(train_labels == 0).sum() / (train_labels == 1).sum()]).to(device)
#loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#optimizer = optim.AdamW(model.parameters(),lr=4e-6, weight_decay=4e-1)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)

#  模型训练
#history = train_model(
#    model=model,
#    device=device,
#    train_loader=train_loader,
#    val_loader=test_loader,
#    loss_fn=loss_fn,
#    optimizer=optimizer,
#    scheduler=scheduler,
#    num_epochs=60,
#    patience=10,
#    base_model_name="ecc_model"
#)

model_path = os.path.join(OUTPUT_DIR, "ecc_model_epoch37_auc0.8225.pth")
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print(f"\n✅ 已加载最佳模型: {model_path}")
else:
    print(f"\n❌ 模型文件不存在: {model_path}")
    # 如果没有模型文件，可以选择退出或继续训练
    exit()


def analyze_modality_correlation(model, data_loader, device, modalities, n_components=3):
    """
    改进版模态相关性分析函数
    主要改进：
    1. 使用PCA降维保留主要变异方向
    2. 自动处理不同模态的维度差异
    3. 添加特征分布对比可视化
    """
    model.eval()
    modality_features = {mod: [] for mod in modalities}
    
    # 1. 特征提取阶段
    with torch.no_grad():
        for batch in data_loader:
            inputs = {k: v.to(device) for k, v in batch['data'].items()}
            masks = {k: v.to(device) for k, v in batch['mask'].items()}
            
            features = {
                'seq': model.seq_net(inputs['seq'] * masks['seq']).squeeze(-1).cpu().numpy(),
                'snp': model.snp_net(inputs['snp'] * masks['snp']).cpu().numpy(),
                'variant': model.variant_net(inputs['variant'] * masks['variant']).cpu().numpy(),
                'm6a': model.m6a_net(inputs['m6a'] * masks['m6a']).cpu().numpy(),
                'methylation': model.methylation_net(inputs['methylation'] * masks['methylation']).cpu().numpy(),
                'expression': model.expression_net(inputs['expression']).cpu().numpy(),
                'DNA6mA': model.DNA6mA_net(inputs['DNA6mA'] * masks['DNA6mA']).cpu().numpy()
            }
            
            for mod in modalities:
                modality_features[mod].append(features[mod])
    
    # 2. 特征预处理
    pca_features = {}
    explained_variances = {}
    
    for mod in modalities:
        if len(modality_features[mod]) == 0:
            raise ValueError(f"模态 {mod} 未提取到任何特征")
        
        # 拼接所有batch的特征
        raw_features = np.concatenate(modality_features[mod], axis=0)  # [N, D]
        
        # 标准化
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(raw_features)
        
        # 动态设置PCA维度（不超过特征原始维度）
        actual_components = min(n_components, scaled_features.shape[1])
        pca = PCA(n_components=actual_components)
        pca_features[mod] = pca.fit_transform(scaled_features)  # [N, n_components]
        explained_variances[mod] = pca.explained_variance_ratio_
        
        print(f"模态 {mod}: 原始维度 {raw_features.shape[1]} -> 降维至 {actual_components} (累计解释方差: {explained_variances[mod].sum():.2f})")

    # 3. 多维度相关性计算
    corr_matrix = np.zeros((len(modalities), len(modalities)))
    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities):
            # 对每个PCA主成分分别计算相关性后取平均
            corr_values = []
            min_dims = min(pca_features[mod1].shape[1], pca_features[mod2].shape[1])
            
            for dim in range(min_dims):
                try:
                    corr = spearmanr(pca_features[mod1][:, dim], 
                                    pca_features[mod2][:, dim])[0]
                    corr_values.append(corr)
                except:
                    continue
            
            # 使用加权平均（按解释方差加权）
            weights = (explained_variances[mod1][:min_dims] + explained_variances[mod2][:min_dims])/2
            weighted_corr = np.average(np.abs(corr_values), weights=weights)
            corr_matrix[i,j] = weighted_corr

    # 4. 创建可视化面板
    plt.figure(figsize=(18, 12))
    
    # 4.1 相关性热力图
    plt.subplot(2, 2, 1)
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        fmt=".3f",
        xticklabels=modalities,
        yticklabels=modalities,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1
    )
    plt.title(f"模态间PCA加权相关性 (n_components={n_components})")
    
    # 4.2 主成分解释方差
    plt.subplot(2, 2, 2)
    for mod in modalities:
        plt.plot(np.cumsum(explained_variances[mod]), 'o-', label=mod)
    plt.xlabel("PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.legend()
    plt.title("各模态PCA解释方差")
    
    # 4.3 特征分布对比（第一主成分）
    plt.subplot(2, 2, 3)
    for mod in modalities:
        sns.kdeplot(pca_features[mod][:, 0], label=f"{mod} (PC1)")
    plt.xlim(-50, 50)
    plt.xlabel("Feature Value")
    plt.ylabel("Density")
    plt.legend()
    plt.title("各模态第一主成分分布")
    
    # 4.4 特征分布对比（所有主成分箱线图）
    plt.subplot(2, 2, 4)
    all_pc_data = []
    for mod in modalities:
        for dim in range(pca_features[mod].shape[1]):
            all_pc_data.append({
                "Modality": mod,
                "PC": f"PC{dim+1}",
                "Value": pca_features[mod][:, dim].mean()
            })
    import pandas as pd
    df_pc = pd.DataFrame(all_pc_data)
    sns.boxplot(x="Modality", y="Value", hue="PC", data=df_pc)
    plt.title("各主成分值分布")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "modality_analysis_comprehensive.pdf"))
    plt.close()
    
    return {
        "correlation_matrix": corr_matrix,
        "pca_features": pca_features,
        "explained_variances": explained_variances
    }
# 调用分析函数（假设modalities已定义）
analysis_results = analyze_modality_correlation(
    model, 
    train_loader, 
    device, 
    modalities,  # 例如: ['seq', 'snp', 'expression']
    n_components=3  # 可根据数据维度调整
)

# 保存所有结果（推荐方式）
import pickle
with open(os.path.join(OUTPUT_DIR, "modality_analysis_results.pkl"), 'wb') as f:
    pickle.dump(analysis_results, f)

# 或者分别保存各组件（兼容旧代码）
np.save(os.path.join(OUTPUT_DIR, "modality_correlation_matrix.npy"), 
        analysis_results['correlation_matrix'])
np.save(os.path.join(OUTPUT_DIR, "pca_features.npy"), 
        analysis_results['pca_features'])
np.save(os.path.join(OUTPUT_DIR, "explained_variances.npy"), 
        analysis_results['explained_variances'])

# 如果只需要相关性矩阵（保持与旧代码兼容）
modality_corr = analysis_results['correlation_matrix']
np.save(os.path.join(OUTPUT_DIR, "modality_correlation.npy"), modality_corr)

