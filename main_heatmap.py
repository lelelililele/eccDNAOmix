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
from sklearn.manifold import MDS 
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

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

# Load training set
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
        assert np.all(train_labels == labels), f"{mod} train labels are inconsistent"

# Training set XGBoost feature processing
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

# Load test set
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
        assert np.all(test_labels == labels), f"{mod} test labels are inconsistent"

# Test set XGBoost feature processing
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

# Initialize model
model = DeepMultimodalModel().to(device)
#pos_weight = torch.tensor([(train_labels == 0).sum() / (train_labels == 1).sum()]).to(device)
#loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
#optimizer = optim.AdamW(model.parameters(),lr=4e-6, weight_decay=4e-1)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=3, verbose=True)

# Model training
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
    print(f"\n✅ Best model loaded: {model_path}")
else:
    print(f"\n❌ Model file does not exist: {model_path}")
    # If no model file, choose to exit or continue training
    exit()


def analyze_modality_correlation(model, data_loader, device, modalities, n_components=3):
    """
    Improved modality correlation analysis function
    Main improvements:
    1. Use PCA to retain the primary directions of variation
    2. Automatically handle dimension differences between modalities
    3. Add feature distribution comparison visualization
    """
    model.eval()
    modality_features = {mod: [] for mod in modalities}
    
    # 1. Feature extraction phase
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
    
    # 2. Feature preprocessing
    pca_features = {}
    explained_variances = {}
    
    for mod in modalities:
        if len(modality_features[mod]) == 0:
            raise ValueError(f"No features extracted for modality {mod}")
        
        # Concatenate features from all batches
        raw_features = np.concatenate(modality_features[mod], axis=0)  # [N, D]
        
        # Standardization
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(raw_features)
        
        # Dynamically set PCA dimensions (not exceeding the original dimension of the feature)
        actual_components = min(n_components, scaled_features.shape[1])
        pca = PCA(n_components=actual_components)
        pca_features[mod] = pca.fit_transform(scaled_features)  # [N, n_components]
        explained_variances[mod] = pca.explained_variance_ratio_
        
        print(f"Modality {mod}: Original dim {raw_features.shape[1]} -> reduced to {actual_components} (Cumulative explained variance: {explained_variances[mod].sum():.2f})")

    # 3. Multidimensional correlation calculation
    corr_matrix = np.zeros((len(modalities), len(modalities)))
    for i, mod1 in enumerate(modalities):
        for j, mod2 in enumerate(modalities):
            # Calculate correlation for each PCA principal component separately then average
            corr_values = []
            min_dims = min(pca_features[mod1].shape[1], pca_features[mod2].shape[1])
            
            for dim in range(min_dims):
                try:
                    corr = spearmanr(pca_features[mod1][:, dim], 
                                     pca_features[mod2][:, dim])[0]
                    corr_values.append(corr)
                except:
                    continue
            
            # Weighted average (weighted by explained variance)
            weights = (explained_variances[mod1][:min_dims] + explained_variances[mod2][:min_dims])/2
            weighted_corr = np.average(np.abs(corr_values), weights=weights)
            corr_matrix[i,j] = weighted_corr

    # 4. Create visualization panel
    plt.figure(figsize=(18, 12))
    
    # 4.1 Correlation heatmap
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
    plt.title(f"PCA-weighted inter-modality correlation (n_components={n_components})")
    
    # 4.2 Principal component explained variance
    plt.subplot(2, 2, 2)
    for mod in modalities:
        plt.plot(np.cumsum(explained_variances[mod]), 'o-', label=mod)
    plt.xlabel("PCA Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.legend()
    plt.title("PCA explained variance per modality")
    
    # 4.3 Feature distribution comparison (first principal component)
    plt.subplot(2, 2, 3)
    for mod in modalities:
        sns.kdeplot(pca_features[mod][:, 0], label=f"{mod} (PC1)")
    plt.xlim(-50, 50)
    plt.xlabel("Feature Value")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Distribution of the first principal component for each modality")
    
    # 4.4 Feature distribution comparison (box plot for all principal components)
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
    plt.title("Distribution of all principal component values")
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "modality_analysis_comprehensive.pdf"))
    plt.close()
    
    return {
        "correlation_matrix": corr_matrix,
        "pca_features": pca_features,
        "explained_variances": explained_variances
    }
# Call analysis function (assuming modalities is defined)
analysis_results = analyze_modality_correlation(
    model, 
    train_loader, 
    device, 
    modalities,  # E.g.: ['seq', 'snp', 'expression']
    n_components=3  # Adjustable based on data dimensions
)

# Save all results (recommended approach)
import pickle
with open(os.path.join(OUTPUT_DIR, "modality_analysis_results.pkl"), 'wb') as f:
    pickle.dump(analysis_results, f)

# Or save each component separately (compatible with old code)
np.save(os.path.join(OUTPUT_DIR, "modality_correlation_matrix.npy"), 
        analysis_results['correlation_matrix'])
np.save(os.path.join(OUTPUT_DIR, "pca_features.npy"), 
        analysis_results['pca_features'])
np.save(os.path.join(OUTPUT_DIR, "explained_variances.npy"), 
        analysis_results['explained_variances'])

# If only the correlation matrix is needed (compatible with old code)
modality_corr = analysis_results['correlation_matrix']
np.save(os.path.join(OUTPUT_DIR, "modality_correlation.npy"), modality_corr)
