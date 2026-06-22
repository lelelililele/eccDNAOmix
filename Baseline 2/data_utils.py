import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset
import torch
from sklearn.linear_model import LogisticRegression  # NEW: used for Baseline feature selection

# Data loading
def load_modality_data(neg_path, pos_path, neg_data_key, neg_mask_key, pos_data_key, pos_mask_key):
    neg_data = np.load(neg_path)[neg_data_key]
    neg_mask = np.load(neg_path)[neg_mask_key]
    pos_data = np.load(pos_path)[pos_data_key]
    pos_mask = np.load(pos_path)[pos_mask_key]

    combined_data = np.concatenate([neg_data, pos_data], axis=0)
    combined_mask = np.concatenate([neg_mask, pos_mask], axis=0)

    if combined_data.ndim == 3 and combined_data.shape[-1] == 1 and combined_data.shape[-2] == 1:
        combined_data = combined_data.squeeze(axis=-1)

    if combined_data.ndim == 3:
        orig_shape = combined_data.shape
        data_flat = combined_data.reshape(-1, orig_shape[-1])
        mask_flat = combined_mask.reshape(-1)
        valid_rows = mask_flat == 1

        if np.any(valid_rows):
            scaler = StandardScaler()
            scaler.fit(data_flat[valid_rows])
            scaled_flat = scaler.transform(data_flat)
        else:
            scaled_flat = data_flat

        scaled_data = scaled_flat.reshape(orig_shape)

    elif combined_data.ndim == 2:
        data_flat = combined_data
        mask_flat = combined_mask.reshape(-1)
        valid_rows = mask_flat == 1

        if np.any(valid_rows):
            scaler = StandardScaler()
            scaler.fit(data_flat[valid_rows])
            scaled_data = scaler.transform(data_flat)
        else:
            scaled_data = data_flat
    else:
        raise ValueError("Unsupported data dimension")

    return scaled_data, combined_mask

# XGBoost features
def generate_xgb_features(modality_name, X_train, X_val, y_train, n_features=32):
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, subsample=0.8, n_jobs=8)
    xgb.fit(X_train, y_train)
    train_leaves = xgb.apply(X_train)
    val_leaves = xgb.apply(X_val)
    return build_sparse_features(train_leaves, n_features), build_sparse_features(val_leaves, n_features), xgb

def transform_xgb_features(xgb_model, X, n_features=32):
    leaves = xgb_model.apply(X)
    return build_sparse_features(leaves, n_features)

def build_sparse_features(leaves, n_features):
    n_estimators = leaves.shape[1]
    indices = leaves % n_features
    rows = np.repeat(np.arange(len(indices)), n_estimators)
    cols = indices.flatten()
    data = np.ones(len(rows))
    features = csr_matrix((data, (rows, cols)), shape=(len(indices), n_features))
    return features.toarray()

# ==========================================
# NEW: LR feature selection function for Baseline
# ==========================================
def generate_lr_features(X_train, X_val, y_train, n_features=32):
    """
    Feature selection using Logistic Regression (LR) with L1 regularization.
    Serves as the feature extractor for the Baseline, replacing XGBoost.
    """ 
    lr = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)
    importance = np.abs(lr.coef_[0])
    top_indices = np.argsort(importance)[-n_features:]  
    train_features = X_train[:, top_indices]
    val_features = X_val[:, top_indices]
    
    return train_features, val_features, top_indices

def transform_lr_features(X, top_indices):
    """
    Directly extract the top_indices (selected from the training set) for the validation/test sets.
    """
    return X[:, top_indices]


def augment_sequence(sequence, mask_prob=0.3, dropout_prob=0.3, jitter_prob=0.15):
    seq = sequence.copy()
    if np.random.rand() < mask_prob:
        mask_idx = np.random.choice(seq.shape[1], size=int(0.05 * seq.shape[1]), replace=False)
        seq[:, mask_idx] = 0
    if np.random.rand() < dropout_prob:
        drop_idx = np.random.choice(seq.shape[1], size=int(0.05 * seq.shape[1]), replace=False)
        seq[:, drop_idx] = 0
    if np.random.rand() < jitter_prob:
        for i in range(seq.shape[1] - 1):
            if np.random.rand() < 0.01:
                seq[:, [i, i + 1]] = seq[:, [i + 1, i]]
    return seq

# Dataset class
class MultimodalDataset(Dataset):
    def __init__(self, modalities_data, modalities_mask, labels, augment=False):
        self.modalities_data = modalities_data
        self.modalities_mask = modalities_mask
        self.labels = labels
        self.augment = augment

    def __getitem__(self, index):
        sample = {}
        for mod in self.modalities_data.keys():
            data = self.modalities_data[mod][index]
            mask = self.modalities_mask[mod][index]

            # Apply data augmentation exclusively to seq/snp training data
            if self.augment and mod in ['seq', 'snp']:
                data = augment_sequence(data)

            sample[f"{mod}_data"] = data
            sample[f"{mod}_mask"] = mask
        sample['label'] = self.labels[index]
        return sample

    def __len__(self):
        return len(self.labels)

# collate_fn
def collate_fn(batch):
    collated = {'data': {}, 'mask': {}, 'labels': []}

    for sample in batch:
        collated['labels'].append(sample['label'])
        for key in sample:
            if key == 'label':
                continue
            mod_name = key.split('_')[0]
            data_type = key.split('_')[1]
            if mod_name not in collated[data_type]:
                collated[data_type][mod_name] = []
            collated[data_type][mod_name].append(sample[key])

    for mod in collated['data'].keys():
        data = np.stack(collated['data'][mod])
        mask = np.stack(collated['mask'][mod])

        if mod in ['DNA6mA', 'methylation', 'expression', 'm6a']:
            data = data.reshape(data.shape[0], 1, -1)
            mask = np.ones_like(data)
            data = data.transpose(0, 2, 1)
            mask = mask.transpose(0, 2, 1)
        else:
            if mod == 'seq':
                data = data.transpose(0, 2, 1)
                mask = mask[:, np.newaxis, :]
            elif mod == 'expression':
                data = data[:, np.newaxis, :]
                mask = mask[:, np.newaxis, :]
            else:
                data = data.transpose(0, 2, 1)
                mask = mask[:, np.newaxis, :]

        collated['data'][mod] = torch.from_numpy(data).float()
        collated['mask'][mod] = torch.from_numpy(mask).float()

    collated['labels'] = torch.tensor(collated['labels']).long()
    return collated
