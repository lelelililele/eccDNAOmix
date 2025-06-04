import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from config import ROC_FIGURE_PATH


def evaluate_modality(model, modality_name, data_loader, modalities, device):
    """评估单个模态在激活时的独立AUC性能"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in data_loader:
            labels = batch['labels'].cpu().numpy()

            # 构建 mask：仅激活指定模态，其余为零
            mod_mask = {
                mod: (batch['mask'][mod] if mod == modality_name else torch.zeros_like(batch['mask'][mod]))
                for mod in modalities
            }

            inputs = {
                'data': {mod: batch['data'][mod].to(device) for mod in modalities},
                'mask': {mod: mod_mask[mod].to(device) for mod in modalities}
            }

            outputs = model(inputs).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_preds.extend(probs)
            all_labels.extend(labels)

    auc = roc_auc_score(all_labels, all_preds)
    return auc


def final_metrics(model, data_loader, modalities, device):

    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            labels = batch['labels'].cpu().numpy()
            inputs = {
                'data': {k: v.to(device) for k, v in batch['data'].items()},
                'mask': {k: v.to(device) for k, v in batch['mask'].items()}
            }

            outputs = model(inputs).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print("\n====== Final Validation Results ======")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC : {auc:.4f}")

    # 绘制 ROC 曲线
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(ROC_FIGURE_PATH)
    plt.close()

    print(f"ROC curve saved to: {ROC_FIGURE_PATH}")

    return acc, f1, auc
