import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, precision_score, recall_score
import matplotlib.pyplot as plt
from config import ROC_FIGURE_PATH

def find_optimal_threshold(model, modality_name, train_loader, modalities, device):
    """
    新增函数：在训练集上寻找无数据泄露的最佳阈值 (Youden's J statistic)
    """
    model.eval()
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in train_loader:
            labels = batch['labels'].cpu().numpy()

            # 构建 mask：如果是单模态，仅激活指定模态；如果是 "all"，激活全部
            if modality_name != "all":
                mod_mask = {
                    mod: (batch['mask'][mod] if mod == modality_name else torch.zeros_like(batch['mask'][mod]))
                    for mod in modalities
                }
            else:
                mod_mask = batch['mask']

            inputs = {
                'data': {mod: batch['data'][mod].to(device) for mod in modalities},
                'mask': {mod: mod_mask[mod].to(device) for mod in modalities}
            }

            outputs = model(inputs).squeeze()
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels)

    # 计算 Youden's J 统计量最大化时的阈值
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    
    return best_threshold

def evaluate_modality(model, modality_name, data_loader, modalities, device, threshold=0.5):
    """评估单个模态在激活时的独立全指标性能（支持自定义阈值）"""
    model.eval()
    all_probs, all_labels = [], []

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
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # 使用传入的最佳阈值
    all_preds = (all_probs > threshold).astype(int)

    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    pre = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"--- Modality: {modality_name} (Threshold: {threshold:.4f}) ---")
    print(f"AUC: {auc:.4f}, Acc: {acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    return auc, acc, pre, rec, f1


def final_metrics(model, data_loader, modalities, device, threshold=0.5):
    """多模态最终评估（支持自定义阈值）"""
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
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # 使用传入的最佳阈值
            preds = (probs > threshold).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    pre = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"\n====== Final Validation Results (Multimodal) (Threshold: {threshold:.4f}) ======")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {pre:.4f}") 
    print(f"Recall: {rec:.4f}")    
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC : {auc:.4f}")

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

    return acc, pre, rec, f1, auc
