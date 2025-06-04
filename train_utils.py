# import torch
# import numpy as np
# from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
# import os
# from config import OUTPUT_DIR  # 不再从 config 读取固定 MODEL_SAVE_PATH
#
# def train_model(
#     model,
#     device,
#     train_loader,
#     val_loader,
#     loss_fn,
#     optimizer,
#     scheduler=None,
#     save_dir=OUTPUT_DIR,         # 默认保存到 outputs
#     num_epochs=100,
#     patience=20,
#     base_model_name="model"
# ):
#     print(f"\nStart Training on {device}...\n")
#
#     history = {
#         'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_auc': [],
#         'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []
#     }
#
#     best_auc = 0.0
#     no_improve_epochs = 0
#     best_model_path = None
#
#     os.makedirs(save_dir, exist_ok=True)
#
#     for epoch in range(num_epochs):
#         # ===== Training =====
#         model.train()
#         train_loss = 0.0
#         train_labels, train_preds, train_probs = [], [], []
#
#         for batch in train_loader:
#             labels = batch['labels'].float().to(device)
#             inputs = {
#                 'data': {k: v.to(device) for k, v in batch['data'].items()},
#                 'mask': {k: v.to(device) for k, v in batch['mask'].items()}
#             }
#
#             optimizer.zero_grad()
#             outputs = model(inputs).squeeze()
#             loss = loss_fn(outputs, labels)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
#             optimizer.step()
#             train_loss += loss.item()
#
#             with torch.no_grad():
#                 probs = torch.sigmoid(outputs).cpu().numpy()
#                 preds = (probs > 0.5).astype(int)
#                 train_probs.extend(probs)
#                 train_preds.extend(preds)
#                 train_labels.extend(labels.cpu().numpy())
#
#         train_auc = roc_auc_score(train_labels, train_probs)
#         train_acc = accuracy_score(train_labels, train_preds)
#         train_f1 = f1_score(train_labels, train_preds)
#
#         history['train_loss'].append(train_loss / len(train_loader))
#         history['train_acc'].append(train_acc)
#         history['train_f1'].append(train_f1)
#         history['train_auc'].append(train_auc)
#
#         # ===== Validation =====
#         model.eval()
#         val_loss = 0.0
#         val_labels, val_preds, val_probs = [], [], []
#
#         with torch.no_grad():
#             for batch in val_loader:
#                 labels = batch['labels'].float().to(device)
#                 inputs = {
#                     'data': {k: v.to(device) for k, v in batch['data'].items()},
#                     'mask': {k: v.to(device) for k, v in batch['mask'].items()}
#                 }
#
#                 outputs = model(inputs).squeeze()
#                 loss = loss_fn(outputs, labels)
#                 val_loss += loss.item()
#
#                 probs = torch.sigmoid(outputs).cpu().numpy()
#                 preds = (probs > 0.5).astype(int)
#                 val_probs.extend(probs)
#                 val_preds.extend(preds)
#                 val_labels.extend(labels.cpu().numpy())
#
#         val_auc = roc_auc_score(val_labels, val_probs)
#         val_acc = accuracy_score(val_labels, val_preds)
#         val_f1 = f1_score(val_labels, val_preds)
#
#         history['val_loss'].append(val_loss / len(val_loader))
#         history['val_acc'].append(val_acc)
#         history['val_f1'].append(val_f1)
#         history['val_auc'].append(val_auc)
#
#         print(f"\nEpoch {epoch + 1}/{num_epochs}")
#         print(f"Train Loss: {history['train_loss'][-1]:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
#         print(f"Val   Loss: {history['val_loss'][-1]:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
#
#         # Save Best Model
#         if val_auc > best_auc:
#             best_auc = val_auc
#             no_improve_epochs = 0
#             best_model_path = os.path.join(save_dir, f"{base_model_name}_best.pth")
#             torch.save(model.state_dict(), best_model_path)
#             print(f" 最佳模型已保存（覆盖旧文件）: {best_model_path}")
#         else:
#             no_improve_epochs += 1
#             if no_improve_epochs >= patience:
#                 print(f"\n Early stopping at epoch {epoch + 1}. Best Val AUC = {best_auc:.4f}")
#                 break
#
#         if scheduler is not None:
#             scheduler.step(val_auc)
#
#     # ===== Load Best Model =====
#     if best_model_path:
#         model.load_state_dict(torch.load(best_model_path))
#         print(f"\n 恢复最佳模型: {best_model_path} (AUC = {best_auc:.4f})")
#
#
#     return history


import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os
from config import OUTPUT_DIR

def train_model(
    model,
    device,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    scheduler=None,
    save_dir=OUTPUT_DIR,
    num_epochs=100,
    patience=20,
    base_model_name="model"
):
    print(f"\nStart Training on {device}...\n")

    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [], 'train_auc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []
    }

    best_auc = 0.0
    no_improve_epochs = 0
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        # ===== Training =====
        model.train()
        train_loss = 0.0
        train_labels, train_preds, train_probs = [], [], []

        for batch in train_loader:
            labels = batch['labels'].float().to(device)
            inputs = {
                'data': {k: v.to(device) for k, v in batch['data'].items()},
                'mask': {k: v.to(device) for k, v in batch['mask'].items()}
            }

            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = loss_fn(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item()

            with torch.no_grad():
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                train_probs.extend(probs)
                train_preds.extend(preds)
                train_labels.extend(labels.cpu().numpy())

        train_auc = roc_auc_score(train_labels, train_probs)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds)

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['train_auc'].append(train_auc)


        model.eval()
        val_loss = 0.0
        val_labels, val_preds, val_probs = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                labels = batch['labels'].float().to(device)
                inputs = {
                    'data': {k: v.to(device) for k, v in batch['data'].items()},
                    'mask': {k: v.to(device) for k, v in batch['mask'].items()}
                }

                outputs = model(inputs).squeeze()
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                val_probs.extend(probs)
                val_preds.extend(preds)
                val_labels.extend(labels.cpu().numpy())

        val_auc = roc_auc_score(val_labels, val_probs)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)

        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {history['train_loss'][-1]:.4f}, AUC: {train_auc:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   Loss: {history['val_loss'][-1]:.4f}, AUC: {val_auc:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            no_improve_epochs = 0
            model_path = os.path.join(save_dir, f"{base_model_name}_epoch{epoch+1}_auc{val_auc:.4f}.pth")
            torch.save(model.state_dict(), model_path)
            print(f" ✅ 新最佳模型已保存: {model_path}")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"\n ⏹️ Early stopping at epoch {epoch + 1}. Best Val AUC = {best_auc:.4f}")
                break

        if scheduler is not None:
            scheduler.step(val_auc)

    return history
