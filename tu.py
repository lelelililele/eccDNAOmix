import re
import pandas as pd
import matplotlib.pyplot as plt

# 读取日志文件
with open("./outputs/out.csv", "r",encoding="utf-8") as f:
    lines = f.readlines()

# 初始化存储指标的列表
epochs = []
train_loss, train_auc, train_acc, train_f1 = [], [], [], []
val_loss, val_auc, val_acc, val_f1 = [], [], [], []

# 正则表达式模式
epoch_pattern = re.compile(r"Epoch (\d+)/\d+")
train_pattern = re.compile(r"Train Loss: ([\d.]+), AUC: ([\d.]+), Acc: ([\d.]+), F1: ([\d.]+)")
val_pattern = re.compile(r"Val\s+Loss: ([\d.]+), AUC: ([\d.]+), Acc: ([\d.]+), F1: ([\d.]+)")

# 提取数据
for line in lines:
    epoch_match = epoch_pattern.search(line)
    train_match = train_pattern.search(line)
    val_match = val_pattern.search(line)

    if epoch_match:
        epochs.append(int(epoch_match.group(1)))
    if train_match:
        train_loss.append(float(train_match.group(1)))
        train_auc.append(float(train_match.group(2)))
        train_acc.append(float(train_match.group(3)))
        train_f1.append(float(train_match.group(4)))
    if val_match:
        val_loss.append(float(val_match.group(1)))
        val_auc.append(float(val_match.group(2)))
        val_acc.append(float(val_match.group(3)))
        val_f1.append(float(val_match.group(4)))

# 构建 DataFrame
df = pd.DataFrame({
    "Epoch": epochs,
    "Train Loss": train_loss,
    "Val Loss": val_loss,
    "Train AUC": train_auc,
    "Val AUC": val_auc,
    "Train Acc": train_acc,
    "Val Acc": val_acc,
    "Train F1": train_f1,
    "Val F1": val_f1,
})

# 画图函数
def plot_metric(metric_name, train_col, val_col):
    plt.figure(figsize=(10, 5))
    plt.plot(df["Epoch"], df[train_col], label=train_col)
    plt.plot(df["Epoch"], df[val_col], label=val_col)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 绘制四个图
plot_metric("Loss", "Train Loss", "Val Loss")
plot_metric("AUC", "Train AUC", "Val AUC")
plot_metric("Accuracy", "Train Acc", "Val Acc")
plot_metric("F1 Score", "Train F1", "Val F1")
