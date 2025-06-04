import numpy as np
import matplotlib.pyplot as plt
import umap
from matplotlib.lines import Line2D
from config import UMAP_FIGURE_PATH


def visualize_umap(train_features, val_features, train_labels, val_labels, save_path=UMAP_FIGURE_PATH):
    # 合并特征与标签
    combined_features = np.vstack([
        train_features.reshape(train_features.shape[0], -1),
        val_features.reshape(val_features.shape[0], -1)
    ])
    combined_labels = np.concatenate([train_labels, val_labels])

    # 执行 UMAP 降维
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    umap_embedding = reducer.fit_transform(combined_features)

    # 可视化
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        umap_embedding[:, 0],
        umap_embedding[:, 1],
        c=combined_labels,
        cmap=plt.cm.tab10,
        alpha=0.6,
        edgecolors='w'
    )
    plt.xlabel('UMAP-1')
    plt.ylabel('UMAP-2')
    plt.title('Feature Space Visualization by UMAP')

    # 自定义图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Negative (0)', markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Positive (1)', markerfacecolor='red', markersize=10)
    ]
    plt.legend(handles=legend_elements, title='Classes', loc='upper right')

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"UMAP 可视化图已保存至: {save_path}")
