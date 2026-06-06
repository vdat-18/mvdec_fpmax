"""Optional plots from the legacy preprocessing notebook."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


def plot_feature_boxplots(df: pd.DataFrame, title_suffix: str = "") -> None:
    """Plot notebook-style boxplots for all preprocessing features."""

    plt.figure(figsize=(8, 6))
    for index, column in enumerate(df.columns):
        plt.subplot(3, 4, index + 1)
        sns.boxplot(x=df[column])
        plt.title(f"{column} Boxplot{title_suffix}")
        plt.tight_layout()
    plt.show()


def plot_k_distance(distances: np.ndarray, min_samples: int = 14) -> None:
    """Plot the sorted k-distance curve used to choose DBSCAN eps."""

    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title(f"k-distance graph (k = {min_samples})")
    plt.xlabel("Points sorted by distance to k-th nearest neighbor")
    plt.ylabel(f"{min_samples}-th Nearest Neighbor Distance")
    plt.grid(False)
    plt.show()


def plot_dbscan_pca_3d(df: pd.DataFrame, labels: np.ndarray) -> None:
    """Plot DBSCAN labels in the first three PCA dimensions."""

    pca_3d = PCA(n_components=3)
    points = pca_3d.fit_transform(df)

    plt.figure(figsize=(12, 9))
    axis = plt.axes(projection="3d")
    for label in np.unique(labels):
        is_noise = label == -1
        color = [0, 0, 1, 1] if is_noise else [1, 0, 0, 1]
        label_name = "Noise" if is_noise else f"Cluster {label}"
        cluster_points = points[labels == label]
        axis.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            cluster_points[:, 2],
            c=[color],
            label=label_name,
            s=50,
            alpha=0.8,
        )

    axis.set_title("3D Visualization of DBSCAN Clustering")
    axis.set_xlabel("PCA1")
    axis.set_ylabel("PCA2")
    axis.set_zlabel("PCA3")
    axis.legend(loc="best")
    plt.show()
