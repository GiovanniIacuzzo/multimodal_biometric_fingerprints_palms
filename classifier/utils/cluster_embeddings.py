import numpy as np
from sklearn.cluster import KMeans
import hdbscan
from sklearn.manifold import TSNE
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# Funzione clustering KMeans
# ============================================================
def cluster_kmeans(embeddings: np.ndarray, n_clusters: int = 10, random_state: int = 42):
    """
    Clustering KMeans
    :param embeddings: array NxD
    :param n_clusters: numero di cluster
    :return: labels predetti
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans

# ============================================================
# Funzione clustering HDBSCAN
# ============================================================
def cluster_hdbscan(embeddings: np.ndarray, min_cluster_size: int = 5):
    """
    Clustering HDBSCAN
    :param embeddings: array NxD
    :param min_cluster_size: minimo numero di punti per cluster
    :return: labels predetti
    """
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(embeddings)
    return labels, clusterer

# ============================================================
# Funzione TSNE
# ============================================================
def visualize_tsne(embeddings: np.ndarray, labels: np.ndarray = None, save_path: str = None, perplexity: int = 30):
    """
    Visualizza embedding con TSNE 2D
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        palette = sns.color_palette("hsv", len(np.unique(labels)))
        sns.scatterplot(x=emb_2d[:,0], y=emb_2d[:,1], hue=labels, palette=palette, legend='full', s=50)
    else:
        plt.scatter(emb_2d[:,0], emb_2d[:,1], s=50)
    plt.title("TSNE Visualization")
    plt.xlabel("TSNE-1")
    plt.ylabel("TSNE-2")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
    return emb_2d

# ============================================================
# Funzione UMAP
# ============================================================
def visualize_umap(embeddings: np.ndarray, labels: np.ndarray = None, save_path: str = None, n_neighbors: int = 15, min_dist: float = 0.1):
    """
    Visualizza embedding con UMAP 2D
    """
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    emb_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(8, 6))
    if labels is not None:
        palette = sns.color_palette("hsv", len(np.unique(labels)))
        sns.scatterplot(x=emb_2d[:,0], y=emb_2d[:,1], hue=labels, palette=palette, legend='full', s=50)
    else:
        plt.scatter(emb_2d[:,0], emb_2d[:,1], s=50)
    plt.title("UMAP Visualization")
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.show()
    return emb_2d
