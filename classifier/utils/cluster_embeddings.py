import os
import numpy as np
from sklearn.cluster import KMeans
import hdbscan
from sklearn.manifold import TSNE
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns

def normalize_embeddings(embeddings, norm="l2"):
    if norm is None:
        return embeddings
    if norm == "l2":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms
    return embeddings

def cluster_kmeans(embeddings: np.ndarray, n_clusters: int = 10, random_state: int = 42, normalize: bool = True):
    X = normalize_embeddings(embeddings, "l2") if normalize else embeddings
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(X)
    return labels, km

def cluster_hdbscan(embeddings: np.ndarray, min_cluster_size: int = 5, normalize: bool = True):
    X = normalize_embeddings(embeddings, "l2") if normalize else embeddings
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(X)
    return labels, clusterer

def _safe_palette(n):
    # returns a palette with n colors, reusing if n > palette size
    base = sns.color_palette("hsv", min(n, 20))
    if n <= 20:
        return base
    # tile the palette
    reps = int(np.ceil(n / len(base)))
    pal = (base * reps)[:n]
    return pal

def visualize_tsne(embeddings: np.ndarray, labels: np.ndarray = None, save_path: str = None, perplexity: int = 30, random_state: int = 42):
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        emb_2d = tsne.fit_transform(embeddings)
    except Exception as e:
        print("t-SNE failed:", e)
        return None

    plt.figure(figsize=(8,6))
    if labels is not None:
        uniq = np.unique(labels)
        pal = _safe_palette(len(uniq))
        sns.scatterplot(x=emb_2d[:,0], y=emb_2d[:,1], hue=labels, palette=pal, legend="full", s=40)
    else:
        plt.scatter(emb_2d[:,0], emb_2d[:,1], s=40)
    plt.title("t-SNE projection")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return emb_2d

def visualize_umap(embeddings: np.ndarray, labels: np.ndarray = None, save_path: str = None, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42):
    try:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=None, n_jobs=-1)
        emb_2d = reducer.fit_transform(embeddings)
    except Exception as e:
        print("UMAP failed:", e)
        return None
        
    plt.figure(figsize=(8,6))
    if labels is not None:
        uniq = np.unique(labels)
        pal = _safe_palette(len(uniq))
        sns.scatterplot(x=emb_2d[:,0], y=emb_2d[:,1], hue=labels, palette=pal, legend="full", s=40)
    else:
        plt.scatter(emb_2d[:,0], emb_2d[:,1], s=40)
    plt.title("UMAP projection")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return emb_2d
