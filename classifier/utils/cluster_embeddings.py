import os
import numpy as np
from sklearn.cluster import KMeans
import hdbscan
from sklearn.manifold import TSNE
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def normalize_embeddings(embeddings, norm="l2"):
    if norm is None:
        return embeddings
    if norm == "l2":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms
    return embeddings


def cluster_kmeans(embeddings, n_clusters=10, random_state=42, normalize=True):
    X = normalize_embeddings(embeddings, "l2") if normalize else embeddings
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)
    return labels, km


def cluster_hdbscan(embeddings, min_cluster_size=5, normalize=True):
    X = normalize_embeddings(embeddings, "l2") if normalize else embeddings
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean", cluster_selection_epsilon=0.05)
    labels = clusterer.fit_predict(X)
    return labels, clusterer


def evaluate_clustering(embeddings, labels):
    """Restituisce metriche di qualità del clustering."""
    mask = labels != -1
    if np.sum(mask) < 2:
        return {"silhouette": np.nan, "davies": np.nan, "calinski": np.nan}
    X, y = embeddings[mask], labels[mask]
    return {
        "silhouette": silhouette_score(X, y),
        "davies": davies_bouldin_score(X, y),
        "calinski": calinski_harabasz_score(X, y),
    }


def _safe_palette(n):
    base = sns.color_palette("tab20", min(n, 20))
    reps = int(np.ceil(n / len(base)))
    return (base * reps)[:n]


def visualize_embeddings(
    embeddings, labels=None, method="umap", save_path=None,
    interactive=False, random_state=42
):
    """
    Visualizza embeddings in 2D con t-SNE o UMAP.
    """
    reducer = (
        TSNE(n_components=2, random_state=random_state, perplexity=30)
        if method == "tsne"
        else umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=random_state)
    )
    try:
        emb_2d = reducer.fit_transform(embeddings)
    except Exception as e:
        print(f"[WARN] {method.upper()} failed: {e}")
        return None

    plt.figure(figsize=(8, 6))
    if labels is not None:
        uniq = np.unique(labels)
        sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], hue=labels,
                        palette=_safe_palette(len(uniq)), legend=False, s=35, alpha=0.9)
    else:
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=35, alpha=0.8, c="gray")
    plt.title(f"{method.upper()} projection ({embeddings.shape[0]} samples)")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if interactive:
        plt.show()
    plt.close()
    return emb_2d
