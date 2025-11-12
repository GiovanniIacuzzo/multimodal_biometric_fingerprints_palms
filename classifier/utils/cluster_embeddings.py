import os
import numpy as np
from sklearn.cluster import KMeans
import hdbscan
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import normalize


def summarize_embeddings(X):
    norms = np.linalg.norm(X, axis=1)
    return {
        "shape": list(X.shape),
        "mean": float(np.mean(X)),
        "std": float(np.std(X)),
        "min": float(np.min(X)),
        "max": float(np.max(X)),
        "l2_mean": float(np.mean(norms)),
        "l2_std": float(np.std(norms))
    }


def evaluate_clustering(X, labels):
    mask = labels != -1
    if np.sum(mask) < 2 or len(np.unique(labels[mask])) < 2:
        return {"silhouette": np.nan, "davies": np.nan, "calinski": np.nan}
    X_masked = X[mask]
    y_masked = labels[mask]
    return {
        "silhouette": float(silhouette_score(X_masked, y_masked)),
        "davies": float(davies_bouldin_score(X_masked, y_masked)),
        "calinski": float(calinski_harabasz_score(X_masked, y_masked))
    }


def preprocess_embeddings(X, method='pca', dim=50, random_state=42):
    """Riduzione dimensionale con PCA o UMAP"""
    X_proc = X.copy()
    if method == 'pca' and X_proc.shape[1] > dim:
        X_proc = PCA(n_components=dim, random_state=random_state).fit_transform(X_proc)
    elif method == 'umap' and X_proc.shape[1] > dim:
        X_proc = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=dim, random_state=random_state).fit_transform(X_proc)
    return X_proc


def cluster_kmeans(X, n_clusters=8, normalize_input=True, dim_reduction='pca', dim=50, random_state=42):
    X_proc = X.copy()
    if normalize_input:
        X_proc = normalize(X_proc, norm='l2')
    X_proc = preprocess_embeddings(X_proc, method=dim_reduction, dim=dim, random_state=random_state)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = km.fit_predict(X_proc)

    report = {
        "algorithm": "kmeans",
        "params": {"n_clusters": n_clusters, "dim_reduction": dim_reduction, "dim": dim},
        "metrics": evaluate_clustering(X_proc, labels),
        "cluster_sizes": {int(i): int(np.sum(labels == i)) for i in np.unique(labels)},
        "embedding_summary": summarize_embeddings(X_proc)
    }
    return labels, report


def cluster_hdbscan(X, min_cluster_size=10, min_samples=None, normalize_input=False,
                    dim_reduction='umap', dim=10, metric='euclidean'):
    X_proc = X.copy()
    if normalize_input:
        X_proc = normalize(X_proc, norm='l2')
    X_proc = preprocess_embeddings(X_proc, method=dim_reduction, dim=dim)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples or min_cluster_size,
        metric=metric,
        cluster_selection_epsilon=0.01,
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(X_proc)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cluster_sizes = {int(i): int(np.sum(labels == i)) for i in np.unique(labels) if i != -1}

    report = {
        "algorithm": "hdbscan",
        "params": {
            "min_cluster_size": min_cluster_size,
            "min_samples": clusterer.min_samples,
            "metric": metric,
            "dim_reduction": dim_reduction,
            "dim": dim
        },
        "metrics": evaluate_clustering(X_proc, labels),
        "n_clusters": int(n_clusters),
        "noise_points": int(np.sum(labels == -1)),
        "cluster_sizes": cluster_sizes,
        "embedding_summary": summarize_embeddings(X_proc),
        "extra": {
            "probabilities_mean": float(np.mean(clusterer.probabilities_)),
            "outlier_score_mean": float(np.mean(clusterer.outlier_scores_)) if hasattr(clusterer, "outlier_scores_") else None
        }
    }
    return labels, report


def auto_tune_kmeans(X, cluster_range=(2, 10), normalize_input=True, dim_reduction='pca', dim=50):
    best_score = -1
    best_labels = None
    best_report = None
    for n in range(cluster_range[0], cluster_range[1]+1):
        labels, report = cluster_kmeans(X, n_clusters=n, normalize_input=normalize_input,
                                        dim_reduction=dim_reduction, dim=dim)
        score = report['metrics']['silhouette']
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_labels = labels
            best_report = report
    return best_labels, best_report


def auto_tune_hdbscan(X, min_cluster_sizes=[5, 10, 20, 30], min_samples_list=[5, 10, 15],
                      normalize_input=False, dim_reduction='umap', dim=10, metric='euclidean'):
    best_score = -1
    best_labels = None
    best_report = None
    for min_size in min_cluster_sizes:
        for min_s in min_samples_list:
            labels, report = cluster_hdbscan(
                X, min_cluster_size=min_size, min_samples=min_s,
                normalize_input=normalize_input, dim_reduction=dim_reduction, dim=dim, metric=metric
            )
            score = report['metrics']['silhouette']
            if not np.isnan(score) and score > best_score:
                best_score = score
                best_labels = labels
                best_report = report
    return best_labels, best_report


def _safe_palette(n):
    base = sns.color_palette("tab20", min(n, 20))
    reps = int(np.ceil(n / len(base)))
    return (base * reps)[:n]


def visualize_embeddings(embeddings, labels=None, method='umap', save_path=None,
                         interactive=False, random_state=42):
    reducer = TSNE(n_components=2, random_state=random_state, perplexity=30) \
        if method.lower() == 'tsne' else umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=random_state)
    try:
        emb_2d = reducer.fit_transform(embeddings)
    except Exception as e:
        print(f"[WARN] {method.upper()} failed: {e}")
        return None

    plt.figure(figsize=(8, 6))
    if labels is not None:
        uniq = np.unique(labels)
        sns.scatterplot(
            x=emb_2d[:, 0], y=emb_2d[:, 1], hue=labels,
            palette=_safe_palette(len(uniq)), legend=False, s=35, alpha=0.9
        )
    else:
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=35, alpha=0.8, c='gray')

    plt.title(f"{method.upper()} projection ({embeddings.shape[0]} samples)")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if interactive:
        plt.show()
    plt.close()
    return emb_2d
