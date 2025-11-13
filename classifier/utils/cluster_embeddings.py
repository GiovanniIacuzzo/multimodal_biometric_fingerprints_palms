import os
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
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
    """Valutazione clustering usando distanza basata sul coseno"""
    mask = labels != -1
    if np.sum(mask) < 2 or len(np.unique(labels[mask])) < 2:
        return {"silhouette": np.nan, "davies": np.nan, "calinski": np.nan}
    X_masked = X[mask]
    y_masked = labels[mask]
    
    # silhouette_score accetta metric='cosine'
    return {
        "silhouette": float(silhouette_score(X_masked, y_masked, metric='cosine')),
        "davies": float(davies_bouldin_score(X_masked, y_masked)),  # non supporta cosine direttamente
        "calinski": float(calinski_harabasz_score(X_masked, y_masked))  # idem
    }

def preprocess_embeddings(X, method='pca', dim=50, random_state=42):
    """Riduzione dimensionale con PCA o UMAP"""
    X_proc = X.copy()
    if method == 'pca' and X_proc.shape[1] > dim:
        X_proc = PCA(n_components=dim, random_state=random_state).fit_transform(X_proc)
    elif method == 'umap' and X_proc.shape[1] > dim:
        X_proc = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=dim, random_state=random_state).fit_transform(X_proc)
    return X_proc

def cluster_kmeans(X, n_clusters=8, dim_reduction='pca', dim=50, random_state=42):
    """KMeans basato su cosine: normalizza gli embeddings prima di KMeans"""
    X_proc = X.copy()
    # normalizzazione L2 per cosine similarity
    X_proc = normalize(X_proc, norm='l2')
    X_proc = preprocess_embeddings(X_proc, method=dim_reduction, dim=dim, random_state=random_state)
    
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    labels = km.fit_predict(X_proc)
    
    report = {
        "algorithm": "kmeans_cosine",
        "params": {"n_clusters": n_clusters, "dim_reduction": dim_reduction, "dim": dim},
        "metrics": evaluate_clustering(X_proc, labels),
        "cluster_sizes": {int(i): int(np.sum(labels == i)) for i in np.unique(labels)},
        "embedding_summary": summarize_embeddings(X_proc)
    }
    return labels, report

def cluster_agglomerative(X, n_clusters=8, linkage='average', metric='cosine',
                          dim_reduction='pca', dim=50, random_state=42):
    """
    Agglomerative Clustering basato su cosine distance.
    """
    X_proc = X.copy()
    X_proc = normalize(X_proc, norm='l2')  # necessario per cosine similarity
    # riduzione dimensionale
    if dim_reduction == 'pca' and X_proc.shape[1] > dim:
        X_proc = PCA(n_components=dim, random_state=random_state).fit_transform(X_proc)
    elif dim_reduction == 'umap' and X_proc.shape[1] > dim:
        X_proc = umap.UMAP(n_neighbors=15, min_dist=0.0, n_components=dim,
                           random_state=random_state).fit_transform(X_proc)

    agg = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
    labels = agg.fit_predict(X_proc)

    report = {
        "algorithm": "agglomerative",
        "params": {"n_clusters": n_clusters, "linkage": linkage, "metric": metric,
                   "dim_reduction": dim_reduction, "dim": dim},
        "metrics": evaluate_clustering(X_proc, labels),
        "cluster_sizes": {int(i): int(np.sum(labels == i)) for i in np.unique(labels)},
        "embedding_summary": summarize_embeddings(X_proc)
    }
    return labels, report

def _safe_palette(n):
    base = sns.color_palette("tab20", min(n, 20))
    reps = int(np.ceil(n / len(base)))
    return (base * reps)[:n]

def visualize_embeddings(embeddings, labels=None, method='umap', save_path=None,
                         interactive=False, random_state=42):
    reducer = TSNE(n_components=2, random_state=random_state, perplexity=30) \
        if method.lower() == 'tsne' else umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=None)
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
