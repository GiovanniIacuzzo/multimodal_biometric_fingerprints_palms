# cluster_embeddings_fast.py
import os
import numpy as np
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import normalize
import math

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

def evaluate_clustering_sample(X, labels, sample_size=5000, random_state=42):
    # valuta su un sottocampione se molto grande
    n = X.shape[0]
    if n > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(n, sample_size, replace=False)
        Xs = X[idx]
        ys = labels[idx]
    else:
        Xs = X
        ys = labels

    mask = ys != -1
    if np.sum(mask) < 2 or len(np.unique(ys[mask])) < 2:
        return {"silhouette": np.nan, "davies": np.nan, "calinski": np.nan}

    try:
        sil = float(silhouette_score(Xs[mask], ys[mask], metric='cosine'))
    except Exception:
        sil = float('nan')
    try:
        dav = float(davies_bouldin_score(Xs[mask], ys[mask]))
    except Exception:
        dav = float('nan')
    try:
        cal = float(calinski_harabasz_score(Xs[mask], ys[mask]))
    except Exception:
        cal = float('nan')

    return {"silhouette": sil, "davies": dav, "calinski": cal}

def preprocess_embeddings(X, method='pca', dim=50, random_state=42, batch_size=1024):
    """
    Riduzione dimensionale memory-friendly.
    Usa IncrementalPCA se molti campioni; PCA normale se piccolo.
    """
    X_proc = X.astype(np.float32, copy=True)
    if X_proc.shape[1] <= dim:
        return X_proc

    n_samples = X_proc.shape[0]
    if method == 'pca':
        if n_samples > 20000:
            ipca = IncrementalPCA(n_components=dim)
            for i in range(0, n_samples, batch_size):
                ipca.partial_fit(X_proc[i:i+batch_size])
            X_proc = ipca.transform(X_proc)
        else:
            X_proc = PCA(n_components=dim, random_state=random_state).fit_transform(X_proc)
    elif method == 'umap':
        # UMAP è spesso più veloce su dimensioni moderate; ritornamo la proiezione umap a dim
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=dim, random_state=random_state)
        X_proc = reducer.fit_transform(X_proc)
    return X_proc

def cluster_kmeans(X, n_clusters=256, dim_reduction='pca', dim=64, random_state=42, mbatch_size=2048):
    """
    Versione scalabile di KMeans: normalizza + riduce dimensione + MiniBatchKMeans.
    n_clusters di default impostato più alto (adatta a fingerprint con molti soggetti).
    """
    # 1) Normalize for cosine
    X_proc = normalize(X.astype(np.float32), norm='l2')

    # 2) Dim reduction
    X_proc = preprocess_embeddings(X_proc, method=dim_reduction, dim=dim, random_state=random_state)

    # 3) MiniBatchKMeans
    mbk = MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=mbatch_size, n_init=3)
    labels = mbk.fit_predict(X_proc)

    report = {
        "algorithm": "minibatch_kmeans_cosine",
        "params": {"n_clusters": int(n_clusters), "dim_reduction": dim_reduction, "dim": int(dim), "mbatch_size": int(mbatch_size)},
        "metrics": evaluate_clustering_sample(X_proc, labels),
        "cluster_sizes": {int(i): int(np.sum(labels == i)) for i in np.unique(labels)},
        "embedding_summary": summarize_embeddings(X_proc)
    }
    return labels, report

def cluster_agglomerative_fast(X, n_clusters=64, linkage='average', metric='cosine',
                               dim_reduction='pca', dim=64, random_state=42,
                               center_reduction=512, mbatch_size=2048):
    """
    Two-stage Agglomerative:
      1) Riduci i punti a 'center_reduction' centri con MiniBatchKMeans
      2) Esegui Agglomerative sui centri (molto più veloce)
      3) Mappa ogni punto al centro più vicino (assegnazione finale)
    Questo evita O(n^2) memory/time su dataset grandi.
    """
    n_samples = X.shape[0]
    X_proc = normalize(X.astype(np.float32), norm='l2')
    X_proc = preprocess_embeddings(X_proc, method=dim_reduction, dim=dim, random_state=random_state)

    # Se il dataset è già piccolo, facciamo l'agglomerative diretto
    if n_samples <= max(2000, center_reduction):
        agg = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
        labels = agg.fit_predict(X_proc)
        report = {
            "algorithm": "agglomerative_direct",
            "params": {"n_clusters": int(n_clusters), "linkage": linkage, "metric": metric, "dim_reduction": dim_reduction, "dim": int(dim)},
            "metrics": evaluate_clustering_sample(X_proc, labels),
            "cluster_sizes": {int(i): int(np.sum(labels == i)) for i in np.unique(labels)},
            "embedding_summary": summarize_embeddings(X_proc)
        }
        return labels, report

    # Altrimenti, riduciamo con MiniBatchKMeans
    n_centers = min(center_reduction, max(64, n_samples // 10))
    mbk = MiniBatchKMeans(n_clusters=n_centers, batch_size=mbatch_size, random_state=random_state, n_init=3)
    centers = mbk.fit_transform(X_proc)  # attenzione: .fit_transform ritorna distanze; usiamo .cluster_centers_ invece
    centers = mbk.cluster_centers_

    # Agglomerative sui centri (molto più piccolo)
    agg_centers = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
    centers_labels = agg_centers.fit_predict(centers)

    # Mappa ogni punto al centro più vicino (euclidean in spazio ridotto)
    # compute nearest center indices efficiently
    # usiamo broadcasting in chunk per limitare memoria
    labels = np.empty(n_samples, dtype=np.int32)
    batch = 5000
    for i in range(0, n_samples, batch):
        chunk = X_proc[i:i+batch]  # (b, dim)
        # distanza euclidea verso centri
        dists = np.linalg.norm(chunk[:, None, :] - centers[None, :, :], axis=2)  # (b, n_centers)
        nearest = np.argmin(dists, axis=1)
        labels[i:i+batch] = centers_labels[nearest]

    report = {
        "algorithm": "agglomerative_two_stage",
        "params": {"n_clusters": int(n_clusters), "center_reduction": int(n_centers), "linkage": linkage, "metric": metric, "dim_reduction": dim_reduction, "dim": int(dim)},
        "metrics": evaluate_clustering_sample(X_proc, labels),
        "cluster_sizes": {int(i): int(np.sum(labels == i)) for i in np.unique(labels)},
        "embedding_summary": summarize_embeddings(X_proc)
    }
    return labels, report

def _safe_palette(n):
    base = sns.color_palette("tab20", min(n, 20))
    reps = int(np.ceil(n / len(base)))
    return (base * reps)[:n]

def visualize_embeddings(embeddings, labels=None, method='umap', save_path=None,
                         interactive=False, random_state=42, max_points_for_projection=3000):
    """
    Visualizzazione memory friendly:
    - Per t-SNE/UMAP su dataset grandi, campiona max_points_for_projection punti.
    - Se labels forniti, usa i colori; altrimenti plot neutro.
    """
    n = embeddings.shape[0]
    sample_idx = None
    if n > max_points_for_projection:
        rng = np.random.RandomState(random_state)
        sample_idx = rng.choice(n, max_points_for_projection, replace=False)
        embeddings_plot = embeddings[sample_idx]
        labels_plot = labels[sample_idx] if labels is not None else None
    else:
        embeddings_plot = embeddings
        labels_plot = labels

    # riduci a 50 dim prima di t-SNE/UMAP per velocità
    X_red = preprocess_embeddings(embeddings_plot, method='pca', dim=50, random_state=random_state)

    if method.lower() == 'tsne':
        perplexity = min(30, max(5, embeddings_plot.shape[0] // 50))
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, max_iter=1000)
    else:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=random_state)

    try:
        emb_2d = reducer.fit_transform(X_red)
    except Exception as e:
        print(f"[WARN] {method.upper()} failed: {e}")
        return None

    plt.figure(figsize=(8, 6))
    if labels_plot is not None:
        uniq = np.unique(labels_plot)
        sns.scatterplot(x=emb_2d[:, 0], y=emb_2d[:, 1], hue=labels_plot,
                        palette=_safe_palette(len(uniq)), legend=False, s=12, alpha=0.9)
    else:
        plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=12, alpha=0.8, c='gray')

    plt.title(f"{method.upper()} projection ({embeddings_plot.shape[0]} samples)")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if interactive:
        plt.show()
    plt.close()
    return emb_2d
