import csv
import json
import shutil
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from config.config_classifier import load_config

CONFIG_SORTED = load_config().sorted

# ======================
# UTILITY
# ======================
def safe_join_dataset(dataset_roots, csv_path_str: str) -> Path:
    """Trova un file cercandolo dentro tutti i dataset root."""
    p = Path(csv_path_str)

    # 1. Se path assoluto valido → usalo
    if p.is_absolute() and p.exists():
        return p

    # 2. Controllo diretto dentro ogni root
    for root in dataset_roots:
        cand = root / p.name
        if cand.exists():
            return cand

    # 3. Path relativo esistente nella working dir
    if p.exists():
        return p

    # 4. Cerca ricorsivamente in *tutti* i root
    for root in dataset_roots:
        matches = list(root.rglob(p.name))
        if matches:
            return matches[0]

    # 5. Fallback (per segnalare mancante)
    return dataset_roots[0] / p.name

def copy_files_to_clusters(rows, dataset_root: Path, out_dir: Path, copy_mode=True):
    clusters = defaultdict(list)
    missing = []
    for r in rows:
        fname = r["filename"]
        csv_path = r.get("path", "") or ""
        file_id = r.get("global_class", "")
        cluster = int(r.get("cluster_label", -1))
        if csv_path:
            src = safe_join_dataset(dataset_root, csv_path)
        else:
            # prova in tutte le root
            candidates = [root / fname for root in dataset_root]
            src = next((c for c in candidates if c.exists()), None)

        if not src.exists():
            missing.append(str(src))
            clusters[cluster].append((fname, None, file_id))
            continue
        cluster_dir = out_dir / f"cluster_{cluster}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        dst = cluster_dir / Path(fname).name
        if dst.exists():
            base, ext = dst.stem, dst.suffix
            i = 1
            while (cluster_dir / f"{base}_{i}{ext}").exists():
                i += 1
            dst = cluster_dir / f"{base}_{i}{ext}"
        if copy_mode:
            shutil.copy2(src, dst)
        else:
            shutil.move(src, dst)
        clusters[cluster].append((fname, str(dst), file_id))
    return clusters, missing


def compute_purity(clusters):
    total = 0
    correct = 0
    per_cluster = {}
    for cid, items in clusters.items():
        labels = [it[2] for it in items if it[2] is not None]
        cnt = Counter(labels)
        if not labels:
            per_cluster[cid] = {"size": 0, "purity": None}
            continue
        major_label, major_count = cnt.most_common(1)[0]
        per_cluster[cid] = {"size": len(labels), "major_label": major_label, "purity": major_count / len(labels)}
        total += len(labels)
        correct += major_count
    global_purity = correct / total if total > 0 else None
    return global_purity, per_cluster


def load_embeddings(path):
    import torch
    path = Path(path)
    if not path.exists():
        return None, None
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        return None, None
    emb = np.asarray(data.get("embeddings"))
    fnames = list(data.get("filenames", []))
    return emb, fnames


def compute_embedding_metrics(embeddings, filenames, clusters, out_dir):
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    fname_to_cluster = {f: cid for cid, items in clusters.items() for f, _, _ in items}
    y = [fname_to_cluster.get(f) for f in filenames if f in fname_to_cluster]
    valid_idx = [i for i, f in enumerate(filenames) if f in fname_to_cluster]
    X = embeddings[valid_idx]
    if len(set(y)) < 2:
        return {"error": "Too few clusters for metric computation."}
    report = {
        "silhouette": float(silhouette_score(X, y)),
        "davies_bouldin": float(davies_bouldin_score(X, y)),
        "calinski_harabasz": float(calinski_harabasz_score(X, y)),
        "num_clusters": len(set(y)),
        "num_samples": len(y)
    }
    with open(out_dir / "embedding_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


# ======================
# MAIN
# ======================
def main():
    csv_path = Path(CONFIG_SORTED.input.csv_path)
    ds_root_cfg = CONFIG_SORTED.input.dataset_root
    if isinstance(ds_root_cfg, list):
        dataset_root = [Path(d) for d in ds_root_cfg]
    else:
        dataset_root = [Path(ds_root_cfg)]
    out_dir = Path(CONFIG_SORTED.output.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Caricamento CSV ---
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))

    clusters, missing = copy_files_to_clusters(
        rows, dataset_root, out_dir, copy_mode=CONFIG_SORTED.output.copy_mode
    )

    global_purity, per_cluster = compute_purity(clusters)

    emb_report = {}
    if CONFIG_SORTED.metrics.compute_metrics and CONFIG_SORTED.input.embeddings_path:
        embeddings, filenames = load_embeddings(CONFIG_SORTED.input.embeddings_path)
        if embeddings is not None:
            emb_report = compute_embedding_metrics(embeddings, filenames, clusters, out_dir)
        else:
            emb_report = {"warning": "Embeddings file not found or invalid"}

    # --- Report finale ---
    report = {
        "csv": str(csv_path),
        "dataset_root": str(dataset_root),
        "output_dir": str(out_dir),
        "num_clusters": len(clusters),
        "num_files": sum(len(v) for v in clusters.values()),
        "missing_count": len(missing),
        "missing_sample": CONFIG_SORTED.metrics.max_missing_display,
        "global_purity": global_purity,
        "per_cluster": per_cluster,
        "embedding_metrics": emb_report
    }

    with open(out_dir / "sorted_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDataset ordinato salvato in: {out_dir}")
    print(f"Global purity: {global_purity:.3f}")
    if emb_report:
        print(f"Metriche embedding: {emb_report}")


if __name__ == "__main__":
    main()
