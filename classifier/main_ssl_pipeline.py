import os
import re
import csv
import json
import torch
import numpy as np
import logging
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader
from colorama import Fore, Style

from classifier.dataset2.dataset import FingerprintDataset
from classifier.models.ssl_model import SSLModel
from classifier.utils.train_ssl import train_ssl
from classifier.utils.extract_embeddings import extract_embeddings
from classifier.utils.cluster_embeddings import (
    cluster_kmeans, visualize_embeddings, cluster_agglomerative
)

# Carica la configurazione (root namespace)
from config.config_classifier import load_config

# --- LOAD CONFIG ---
cfg = load_config()        # cfg: root namespace, contiene .paths e .ssl
paths = cfg.paths          # percorsi principali
ssl_cfg = cfg.ssl          # sezione ssl con model, training, logging, ecc.

# ==========================================================
# LOGGING SETUP
# ==========================================================
# Assicuriamoci che le directory esistano prima di usarle
os.makedirs(paths.save_dir, exist_ok=True)
os.makedirs(paths.figures_dir, exist_ok=True)

# Nota: nel tuo YAML log_file è dentro ssl.logging
log_file = getattr(ssl_cfg, "logging", None)
if log_file is not None:
    log_file = getattr(ssl_cfg.logging, "log_file", os.path.join(paths.save_dir, "train.log"))
else:
    log_file = os.path.join(paths.save_dir, "train.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def extract_id(fname: str) -> str:
    """
    Estrae l'ID dal nome del file:
    - prende la prima parte prima di '_' 
    - rimuove eventuali zeri iniziali
    - se non trova nulla, ritorna '0'
    """
    stem = Path(fname).stem
    part = stem.split('_')[0]
    return part.lstrip("0") or "0"

def console_step(title):
    """Stampa un titolo colorato per una sezione della pipeline."""
    print(f"\n{Fore.CYAN}{'=' * 60}")
    print(f"{Fore.YELLOW}{title.upper()}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 60}{Style.RESET_ALL}")


def safe_json(obj):
    """Rimuove o converte oggetti non serializzabili (es. sklearn)."""
    if isinstance(obj, dict):
        return {k: safe_json(v) for k, v in obj.items()
                if not hasattr(v, "__class__") or "sklearn" not in str(v.__class__)}
    elif isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    elif isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)


# ==========================================================
# MAIN PIPELINE
# ==========================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    os.makedirs(paths.figures_dir, exist_ok=True)

    console_step("Inizializzazione")
    print(f"{Fore.GREEN}Device in uso:{Style.RESET_ALL} {device}")
    logging.info(f"Using device: {device}")

    # ------------------------------------------------------
    # 1. Dataset SSL
    # ------------------------------------------------------
    console_step("Caricamento Dataset")
    dataset_ssl = FingerprintDataset(paths.dataset_path)
    dataloader_ssl = DataLoader(
        dataset_ssl,
        batch_size=ssl_cfg.dataset.batch_size,
        shuffle=True,
        num_workers=ssl_cfg.dataset.num_workers,
        drop_last=True
    )
    print(f"→ Trovate {len(dataset_ssl)} immagini da processare.")

    # ------------------------------------------------------
    # 2. Modello SSL
    # ------------------------------------------------------
    console_step("Caricamento o Training Modello")
    ssl_model_path = os.path.join(paths.save_dir, "ssl_model_final.pth")

    model = SSLModel(
        backbone_name=ssl_cfg.model.backbone,
        pretrained=True,
        embedding_dim=ssl_cfg.model.embedding_dim,
        proj_hidden_dim=ssl_cfg.model.proj_hidden_dim,
        proj_output_dim=ssl_cfg.model.proj_output_dim,
        proj_num_layers=ssl_cfg.model.proj_num_layers,
        freeze_backbone=ssl_cfg.model.freeze_backbone
    ).to(device)

    if os.path.exists(ssl_model_path):
        checkpoint = torch.load(ssl_model_path, map_location=device)
        # supporta checkpoint sia intero che dict con "model_state"
        state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state, strict=False)
        print(f"{Fore.GREEN}✔ Modello trovato e caricato da:{Style.RESET_ALL} {ssl_model_path}")
    else:
        print(f"{Fore.YELLOW}⚙ Addestramento modello SSL...{Style.RESET_ALL}")
        model = train_ssl(
            model=model,
            dataloader=dataloader_ssl,
            data_dir=paths.dataset_path,
            device=device,
            epochs=ssl_cfg.training.epochs,
            lr=ssl_cfg.training.lr,
            temperature=ssl_cfg.training.temperature,
            save_dir=paths.save_dir,
            gradient_clip=ssl_cfg.training.gradient_clip,
            amp=ssl_cfg.training.amp,
            save_every=ssl_cfg.training.save_every,
            warmup_epochs=ssl_cfg.training.warmup_epochs
        )
        torch.save(model.state_dict(), ssl_model_path)
        print(f"{Fore.GREEN}✔ Modello salvato in:{Style.RESET_ALL} {ssl_model_path}")

    # ------------------------------------------------------
    # 3. Estrazione embeddings
    # ------------------------------------------------------
    console_step("Estrazione Embeddings")
    embeddings, filenames = extract_embeddings(
        data_dir=paths.dataset_path,
        save_dir=paths.save_dir,
        model=model,
        device=device,
        batch_size=ssl_cfg.dataset.batch_size
    )
    print(f"→ Embeddings estratti: {embeddings.shape[0]} campioni di dimensione {embeddings.shape[1]}")
    torch.save({"embeddings": embeddings, "filenames": filenames},
               os.path.join(paths.save_dir, "embeddings.pth"))

    # ------------------------------------------------------
    # 4. Clustering globale
    # ------------------------------------------------------
    console_step("Clustering Globale")

    # --- KMeans ---
    labels_kmeans, report_kmeans = cluster_kmeans(
        embeddings, 
        n_clusters=ssl_cfg.clustering.n_clusters,
        dim_reduction=ssl_cfg.clustering.dim_reduction,
        dim=ssl_cfg.clustering.dim
    )

    # --- Agglomerative ---
    labels_agg, report_agg = cluster_agglomerative(
        embeddings,
        n_clusters=ssl_cfg.clustering.n_clusters,
        metric='cosine',
        dim_reduction=ssl_cfg.clustering.dim_reduction,
        dim=ssl_cfg.clustering.dim
    )

    # --- Salvataggio report completo ---
    metrics_path = os.path.join(paths.save_dir, "clustering_report_detailed.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "kmeans": report_kmeans,
            "agglomerative": report_agg
        }, f, indent=2)

    print(f"✔ Report dettagliato salvato in {metrics_path}")

    # ------------------------------------------------------
    # 5. Visualizzazione
    # ------------------------------------------------------
    if ssl_cfg.visualization.visualize_tsne:
        console_step("t-SNE Visualization")
        visualize_embeddings(
            embeddings, labels=labels_kmeans,
            method="tsne",
            save_path=os.path.join(paths.figures_dir, "tsne_kmeans.png")
        )
        visualize_embeddings(
            embeddings, labels=labels_agg,
            method="tsne",
            save_path=os.path.join(paths.figures_dir, "tsne_agglomerative.png")
        )

    if ssl_cfg.visualization.visualize_umap:
        console_step("UMAP Visualization")
        visualize_embeddings(
            embeddings, labels=labels_kmeans,
            method="umap",
            save_path=os.path.join(paths.figures_dir, "umap_kmeans.png")
        )
        visualize_embeddings(
            embeddings, labels=labels_agg,
            method="umap",
            save_path=os.path.join(paths.figures_dir, "umap_agglomerative.png")
        )

    # ------------------------------------------------------
    # 6. Aggregazione Embeddings per ID
    # ------------------------------------------------------
    console_step("Aggregazione Embeddings per ID")

    id_to_embeddings = defaultdict(list)
    id_to_filenames = defaultdict(list)

    for emb, fname in zip(embeddings, filenames):
        file_id = extract_id(fname)
        id_to_embeddings[file_id].append(emb)
        id_to_filenames[file_id].append(str(fname))

    agg_embeddings = np.stack([
        np.mean(np.stack(v), axis=0) for v in id_to_embeddings.values()
    ])
    id_list = list(id_to_embeddings.keys())

    print(f"→ Raggruppati {len(id_list)} ID unici ({len(embeddings)} immagini totali).")
    logging.info(f"Aggregati {len(id_list)} ID unici da {len(embeddings)} immagini totali.")


    # ------------------------------------------------------
    # 7. Salvataggio Risultati
    # ------------------------------------------------------
    console_step("Salvataggio Risultati")

    id_labels = []
    for fid, emb_list in id_to_embeddings.items():
        mean_emb = np.mean(np.stack(emb_list), axis=0)
        distances = np.linalg.norm(embeddings - mean_emb, axis=1)
        closest_idx = np.argmin(distances)
        id_labels.append(int(labels_kmeans[closest_idx]))

    csv_path = os.path.join(paths.save_dir, "id_clusters.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "path", "global_id", "cluster_label"])

        for fid, cluster_label in zip(id_list, id_labels):
            for full_fname in id_to_filenames[fid]:
                filename_only = Path(full_fname).name
                full_path = str(Path(paths.dataset_path) / filename_only)
                writer.writerow([
                    filename_only,   # nome file
                    full_path,       # path completo
                    fid,             # ID globale
                    cluster_label    # cluster assegnato
                ])

    print(f"{Fore.GREEN}✔ Risultati finali salvati in:{Style.RESET_ALL} {csv_path}")
    logging.info(f"Risultati finali salvati in: {csv_path}")

    print(f"\n{Fore.CYAN}✨ Pipeline SSL completata con successo! ✨{Style.RESET_ALL}")

# ==========================================================
# ENTRYPOINT
# ==========================================================
if __name__ == "__main__":
    main()
