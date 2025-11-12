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
    cluster_kmeans, cluster_hdbscan, visualize_embeddings
)
from config.config_classifier import CONFIG


# ==========================================================
# LOGGING SETUP
# ==========================================================
os.makedirs(CONFIG.save_dir, exist_ok=True)
logging.basicConfig(
    filename=CONFIG.log_file,
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


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
    os.makedirs(CONFIG.figures_dir, exist_ok=True)

    console_step("Inizializzazione")
    print(f"{Fore.GREEN}Device in uso:{Style.RESET_ALL} {device}")
    logging.info(f"Using device: {device}")

    # ------------------------------------------------------
    # 1. Dataset SSL
    # ------------------------------------------------------
    console_step("Caricamento Dataset")
    dataset_ssl = FingerprintDataset(CONFIG.dataset_path)
    dataloader_ssl = DataLoader(
        dataset_ssl,
        batch_size=CONFIG.batch_size,
        shuffle=True,
        num_workers=CONFIG.num_workers,
        drop_last=True
    )
    print(f"→ Trovate {len(dataset_ssl)} immagini da processare.")

    # ------------------------------------------------------
    # 2. Modello SSL
    # ------------------------------------------------------
    console_step("Caricamento o Training Modello")
    ssl_model_path = os.path.join(CONFIG.save_dir, "ssl_model_final.pth")

    model = SSLModel(
        backbone_name=CONFIG.backbone,
        pretrained=True,
        embedding_dim=CONFIG.embedding_dim,
        proj_hidden_dim=CONFIG.proj_hidden_dim,
        proj_output_dim=CONFIG.proj_output_dim,
        proj_num_layers=CONFIG.proj_num_layers,
        freeze_backbone=CONFIG.freeze_backbone
    ).to(device)

    if os.path.exists(ssl_model_path):
        checkpoint = torch.load(ssl_model_path, map_location=device)
        model.load_state_dict(checkpoint.get("model_state", checkpoint), strict=False)
        print(f"{Fore.GREEN}✔ Modello trovato e caricato da:{Style.RESET_ALL} {ssl_model_path}")
    else:
        print(f"{Fore.YELLOW}⚙ Addestramento modello SSL...{Style.RESET_ALL}")
        model = train_ssl(
            model=model,
            dataloader=dataloader_ssl,
            data_dir=CONFIG.dataset_path,
            device=device,
            epochs=CONFIG.epochs,
            lr=CONFIG.lr,
            temperature=CONFIG.temperature,
            save_dir=CONFIG.save_dir,
            gradient_clip=getattr(CONFIG, "gradient_clip", 1.0),
            amp=getattr(CONFIG, "amp", True),
            save_every=getattr(CONFIG, "save_every", 5),
            warmup_epochs=getattr(CONFIG, "warmup_epochs", 5)
        )
        torch.save(model.state_dict(), ssl_model_path)
        print(f"{Fore.GREEN}✔ Modello salvato in:{Style.RESET_ALL} {ssl_model_path}")

    # ------------------------------------------------------
    # 3. Estrazione embeddings
    # ------------------------------------------------------
    console_step("Estrazione Embeddings")
    embeddings, filenames = extract_embeddings(
        data_dir=CONFIG.dataset_path,
        model=model,
        device=device,
        batch_size=CONFIG.batch_size
    )
    print(f"→ Embeddings estratti: {embeddings.shape[0]} campioni di dimensione {embeddings.shape[1]}")
    torch.save({"embeddings": embeddings, "filenames": filenames},
               os.path.join(CONFIG.save_dir, "embeddings.pth"))

    # ------------------------------------------------------
    # 4. Clustering globale
    # ------------------------------------------------------
    console_step("Clustering Globale")
    labels_kmeans, metrics_kmeans = cluster_kmeans(embeddings, n_clusters=CONFIG.n_clusters)
    labels_hdbscan, metrics_hdbscan = cluster_hdbscan(embeddings, min_cluster_size=CONFIG.min_cluster_size)

    metrics_path = os.path.join(CONFIG.save_dir, "clustering_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"kmeans": safe_json(metrics_kmeans), "hdbscan": safe_json(metrics_hdbscan)}, f, indent=2)
    print(f"✔ Metriche di clustering salvate in {metrics_path}")

    # ------------------------------------------------------
    # 5. Visualizzazione
    # ------------------------------------------------------
    if CONFIG.visualize_tsne:
        console_step("t-SNE Visualization")
        visualize_embeddings(embeddings, labels=labels_kmeans,
                     method="tsne",
                     save_path=os.path.join(CONFIG.figures_dir, "tsne_kmeans.png"))
        visualize_embeddings(embeddings, labels=labels_hdbscan,
                     method="tsne",
                     save_path=os.path.join(CONFIG.figures_dir, "tsne_hdbscan.png"))
    if CONFIG.visualize_umap:
        console_step("UMAP Visualization")
        visualize_embeddings(embeddings, labels=labels_kmeans,
                     method="umap",
                     save_path=os.path.join(CONFIG.figures_dir, "umap_kmeans.png"))
        visualize_embeddings(embeddings, labels=labels_hdbscan,
                     method="umap",
                     save_path=os.path.join(CONFIG.figures_dir, "umap_hdbscan.png"))


    # ------------------------------------------------------
    # 6. Aggregazione per ID
    # ------------------------------------------------------
    console_step("Aggregazione Embeddings per ID")
    id_pattern = re.compile(r"^0*([0-9]+)")
    id_to_embeddings = defaultdict(list)
    id_to_filenames = defaultdict(list)

    for emb, fname in zip(embeddings, filenames):
        stem = Path(str(fname)).stem
        match = id_pattern.match(stem)
        file_id = match.group(1) if match else stem
        id_to_embeddings[file_id].append(emb)
        id_to_filenames[file_id].append(str(fname))

    agg_embeddings = np.stack([np.mean(np.stack(v), axis=0) for v in id_to_embeddings.values()])
    id_list = list(id_to_embeddings.keys())
    print(f"→ Raggruppati {len(id_list)} ID unici ({len(embeddings)} immagini totali).")

    # ------------------------------------------------------
    # 7. Clustering a livello ID
    # ------------------------------------------------------
    console_step("Clustering per ID")
    if len(id_list) < CONFIG.n_clusters:
        raise ValueError(f"Non ci sono abbastanza ID ({len(id_list)}) per {CONFIG.n_clusters} cluster.")
    id_labels, _ = cluster_kmeans(agg_embeddings, n_clusters=CONFIG.n_clusters)
    print(f"✔ Clustering completato: {CONFIG.n_clusters} cluster generati.")

    # ------------------------------------------------------
    # 8. Salvataggio risultati
    # ------------------------------------------------------
    console_step("Salvataggio Risultati")
    csv_path = os.path.join(CONFIG.save_dir, "id_level_clusters.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "path", "global_class", "cluster_in_class"])
        for fid, cluster in zip(id_list, id_labels):
            for fname in id_to_filenames[fid]:
                writer.writerow([fname, os.path.join(CONFIG.dataset_path, fname), fid, int(cluster)])
    print(f"{Fore.GREEN}✔ Risultati finali salvati in:{Style.RESET_ALL} {csv_path}")
    print(f"\n{Fore.CYAN}✨ Pipeline SSL completata con successo! ✨{Style.RESET_ALL}")

# ==========================================================
# ENTRYPOINT
# ==========================================================
if __name__ == "__main__":
    main()
