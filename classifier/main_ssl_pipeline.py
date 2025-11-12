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
        save_dir=CONFIG.save_dir,
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

    labels_kmeans, report_kmeans = cluster_kmeans(embeddings, n_clusters=CONFIG.n_clusters)
    labels_hdbscan, report_hdbscan = cluster_hdbscan(embeddings, min_cluster_size=CONFIG.min_cluster_size)

    # salva report completo
    metrics_path = os.path.join(CONFIG.save_dir, "clustering_report_detailed.json")
    with open(metrics_path, "w") as f:
        json.dump({"kmeans": report_kmeans, "hdbscan": report_hdbscan}, f, indent=2)

    print(f"✔ Report dettagliato salvato in {metrics_path}")

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


"""
ULTIMO RISULTATO UTILE:
(multimodal_biometric) giovanni02@MacBook-Air-del-Professore multimodal_biometric_fingerprints_palms % python -m classifier.main_ssl_pipeline

============================================================
INIZIALIZZAZIONE
============================================================
Device in uso: mps

============================================================
CARICAMENTO DATASET
============================================================
→ Trovate 1480 immagini da processare.

============================================================
CARICAMENTO O TRAINING MODELLO
============================================================
/opt/miniconda3/envs/multimodal_biometric/lib/python3.10/site-packages/torch/nn/utils/weight_norm.py:28: UserWarning: torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.
  warnings.warn("torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.")
⚙ Addestramento modello SSL...
Epoch 1/5: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 92/92 [02:22<00:00,  1.55s/b]
[1/5] loss=2.6436 lr=2.00e-04 time=142.6s
Epoch 2/5: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 92/92 [02:18<00:00,  1.50s/b]
[2/5] loss=2.3810 lr=4.00e-04 time=138.1s
Epoch 3/5: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 92/92 [02:17<00:00,  1.49s/b]
[3/5] loss=2.3243 lr=6.00e-04 time=137.5s
Epoch 4/5: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 92/92 [02:19<00:00,  1.52s/b]
[4/5] loss=2.3003 lr=8.00e-04 time=139.6s
Epoch 5/5: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 92/92 [02:28<00:00,  1.61s/b]
[5/5] loss=2.2797 lr=1.00e-03 time=148.4s
[DEBUG] Sample embedding norm mean: 24.5183
[DONE] Training completed. Best loss: 2.2797
✔ Modello salvato in: /Users/giovanni02/Desktop/UNIKORE/multimodal_biometric_fingerprints_palms/classifier/save_models/ssl_model_final.pth

============================================================
ESTRAZIONE EMBEDDINGS
============================================================
Extracting embeddings: 100%|███████████████████████████████████████████████████████████████████████████████████████| 93/93 [01:21<00:00,  1.14batch/s]
[DONE] Saved embeddings: (1480, 256) → /Users/giovanni02/Desktop/UNIKORE/multimodal_biometric_fingerprints_palms/classifier/save_models/embeddings.npz
→ Embeddings estratti: 1480 campioni di dimensione 256

============================================================
CLUSTERING GLOBALE
============================================================
/opt/miniconda3/envs/multimodal_biometric/lib/python3.10/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
OMP: Info #276: omp_set_nested routine deprecated, please use omp_set_max_active_levels instead.
/opt/miniconda3/envs/multimodal_biometric/lib/python3.10/site-packages/sklearn/utils/deprecation.py:132: FutureWarning: 'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.
  warnings.warn(
/opt/miniconda3/envs/multimodal_biometric/lib/python3.10/site-packages/sklearn/utils/deprecation.py:132: FutureWarning: 'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.
  warnings.warn(
✔ Report dettagliato salvato in /Users/giovanni02/Desktop/UNIKORE/multimodal_biometric_fingerprints_palms/classifier/save_models/clustering_report_detailed.json

============================================================
T-SNE VISUALIZATION
============================================================

============================================================
UMAP VISUALIZATION
============================================================
/opt/miniconda3/envs/multimodal_biometric/lib/python3.10/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(
/opt/miniconda3/envs/multimodal_biometric/lib/python3.10/site-packages/umap/umap_.py:1952: UserWarning: n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism.
  warn(

============================================================
AGGREGAZIONE EMBEDDINGS PER ID
============================================================
→ Raggruppati 148 ID unici (1480 immagini totali).

============================================================
CLUSTERING PER ID
============================================================
✔ Clustering completato: 5 cluster generati.

============================================================
SALVATAGGIO RISULTATI
============================================================
✔ Risultati finali salvati in: /Users/giovanni02/Desktop/UNIKORE/multimodal_biometric_fingerprints_palms/classifier/save_models/id_level_clusters.csv

✨ Pipeline SSL completata con successo! ✨
(multimodal_biometric) giovanni02@MacBook-Air-del-Professore multimodal_biometric_fingerprints_palms % 

"""