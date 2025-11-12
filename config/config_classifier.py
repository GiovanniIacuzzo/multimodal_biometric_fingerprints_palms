import os
from types import SimpleNamespace

# ============================================================
# PATH PRINCIPALI
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))      # directory corrente /config
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../"))  # progetto principale

DATASET_PATH = os.path.join(ROOT_DIR, "dataset", "DBII")
SAVE_DIR = os.path.join(ROOT_DIR, "classifier", "save_models")
FIGURES_DIR = os.path.join(ROOT_DIR, "classifier", "figures")
SORTED_DIR = os.path.join(ROOT_DIR, "dataset", "sorted_dataset")

# Crea automaticamente le directory se non esistono
for d in [SAVE_DIR, FIGURES_DIR, SORTED_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# CONFIGURAZIONE PRINCIPALE DEL CLASSIFICATORE SSL
# ============================================================
CONFIG = SimpleNamespace(
    # === Dataset ===
    dataset_path=os.path.join(ROOT_DIR, "dataset", "DBII"),
    batch_size=16,
    num_workers=2,
    seed=42,

    # === Modello ===
    backbone="resnet50",
    embedding_dim=512,
    proj_hidden_dim=512,
    proj_output_dim=256,
    proj_num_layers=2,
    freeze_backbone=True,

    # === Training SSL ===
    epochs=5,
    lr=1e-3,
    temperature=0.5,
    optimizer="adam",
    weight_decay=1e-4,
    gradient_clip=1.0,
    amp=False,
    save_every=10,
    warmup_epochs=5,

    # === Clustering ===
    n_clusters=5,
    min_cluster_size=5,
    cluster_metric="cosine",

    # === Fine-tuning supervisionato (opzionale) ===
    num_classes=8,
    finetune_epochs=10,
    train_split=0.8,

    # === Logging e salvataggi ===
    save_dir=os.path.join(ROOT_DIR, "classifier", "save_models"),
    figures_dir=os.path.join(ROOT_DIR, "classifier", "figures"),
    log_file=os.path.join(ROOT_DIR, "classifier", "save_models", "train.log"),

    # === Visualizzazione ===
    visualize_tsne=True,
    visualize_umap=True
)


# ============================================================
# CONFIGURAZIONE PER ORDINAMENTO E ANALISI CLUSTER
# ============================================================
CONFIG_SORTED = {
    # === Input ===
    "csv_path": os.path.join(SAVE_DIR, "id_level_clusters.csv"),
    "dataset_root": DATASET_PATH,
    "embeddings_path": os.path.join(SAVE_DIR, "classifier/save_models/embeddings.pth"),

    # === Output ===
    "output_dir": SORTED_DIR,
    "copy_mode": True,                        # True = copia file, False = sposta
    "overwrite_existing": False,

    # === Metriche ===
    "compute_metrics": True,                  # silhouette, davies-bouldin, calinski-harabasz
    "max_missing_display": 10,

    # === Visualizzazione ===
    "save_cluster_figures": True,
    "figure_format": "png"
}

# ============================================================
# STAMPA RIEPILOGO CONFIGURAZIONE
# ============================================================
if __name__ == "__main__":
    print("=== CONFIGURAZIONE SSL ===")
    for k, v in CONFIG.items():
        print(f"{k:25s}: {v}")
    print("\n=== CONFIGURAZIONE SORTED ===")
    for k, v in CONFIG_SORTED.items():
        print(f"{k:25s}: {v}")
