import os

# ============================================================
# DIRECTORY E SALVATAGGIO
# ============================================================
DATASET_PATH = "dataset/DBII"
SAVE_DIR = "classifier/save_models"
FIGURES_DIR = "classifier/figures"

# ============================================================
# SSL / CONTRASTIVE LEARNING
# ============================================================
CONFIG = {
    # Dataset
    "dataset_path": DATASET_PATH,
    "batch_size": 64,
    
    # Backbone
    "backbone": "resnet18",          # resnet18, resnet50, efficientnet_b0, ecc.
    "projection_dim": 128,           # Dimensione dello spazio embedding
    
    # Training SSL
    "epochs": 1,
    "lr": 1e-3,
    "temperature": 0.5,              # Parametro per NT-Xent loss
    
    # Clustering
    "n_clusters": 5,
    "min_cluster_size": 5,
    
    # Fine-tuning supervisionato
    "num_classes": 5,               # Se si hanno label reali / pseudo-label
    "finetune_epochs": 10,
    "train_split": 0.8,
    
    # Altri
    "save_dir": SAVE_DIR,
    "figures_dir": FIGURES_DIR
}
