import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path

from classifier.dataset2.dataset import BaseDataset
from classifier.models.backbone import CNNBackbone
from classifier.models.projection_head import ProjectionHead
from classifier.models.ssl_model import SSLModel
from classifier.utils.utils import load_model
import cv2

# ===========================================================
# Funzione sicura per leggere e preprocessare immagini
# ===========================================================
def preprocess_image(img_path_or_array, resize=(256, 256), local_norm=True, align=True):
    # Se è un path, carica l'immagine
    if isinstance(img_path_or_array, (str, Path)):
        img = cv2.imread(str(img_path_or_array), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros(resize, dtype=np.uint8)
    else:
        img = img_path_or_array

    # Resize e normalizzazione globale
    img_resized = cv2.resize(img, resize).astype(np.float32)
    img_resized /= 255.0

    if local_norm:
        # Normalizzazione locale
        mean_local = cv2.blur(img_resized, (15,15))
        std_local = cv2.blur((img_resized - mean_local)**2, (15,15))**0.5 + 1e-8
        img_resized = (img_resized - mean_local) / std_local
        img_resized = (img_resized - img_resized.min()) / (img_resized.max() - img_resized.min() + 1e-8)

    if align:
        # Allineamento secondo orientazione dominante
        gy, gx = np.gradient(img_resized)
        orientation = np.arctan2(gy, gx)
        hist, bins = np.histogram(orientation, bins=180, range=(-np.pi, np.pi))
        angle = bins[np.argmax(hist)]
        angle_deg = np.degrees(angle)
        h, w = img_resized.shape
        M = cv2.getRotationMatrix2D((w//2, h//2), angle_deg, 1.0)
        img_resized = cv2.warpAffine(img_resized, M, (w, h), flags=cv2.INTER_LINEAR)

    # Converti a tensor CxHxW
    img_tensor = torch.from_numpy(img_resized).unsqueeze(0)  # 1xHxW
    return img_tensor


# ===========================================================
# Funzione principale di estrazione embeddings
# ===========================================================
def extract_embeddings(
    data_dir: str = None,
    dataset: torch.utils.data.Dataset = None,
    model_path: str = None,
    save_dir: str = "classifier/save_models/save_model_embeddings/embeddings",
    batch_size: int = 64,
    device: str = None,
    pretrained_backbone: bool = False,
    model: torch.nn.Module = None,
):
    """
    Estrae embeddings dalle immagini usando un modello SSL pretrained.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(save_dir, exist_ok=True)

    # ----------------------------
    # DATASET & DATALOADER
    # ----------------------------
    if dataset is not None:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    elif data_dir is not None:
        dataset = BaseDataset(data_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        raise ValueError("Devi fornire almeno `data_dir` o `dataset`")

    # ----------------------------
    # MODEL
    # ----------------------------
    if model is None:
        backbone = CNNBackbone(pretrained=pretrained_backbone)
        projection = ProjectionHead(input_dim=backbone.output_dim, hidden_dim=512, output_dim=128)
        model = SSLModel(backbone, projection).to(device)
        load_model(model, model_path)
    else:
        model = model.to(device)

    model.eval()

    # ----------------------------
    # ESTRAZIONE EMBEDDING
    # ----------------------------
    all_embeddings = []
    all_paths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # Supporta dataset che restituiscono tuple (img_tensor, path)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                imgs, paths = batch
            else:
                imgs = batch
                paths = [None]*len(imgs)

            imgs = imgs.to(device)
            z = model(imgs)  # usa forward normale
            z = z.cpu().numpy()
            all_embeddings.append(z)
            all_paths.extend(paths)

    all_embeddings = np.vstack(all_embeddings)

    # ----------------------------
    # Salvataggio
    # ----------------------------
    np.save(os.path.join(save_dir, "embeddings.npy"), all_embeddings)
    with open(os.path.join(save_dir, "paths.txt"), "w") as f:
        for p in all_paths:
            f.write(f"{p}\n")

    print(f"Saved embeddings: {all_embeddings.shape} → {save_dir}")
    return all_embeddings, all_paths
