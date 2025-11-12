import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from classifier.dataset2.dataset import BaseDataset
from classifier.models.ssl_model import SSLModel
from classifier.utils.utils import load_model


def extract_embeddings(
    data_dir: str = None,
    dataset: torch.utils.data.Dataset = None,
    model: torch.nn.Module = None,
    model_path: str = None,
    save_dir: str = "save_model_embeddings/embeddings",
    batch_size: int = 64,
    device: str = None,
    l2_normalize: bool = True,
    num_workers: int = 4,
    overwrite: bool = False,
    amp: bool = True,
):
    """
    Estrae embeddings da immagini biometriche con caching, mixed precision e normalizzazione opzionale.
    """
    device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    npz_path = save_dir / "embeddings.npz"
    if npz_path.exists() and not overwrite:
        print(f"[INFO] Embeddings già esistenti in {npz_path}, caricamento in corso...")
        data = np.load(npz_path, allow_pickle=True)
        return data["embeddings"], data["paths"].tolist()

    # Dataset
    if dataset is None:
        if data_dir is None:
            raise ValueError("Provide either dataset or data_dir")
        dataset = BaseDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Modello
    if model is None:
        model = SSLModel(backbone_name="resnet18").to(device)
        if model_path:
            load_model(model, model_path, device=device)
    else:
        model = model.to(device)
    model.eval()

    embeddings_list, paths_list = [], []
    autocast = torch.cuda.amp.autocast if (amp and device == "cuda") else torch.cpu.amp.autocast

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings", unit="batch"):
            try:
                imgs, paths = batch if isinstance(batch, (tuple, list)) else (batch, [None] * len(batch))
                imgs = imgs.to(device, non_blocking=True)
                with autocast():
                    emb = model.backbone(imgs)
                emb = emb.detach().cpu().numpy()
                embeddings_list.append(emb)
                paths_list.extend(paths)
            except Exception as e:
                print(f"[WARN] Skipped batch due to error: {e}")
                continue

    if not embeddings_list:
        raise RuntimeError("No embeddings extracted! Check dataset or model.")

    embeddings = np.vstack(embeddings_list)
    if l2_normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings /= np.clip(norms, 1e-8, None)

    np.savez_compressed(npz_path, embeddings=embeddings, paths=np.array(paths_list))
    print(f"[DONE] Saved embeddings: {embeddings.shape} → {npz_path}")
    return embeddings, paths_list
