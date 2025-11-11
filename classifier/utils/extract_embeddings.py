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
    num_workers: int = 4
):
    """
    Estrae embeddings da immagini.

    Parametri:
    - data_dir: cartella contenente le immagini (se dataset non fornito)
    - dataset: dataset PyTorch già creato (opzionale)
    - model: modello PyTorch già istanziato (opzionale)
    - model_path: percorso checkpoint modello da caricare (opzionale)
    - save_dir: cartella dove salvare embeddings
    - batch_size: dimensione batch
    - device: 'cuda', 'mps' o 'cpu'
    - l2_normalize: se True normalizza gli embeddings
    - num_workers: worker DataLoader
    """
    # device
    device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    os.makedirs(save_dir, exist_ok=True)

    # dataset
    if dataset is None:
        if data_dir is None:
            raise ValueError("Provide either dataset or data_dir")
        dataset = BaseDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # modello
    if model is None:
        model = SSLModel(backbone_name="resnet18").to(device)
        if model_path:
            load_model(model, model_path, device=device)
    else:
        model = model.to(device)

    model.eval()

    embeddings_list = []
    paths_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # batch può essere (imgs, paths) o solo imgs
            if isinstance(batch, (tuple, list)) and len(batch) == 2 and not torch.is_tensor(batch[1]):
                imgs, paths = batch
            elif isinstance(batch, (tuple, list)) and len(batch) == 2 and torch.is_tensor(batch[1]):
                imgs, paths = batch[0], [None]*batch[0].size(0)
            else:
                imgs = batch
                paths = [None] * (imgs.size(0) if torch.is_tensor(imgs) else len(imgs))

            imgs = imgs.to(device, non_blocking=True)

            # forward
            if hasattr(model, "backbone") and hasattr(model, "projection_head"):
                # SSLModel: prendiamo output backbone
                emb = model.backbone(imgs)
            else:
                emb = model(imgs)

            emb = emb.detach().cpu().numpy()
            embeddings_list.append(emb)
            paths_list.extend(paths)

    if len(embeddings_list) == 0:
        embeddings = np.zeros((0,0), dtype=float)
    else:
        embeddings = np.vstack(embeddings_list)

    # normalizzazione L2 opzionale
    if l2_normalize and embeddings.shape[0] > 0:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        embeddings = embeddings / norms

    # salvataggio
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "embeddings.npy"), embeddings)
    with open(os.path.join(save_dir, "paths.txt"), "w") as f:
        for p in paths_list:
            f.write(f"{p}\n")

    # torch-friendly dict
    try:
        torch.save({"embeddings": embeddings, "filenames": paths_list},
                   os.path.join(save_dir, "embeddings.pth"))
    except Exception as e:
        print(f"Warning: unable to save embeddings.pth: {e}")

    print(f"Saved embeddings: {embeddings.shape} -> {save_dir}")
    return embeddings, paths_list
