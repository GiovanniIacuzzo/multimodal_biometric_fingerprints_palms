import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
from typing import List, Tuple
from pathlib import Path
from classifier.dataset2.preprocessing import preprocess_image


# ==========================================================
# Augmentazioni per contrastive learning
# ==========================================================
class FingerprintAugmentations:
    """
    Augmentazioni realistiche per contrastive learning su impronte digitali.
    Include:
    - Rotazioni casuali lievi (+ occasionali rotazioni forti)
    - Flip orizzontale e verticale
    - Crop e resize
    - Jitter di contrasto/luminosità
    - Noise gaussiano
    """

    def __init__(self, image_size=256):
        self.image_size = image_size

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        # Assicuriamoci che l'immagine sia float [0,1]
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0

        h, w = img.shape

        # --- Rotazione casuale (piccola o multipla di 90°) ---
        if random.random() < 0.8:
            angle = random.uniform(-15, 15)  # piccola rotazione
        else:
            angle = random.choice([0, 90, 180, 270])  # forte rotazione
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        # --- Flip orizzontale e verticale ---
        if random.random() < 0.5:
            img = np.fliplr(img)
        if random.random() < 0.3:
            img = np.flipud(img)

        # --- Random crop + resize ---
        crop_scale = random.uniform(0.8, 1.0)
        crop_size = int(crop_scale * min(h, w))
        if crop_size < min(h, w):
            x = random.randint(0, w - crop_size)
            y = random.randint(0, h - crop_size)
            img = img[y:y + crop_size, x:x + crop_size]
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        # --- Jitter di luminosità/contrasto ---
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)  # contrasto
            beta = random.uniform(-0.1, 0.1)  # luminosità
            img = np.clip(alpha * img + beta, 0, 1)

        # --- Gaussian noise ---
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.015, img.shape)
            img = np.clip(img + noise, 0, 1)

        # --- Normalizzazione finale ---
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()  # (1, H, W)
        return img_tensor


# ==========================================================
# Dataset contrastivo (SimCLR-style)
# ==========================================================
class FingerprintDataset(Dataset):
    """
    Dataset per contrastive learning:
    - Carica immagini grayscale
    - Restituisce due versioni augmentate della stessa immagine
    """

    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = Path(dataset_dir)
        self.image_paths = sorted(list(self.dataset_dir.glob("**/*.jpg")))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"Nessuna immagine trovata in {dataset_dir}")
        self.transform = transform if transform else FingerprintAugmentations()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Immagine non leggibile: {img_path}")

        # due versioni augmentate
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2


# ==========================================================
# Dataset base per feature extraction / clustering
# ==========================================================
class BaseDataset(Dataset):
    """
    Dataset per fase di embedding o clustering:
    - Preprocessing coerente con quello usato nel training SSL
    """

    def __init__(self, root_dir: str, extensions: List[str] = [".jpg", ".png"]):
        self.root_dir = Path(root_dir)
        self.img_paths = [p for p in self.root_dir.glob("**/*") if p.suffix.lower() in extensions]
        self.img_paths = sorted(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = self.img_paths[idx]
        img = preprocess_image(img_path)

        if img is None:
            img = np.zeros((256, 256), dtype=np.float32)

        img_tensor = torch.from_numpy(img).unsqueeze(0).float()  # (1,H,W)
        return img_tensor, str(img_path)
