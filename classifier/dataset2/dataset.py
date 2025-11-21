import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
from typing import List, Tuple, Union
from pathlib import Path
from classifier.dataset2.preprocessing import preprocess_image

# ==========================================================
# Augmentazioni per contrastive learning
# ==========================================================
class FingerprintAugmentations:
    def __init__(self, image_size=224):
        self.image_size = image_size

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        img = img.astype(np.float32)
        if img.max() > 1.0:
            img /= 255.0

        h, w = img.shape

        # --- Rotazione casuale ---
        if random.random() < 0.8:
            angle = random.uniform(-15, 15)
        else:
            angle = random.choice([0, 90, 180, 270])
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        # --- Flips ---
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

        # --- Jitter luminosità e contrasto ---
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-0.1, 0.1)
            img = np.clip(alpha * img + beta, 0, 1)

        # --- Rumore gaussiano ---
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.015, img.shape)
            img = np.clip(img + noise, 0, 1)

        return torch.from_numpy(img).unsqueeze(0).float()

def collect_image_paths(dirs: List[Path], extensions: List[str]) -> List[Path]:
    paths = []
    for d in dirs:
        for ext in extensions:
            paths.extend(d.rglob(f"*{ext}"))
    return sorted(paths)

class FingerprintDataset(Dataset):
    def __init__(self, dataset_dirs: Union[str, Path, List[Union[str, Path]]], transform=None):
        # Normalizzazione input in lista
        if isinstance(dataset_dirs, (str, Path)):
            dataset_dirs = [dataset_dirs]

        self.dataset_dirs = [Path(d) for d in dataset_dirs]
        self.transform = transform if transform else FingerprintAugmentations()

        # Estensioni supportate
        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

        # Raccolta immagini
        self.image_paths = collect_image_paths(self.dataset_dirs, extensions)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"Nessuna immagine trovata nelle directory: {self.dataset_dirs}")

        print(f"[FingerprintDataset] Caricate {len(self.image_paths)} immagini da {len(self.dataset_dirs)} directory.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Immagine non leggibile: {img_path}")

        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2


# ==========================================================
# Dataset base per feature extraction / clustering
# ==========================================================
class BaseDataset(Dataset):
    def __init__(
        self,
        root_dirs: Union[str, Path, List[Union[str, Path]]],
        extensions: List[str] = [".jpg", ".jpeg", ".png", ".bmp"],
        image_size: int = 224
    ):
        # Normalizza input a lista
        if isinstance(root_dirs, (str, Path)):
            root_dirs = [root_dirs]

        self.root_dirs = [Path(d) for d in root_dirs]
        self.image_size = image_size

        # Raccolta immagini
        self.img_paths = collect_image_paths(self.root_dirs, extensions)

        if len(self.img_paths) == 0:
            raise RuntimeError(f"Nessuna immagine trovata nelle directory: {self.root_dirs}")

        print(f"[BaseDataset] Caricate {len(self.img_paths)} immagini da {len(self.root_dirs)} directory.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = self.img_paths[idx]
        img = preprocess_image(img_path)

        if img is None:
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        if img.shape != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0

        img_tensor = torch.from_numpy(img).unsqueeze(0).float()
        return img_tensor, str(img_path)
