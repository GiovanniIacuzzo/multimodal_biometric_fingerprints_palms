import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
from typing import List, Tuple
from pathlib import Path
from classifier.dataset2.preprocessing import preprocess_image

# ==========================================
# Funzioni di augmentation
# ==========================================

class FingerprintAugmentations:
    """
    Transformazioni per contrastive learning:
    - Rotazioni casuali
    - Flip orizzontale/verticale
    - Noise gaussiano
    - Crop casuale
    """
    def __init__(self, image_size=256):
        self.image_size = image_size

    def __call__(self, img):
        # img: numpy array HxW
        img = img.astype(np.float32) / 255.0

        # --- Random rotation ---
        angle = random.choice([0, 90, 180, 270])
        M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # --- Random flip ---
        if random.random() > 0.5:
            img = np.fliplr(img)
        if random.random() > 0.5:
            img = np.flipud(img)

        # --- Random crop and resize ---
        h, w = img.shape
        crop_size = int(0.9 * min(h, w))
        x = random.randint(0, w - crop_size)
        y = random.randint(0, h - crop_size)
        img = img[y:y+crop_size, x:x+crop_size]
        img = cv2.resize(img, (self.image_size, self.image_size))

        # --- Add small Gaussian noise ---
        if random.random() > 0.5:
            noise = np.random.normal(0, 0.02, img.shape)
            img = np.clip(img + noise, 0, 1)

        # Convert to torch tensor
        img_tensor = torch.from_numpy(img).unsqueeze(0).float()  # 1xHxW
        return img_tensor

# ==========================================
# Dataset PyTorch
# ==========================================

class FingerprintDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        """
        dataset_dir: Path o str della cartella contenente immagini
        transform: funzione di augmentation
        """
        self.dataset_dir = Path(dataset_dir)
        self.image_paths = sorted(list(self.dataset_dir.glob("**/*.jpg")))
        self.transform = transform if transform else FingerprintAugmentations()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Unable to read image: {img_path}")

        # Genera due versioni augmentate per contrastive learning
        img1 = self.transform(img)
        img2 = self.transform(img)
        return img1, img2

class BaseDataset(Dataset):
    """
    Dataset PyTorch per immagini:
    - Carica immagini da una cartella
    - Applica preprocessing base (resize, normalize, local contrast)
    - Restituisce (img_tensor, path)
    """
    def __init__(self, root_dir: str, extensions: List[str] = [".jpg", ".png"]):
        self.root_dir = Path(root_dir)
        self.img_paths = [p for p in self.root_dir.glob("**/*") if p.suffix.lower() in extensions]
        self.img_paths = sorted(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = self.img_paths[idx]
        img = preprocess_image(img_path)  # preprocessing avanzato

        if img is None:
            # fallback: immagine nera
            img = np.zeros((256, 256), dtype=np.float32)

        # converti a tensor CxHxW
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # grayscale -> 1xHxW
        return img_tensor, str(img_path)