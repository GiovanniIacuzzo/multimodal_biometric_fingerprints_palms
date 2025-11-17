import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class FingerprintDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, image_size=(512, 512)):
        """
        img_dir, mask_dir: stringa (cartella) o lista di percorsi file
        transform: funzione Albumentations o callable
        image_size: tuple (h, w)
        """
        self.transform = transform
        self.image_size = tuple(image_size)

        # --- Gestione tipi ---
        if isinstance(img_dir, (list, tuple)) and isinstance(mask_dir, (list, tuple)):
            self.img_paths = [str(p) for p in img_dir]
            self.mask_paths = [str(p) for p in mask_dir]
        elif isinstance(img_dir, (str, Path)) and isinstance(mask_dir, (str, Path)):
            # vecchio comportamento: cartelle singole
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
            all_imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]
            self.img_paths = [os.path.join(img_dir, f) for f in all_imgs if os.path.exists(os.path.join(mask_dir, f))]
            self.mask_paths = [os.path.join(mask_dir, f) for f in all_imgs if os.path.exists(os.path.join(mask_dir, f))]
        else:
            raise ValueError("img_dir e mask_dir devono essere entrambi liste o entrambi stringhe di cartella")

        if len(self.img_paths) == 0:
            raise RuntimeError(f"Nessuna immagine valida trovata in {img_dir} con maschere in {mask_dir}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # --- Lettura immagine RGB e maschera in scala di grigi ---
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)        # legge BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)         # converte in RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # maschera 1 canale

        if img is None:
            raise ValueError(f"Immagine non leggibile: {img_path}")
        if mask is None:
            raise ValueError(f"Maschera non leggibile: {mask_path}")

        # --- Resize coerente ---
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        # --- Normalizzazione ---
        img = np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)  # RGB 0-1
        mask = np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0) # 0-1

        # --- Trasformazioni Albumentations o callable ---
        if self.transform is not None:
            if callable(self.transform):
                try:
                    transformed = self.transform(image=img, mask=mask)
                    img, mask = transformed["image"], transformed["mask"]
                except Exception:
                    img, mask = self.transform(img, mask)

        # --- Assicurati che la maschera abbia canale ---
        if isinstance(mask, torch.Tensor) and mask.ndim == 2:
            mask = mask.unsqueeze(0)  # [H,W] -> [1,H,W]
        elif isinstance(mask, np.ndarray) and mask.ndim == 2:
            mask = torch.from_numpy(mask).float().unsqueeze(0)

        # --- Conversione immagine in tensor torch ---
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img.transpose(2,0,1)).float()  # [H,W,3] -> [3,H,W]
        else:
            img = img.float()

        if not isinstance(mask, torch.Tensor):
            mask = mask.float()

        # --- Check integrità ---
        if torch.isnan(img).any() or torch.isnan(mask).any():
            raise ValueError(f"NaN trovato in immagine o maschera: {img_path}")
        if mask.max() > 1.0 or mask.min() < 0.0:
            mask = mask.clamp(0,1)

        return img, mask
