import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class FingerprintDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, image_size=(512, 512)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = tuple(image_size)

        # Filtra solo immagini con maschera corrispondente
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        all_imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_exts)]
        self.files = [f for f in all_imgs if os.path.exists(os.path.join(mask_dir, f))]

        if len(self.files) == 0:
            raise RuntimeError(f"Nessuna immagine valida trovata in {img_dir} con maschere in {mask_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # --- Lettura sicura ---
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Immagine non leggibile: {img_path}")
        if mask is None:
            raise ValueError(f"Maschera non leggibile: {mask_path}")

        # --- Resize coerente ---
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        # --- Normalizzazione robusta ---
        img = np.clip(img.astype(np.float32) / 255.0, 0.0, 1.0)
        mask = np.clip(mask.astype(np.float32) / 255.0, 0.0, 1.0)

        # --- Albumentations o trasformazioni torch ---
        if self.transform is not None:
            if callable(self.transform):
                try:
                    transformed = self.transform(image=img, mask=mask)
                    img, mask = transformed["image"], transformed["mask"]
                except Exception:
                    # Se non è Albumentations, prova a chiamarla come funzione torch
                    img, mask = self.transform(img, mask)
        # --- Conversione in torch tensor ---
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).float().unsqueeze(0)
        else:
            img = img.float()

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float().unsqueeze(0)
        else:
            mask = mask.float()

        # --- Verifica integrità ---
        if torch.isnan(img).any() or torch.isnan(mask).any():
            raise ValueError(f"NaN trovato in immagine o maschera: {img_name}")

        if mask.max() > 1.0 or mask.min() < 0.0:
            print(f"[WARN] Maschera non normalizzata: {mask.min().item()}–{mask.max().item()} per {mask_path}")
            mask = mask.clamp(0, 1)

        return img, mask
