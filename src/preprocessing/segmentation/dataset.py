import os
from torch.utils.data import Dataset
import cv2
import torch
import numpy as np

class FingerprintDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, image_size=(512, 512)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_size = image_size

        # Filtra solo i file che hanno maschera corrispondente
        all_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        self.files = [f for f in all_files if os.path.exists(os.path.join(mask_dir, f))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        # Leggi immagine e maschera (entrambi garantiti esistenti)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # --- Resize ---
        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        # Normalizza e aggiungi canale
        img = torch.tensor(img/255.0, dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask/255.0, dtype=torch.float32).unsqueeze(0)

        return img, mask
