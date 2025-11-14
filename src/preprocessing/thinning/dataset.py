import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from config import config_fingerprint


class FingerprintThinningDataset(Dataset):
    def __init__(self,
                 binary_dir: Optional[str] = None,
                 thinning_dir: Optional[str] = None,
                 img_size: Tuple[int, int] = (256, 256),
                 augment: bool = False):
        super().__init__()

        # Usa percorsi da config se non specificati
        self.binary_dir = Path(binary_dir or config_fingerprint.PROCESSED_DIR)
        self.thinning_dir = Path(thinning_dir or config_fingerprint.PROCESSED_DIR)
        self.img_size = img_size
        self.augment = augment

        if not self.binary_dir.exists():
            raise FileNotFoundError(f"Cartella binary non trovata: {self.binary_dir}")
        if not self.thinning_dir.exists():
            raise FileNotFoundError(f"Cartella skeleton non trovata: {self.thinning_dir}")

        # Prendi tutti i file binari
        self.files = sorted([f for f in self.binary_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]])
        if len(self.files) == 0:
            raise RuntimeError(f"Nessuna immagine trovata in {self.binary_dir}")

        # Trasformazioni di base
        self.transform_input = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        self.transform_target = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

        # Augmentazioni leggere
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        bin_path = self.files[idx]
        thinning_path = self.thinning_dir / bin_path.name

        if not thinning_path.exists():
            # logga e ritorna immagine vuota o raise a seconda delle necessità
            raise FileNotFoundError(f"Target thinning non trovato: {thinning_path}")

        # --- carica immagini ---
        img_bin = Image.open(bin_path).convert("L")
        img_thinning = Image.open(thinning_path).convert("L")

        # --- augmentazioni sincronizzate ---
        if self.augment:
            seed = np.random.randint(0, 99999)
            torch.manual_seed(seed)
            img_bin = self.augment_transform(img_bin)
            torch.manual_seed(seed)
            img_thinning = self.augment_transform(img_thinning)

        # --- trasformazioni finali ---
        img_bin = self.transform_input(img_bin)
        img_thinning = self.transform_target(img_thinning)

        # binarizza target (0 o 1)
        img_thinning = (img_thinning > 0.5).float()

        return {
            "input": img_bin,
            "target": img_thinning
        }


if __name__ == "__main__":
    dataset = FingerprintThinningDataset(
        img_size=(256, 256),
        augment=True
    )
    print(f"Numero immagini: {len(dataset)}")
    sample = dataset[0]
    print(f"Input shape: {sample['input'].shape}, Target shape: {sample['target'].shape}")
