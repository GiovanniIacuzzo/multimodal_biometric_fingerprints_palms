import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# ================================================
# DATASET
# ================================================
class FingerprintDataset(Dataset):
    """
    Dataset per immagini di impronte digitali.
    Ogni sottocartella rappresenta un soggetto, contenente una o più immagini.
    Durante il training, genera (anchor, positive, negative) dinamici.
    """
    def __init__(self, root_dir, img_name="enhanced.png", transform=None, augment=True):
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.img_name = img_name
        self.samples = self._scan_dataset()

        if len(self.samples) == 0:
            raise RuntimeError(f"Nessuna immagine trovata in {root_dir}")

        # Data augmentation opzionale
        self.augmentation = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomHorizontalFlip(),
        ]) if augment else None

    def _scan_dataset(self):
        """Crea lista [(classe, path_img), ...]"""
        samples = []
        for subject in sorted(os.listdir(self.root_dir)):
            subject_path = os.path.join(self.root_dir, subject)
            if os.path.isdir(subject_path):
                for file in os.listdir(subject_path):
                    if file.endswith((".png", ".jpg", ".jpeg")):
                        samples.append((subject, os.path.join(subject_path, file)))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Ritorna una tupla (anchor, positive, negative)."""
        subject, anchor_path = self.samples[idx]
        anchor_img = Image.open(anchor_path).convert("L")

        # Trova una positiva dello stesso soggetto
        same_subject_imgs = [p for s, p in self.samples if s == subject and p != anchor_path]
        if same_subject_imgs:
            positive_path = same_subject_imgs[0]
        else:
            positive_path = anchor_path  # fallback se solo un'immagine
        positive_img = Image.open(positive_path).convert("L")

        # Negativa da un altro soggetto
        neg_subject, neg_path = next((s, p) for s, p in self.samples if s != subject)
        negative_img = Image.open(neg_path).convert("L")

        # Trasformazioni
        transform = self.transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

        def apply(img):
            if self.augment and self.augmentation:
                img = self.augmentation(img)
            return transform(img)

        return apply(anchor_img), apply(positive_img), apply(negative_img)