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
    Ogni immagine ha nome nel formato: <id>_<session>_<index>.jpg
    (es: 117_2_5.jpg → soggetto 117)
    Durante il training, genera coppie (anchor, positive, negative) dinamiche.
    """
    def __init__(self, root_dir, transform=None, augment=True):
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
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
        """Crea lista [(id_soggetto, path_img), ...] basata sul nome file."""
        samples = []
        for file in sorted(os.listdir(self.root_dir)):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                subject_id = file.split("_")[0]  # prende il primo numero come ID
                path = os.path.join(self.root_dir, file)
                samples.append((subject_id, path))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Ritorna una tupla (anchor, positive, negative)."""
        subject, anchor_path = self.samples[idx]
        anchor_img = Image.open(anchor_path).convert("L")

        # Positiva dello stesso soggetto
        same_subject_imgs = [p for s, p in self.samples if s == subject and p != anchor_path]
        positive_path = same_subject_imgs[0] if same_subject_imgs else anchor_path
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
