# src/models/descriptors_deep.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# -----------------------
# Optional import config
# -----------------------
try:
    from scripts.config import DATASET_DIR as CONFIG_DATASET_DIR, MODELS_DIR as CONFIG_MODELS_DIR, BATCH_SIZE as CONFIG_BATCH_SIZE
except Exception:
    CONFIG_DATASET_DIR = None
    CONFIG_MODELS_DIR = None
    CONFIG_BATCH_SIZE = None

# ==============================
# 1. DATASET
# ==============================
class FingerprintDataset(Dataset):
    """
    Dataset per immagini fingerprint in cartelle strutturate.
    Usa un'immagine specifica per ogni soggetto, es. 'enhanced.png'.
    Tutte le immagini vengono convertite in 3 canali, resized e normalizzate.
    """
    def __init__(self, root_dir, transform=None, img_name="enhanced.png"):
        self.paths = []
        self.transform = transform
        self.img_name = img_name
        self.root_dir = root_dir

        # Scansione delle sottocartelle
        if not os.path.isdir(root_dir):
            raise ValueError(f"root_dir non valido: {root_dir}")
        for subfolder in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, subfolder)
            if os.path.isdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                if os.path.exists(img_path):
                    self.paths.append(img_path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = Image.open(img_path).convert("L")  # grayscale

        if self.transform:
            img = self.transform(img)  # potrebbe diventare 3xHxW se Grayscale(num_output_channels=3)
        else:
            # trasformazione di default: 3 canali + 224x224 + normalize
            transform_default = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
            ])
            img = transform_default(img)

        return img

# ==============================
# 2. FEATURE EXTRACTOR
# ==============================
class FingerprintFeatureExtractor(nn.Module):
    def __init__(self, embedding_dim=256, pretrained=True):
        super().__init__()
        # compatibilità con diverse versioni torchvision
        try:
            base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        except Exception:
            base = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(base.children())[:-1])  # rimuove FC finale
        self.fc = nn.Linear(base.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.features(x)        # (B, 512, 1, 1)
        x = torch.flatten(x, 1)     # (B, 512)
        x = self.fc(x)              # (B, embedding_dim)
        x = F.normalize(x, dim=1)   # L2 normalize
        return x

# ==============================
# 3. SIAMESE / HELPER
# ==============================
class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=256, pretrained=True):
        super().__init__()
        self.encoder = FingerprintFeatureExtractor(embedding_dim, pretrained=pretrained)

    def forward(self, x1, x2):
        emb1 = self.encoder(x1)
        emb2 = self.encoder(x2)
        return emb1, emb2

    def compute_distance(self, x1, x2):
        emb1, emb2 = self.forward(x1, x2)
        return torch.norm(emb1 - emb2, dim=1)

# ==============================
# 4. TRAINING MAIN
# ==============================
def main(
    dataset_dir: str = None,
    save_path: str = None,
    epochs: int = 5,
    batch_size: int = 16,
    embedding_dim: int = 256,
    lr: float = 1e-4,
    device: str = None,
    pretrained_backbone: bool = True
):
    """
    Funzione main per eseguire il training del feature extractor.
    Nota: il training triplet qui è un semplice placeholder che crea triplet
    dall'interno dello stesso batch (anchor==positive, negative=flipped).
    """
    import time

    # fallback sui config (se esiste src.config)
    if dataset_dir is None:
        dataset_dir = CONFIG_DATASET_DIR or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
    if save_path is None:
        models_dir = CONFIG_MODELS_DIR or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        os.makedirs(models_dir, exist_ok=True)
        save_path = os.path.join(models_dir, "fingerprint_deep.pt")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"📁 Dataset dir: {dataset_dir}")
    print(f"💾 Save model to: {save_path}")
    print(f"🖥 Device: {device}")
    print(f"📦 Batch size: {batch_size}, Epochs: {epochs}, LR: {lr}")

    # transforms
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    # dataset + loader
    dataset = FingerprintDataset(dataset_dir, transform=train_transform, img_name="enhanced.png")
    print(f"ℹ️ Immagini trovate nel dataset: {len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError(f"Nessuna immagine trovata in {dataset_dir}. Controlla la struttura delle cartelle.")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
    print(f"ℹ️ Numero batch: {len(loader)}")

    # model, loss, optimizer
    model = FingerprintFeatureExtractor(embedding_dim=embedding_dim, pretrained=pretrained_backbone).to(device)
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("🚀 Inizio training...")
    start_time = time.time()

    # training loop (semplice)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for i, batch in enumerate(loader):
            # batch: tensor (B, C, H, W)
            anchor = batch.to(device)
            positive = batch.to(device)             # placeholder: usare positive reali
            negative = batch.flip(0).to(device)     # placeholder: negative diverso (shuffled)

            emb_a = model(anchor)
            emb_p = model(positive)
            emb_n = model(negative)

            loss = criterion(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            print(f"    ⚡ Batch {i+1}/{len(loader)} - Loss: {loss.item():.6f}")

        avg_loss = total_loss / max(1, n_batches)
        print(f"✅ Epoch [{epoch+1}/{epochs}] completata - Loss media: {avg_loss:.6f}")

    elapsed = time.time() - start_time
    print(f"⏱ Tempo totale training: {elapsed:.1f} sec")

    # salva solo lo stato del modello
    torch.save(model.state_dict(), save_path)
    print(f"💾 Modello salvato in {save_path}")

    return save_path

# Permette di eseguire il file direttamente con parametri
if __name__ == "__main__":
    # defaults utili per esecuzione diretta
    DEFAULT_DATASET = CONFIG_DATASET_DIR or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
    DEFAULT_MODELS_DIR = CONFIG_MODELS_DIR or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)
    DEFAULT_SAVE = os.path.join(DEFAULT_MODELS_DIR, "fingerprint_deep.pt")

    main(dataset_dir=DEFAULT_DATASET, save_path=DEFAULT_SAVE)
