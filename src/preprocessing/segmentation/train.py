import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torchvision.transforms as T

from src.preprocessing.segmentation.dataset import FingerprintDataset
from src.preprocessing.segmentation.model import FingerprintSegmentationModel

def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return 1 - (2 * intersection + eps) / (union + eps)

def train_segmentation(
    img_dir,
    mask_dir,
    image_size=(224,224),
    epochs=20,
    batch_size=8,
    lr=5e-5,
    device="cuda"
):
    device = torch.device(device if torch.cuda.is_available() or device=="cpu" else "mps")
    print(f"Training on device: {device}")

    # Data augmentation base
    transform = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.RandomRotation(5),
    ])

    dataset = FingerprintDataset(img_dir, mask_dir, image_size=image_size, transform=None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FingerprintSegmentationModel(
        num_labels=1,
        image_size=(224,224),
        pretrained_model="nvidia/segformer-b2-finetuned-ade-512-512"
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    bce_loss = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, masks in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            if imgs.shape[1] == 1:
                imgs = imgs.repeat(1,3,1,1)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = bce_loss(outputs, masks) + dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

    torch.save(model, "src/preprocessing/segmentation/model/segmentation.pth")
    print("Model saved: segmentation.pth")

if __name__ == "__main__":
    train_segmentation(
        img_dir="dataset/DBII",
        mask_dir="dataset/masks",
        image_size=(224,224),
        epochs=6,
        batch_size=8,
        lr=5e-5,
        device="mps"
    )
