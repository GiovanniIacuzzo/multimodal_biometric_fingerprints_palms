import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import numpy as np

# ---------------------------
# Utils
# ---------------------------
def extract_id(fname: str):
    """ Estrae ID dal file _segmented.png """
    base = os.path.splitext(fname)[0]
    if base.endswith("_segmented"):
        base = base.replace("_segmented", "")
    return base

def binarize_pil(img: Image.Image, threshold=128):
    """ Binarizzazione forte 0/255 """
    return img.point(lambda p: 255 if p > threshold else 0)

# ---------------------------
# Dataset
# ---------------------------
class FingerprintBinaryDataset(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        val_split=0.15,
        subset_size=-1,
        subset_seed=42,
        img_size=(256, 256),
        augment=False,
    ):
        self.img_size = img_size
        self.augment = augment

        # --- Cartelle ---
        input_dir = os.path.join(root_dir, "segmentation")
        target_dir = os.path.join(root_dir, "debug/binary")

        # print(f"[INFO] Input dir: {input_dir}")
        # print(f"[INFO] Target dir: {target_dir}")

        input_files = [f for f in sorted(os.listdir(input_dir)) if f.endswith("_segmented.png")]
        target_files = [f for f in sorted(os.listdir(target_dir)) if f.lower().endswith((".png", ".tif", ".tiff", ".jpg"))]

        # print(f"[INFO] Trovati {len(input_files)} input (_segmented.png)")
        # print(f"[INFO] Trovati {len(target_files)} target (png/tif/tiff)")

        # --- Mapping ID → target file ---
        target_map = {os.path.splitext(t)[0]: t for t in target_files}

        matched_inputs, matched_targets = [], []
        missing_targets = []

        # --- Matching input → target ---
        for f in input_files:
            fid = extract_id(f)
            if fid in target_map:
                matched_inputs.append(f)
                matched_targets.append(target_map[fid])
            else:
                missing_targets.append(fid)

        if missing_targets:
            print(f"[WARNING] {len(missing_targets)} input non hanno target corrispondente. Esempi: {missing_targets[:10]}")

        if len(matched_inputs) == 0:
            raise RuntimeError("❌ Nessuna coppia input-target trovata. Controlla naming o directory!")

        self.inputs = matched_inputs
        self.targets = matched_targets
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.to_tensor = T.ToTensor()

        # --- Subset opzionale ---
        if subset_size > 0:
            random.seed(subset_seed)
            idx = list(range(len(self.inputs)))
            random.shuffle(idx)
            idx = idx[:subset_size]

            self.inputs = [self.inputs[i] for i in idx]
            self.targets = [self.targets[i] for i in idx]
            # print(f"[INFO] Subset selezionato: {len(self.inputs)} coppie")

        # --- Split train/val ---
        total = len(self.inputs)
        n_val = int(total * val_split)

        if split == "train":
            sel = slice(n_val, total)
        else:
            sel = slice(0, n_val)

        self.inputs = self.inputs[sel]
        self.targets = self.targets[sel]

        # print(f"[INFO] Dataset finale ({split}): {len(self.inputs)} esempi")

        # --- Augmentation ---
        self.train_aug = T.Compose([
            T.RandomApply([T.GaussianBlur(5)], p=0.25),
            T.RandomAutocontrast(p=0.25),
            T.RandomAdjustSharpness(2, p=0.25),
        ]) if augment else None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        img_in_path = os.path.join(self.input_dir, self.inputs[idx])
        img_gt_path = os.path.join(self.target_dir, self.targets[idx])

        # --- Caricamento immagini ---
        img_in = Image.open(img_in_path).convert("L")
        img_gt = Image.open(img_gt_path).convert("L")

        # --- Resize ---
        img_in = img_in.resize(self.img_size, resample=Image.BILINEAR)
        img_gt = img_gt.resize(self.img_size, resample=Image.NEAREST)

        # --- Ground truth binaria ---
        img_gt = binarize_pil(img_gt, threshold=128)

        # --- Augmentation ---
        if self.augment and self.train_aug is not None:
            img_in = self.train_aug(img_in)

        # --- Conversione a tensori ---
        img_in = self.to_tensor(img_in)
        img_gt = self.to_tensor(img_gt)

        return img_in, img_gt

# ---------------------------
# __main__ per debug
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="dataset/processed/")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--subset", type=int, default=-1)
    parser.add_argument("--img_size", type=int, nargs=2, default=[256,256])
    args = parser.parse_args()

    ds = FingerprintBinaryDataset(
        root_dir=args.root,
        split=args.split,
        subset_size=args.subset,
        img_size=tuple(args.img_size),
        augment=False
    )

    print(f"[TEST] Lunghezza dataset: {len(ds)}")

    # Stampa primi 5 ID e shape immagini
    for i in range(min(5, len(ds))):
        img, gt = ds[i]
        print(f"[TEST] Sample {i}: img shape={img.shape}, gt shape={gt.shape}, min={gt.min().item()}, max={gt.max().item()}")
