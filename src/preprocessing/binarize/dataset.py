import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import random


def extract_id(fname: str):
    """ Estrae l'ID dal file _segmented.png """
    base = os.path.splitext(fname)[0]   # togli estensione
    if base.endswith("_segmented"):
        base = base.replace("_segmented", "")
    return base


class FingerprintBinaryDataset(Dataset):
    def __init__(self, root_dir, split="train", val_split=0.15,
                 transform=None, subset_size=-1, subset_seed=42, img_size=(256, 256)):

        # print("\n===== FingerprintBinaryDataset =====")
        self.img_size = img_size

        input_dir = os.path.join(root_dir, "segmentation")
        target_dir = os.path.join(root_dir, "debug/binary")

        input_files = [f for f in sorted(os.listdir(input_dir)) if f.endswith("_segmented.png")]
        target_files = [f for f in sorted(os.listdir(target_dir)) if f.endswith(".jpg")]

        # print(f"[INFO] Trovati {len(input_files)} input (_segmented.png)")
        # print(f"[INFO] Trovati {len(target_files)} target (.jpg)")

        # --- crea mapping degli ID dai target ---
        target_map = {os.path.splitext(t)[0]: t for t in target_files}

        matched_inputs = []
        matched_targets = []

        # print("[INFO] Matching files...")
        for f in input_files:
            fid = extract_id(f)
            if fid in target_map:
                matched_inputs.append(f)
                matched_targets.append(target_map[fid])

        # print(f"[INFO] Match trovati: {len(matched_inputs)}")

        if len(matched_inputs) == 0:
            raise RuntimeError("❌ Nessuna coppia input-target trovata. Check naming!")

        # salva liste
        self.inputs = matched_inputs
        self.targets = matched_targets
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.to_tensor = T.ToTensor()

        # --- subset opzionale ---
        if subset_size > 0:
            # print(f"[INFO] Subset: {subset_size}")
            random.seed(subset_seed)
            idx = list(range(len(self.inputs)))
            random.shuffle(idx)
            idx = idx[:subset_size]

            self.inputs = [self.inputs[i] for i in idx]
            self.targets = [self.targets[i] for i in idx]

        # --- split train/val ---
        total = len(self.inputs)
        n_val = int(total * val_split)

        if split == "train":
            sel = slice(n_val, total)
        else:
            sel = slice(0, n_val)

        self.inputs = self.inputs[sel]
        self.targets = self.targets[sel]

        # print(f"[INFO] Dataset finale ({split}): {len(self.inputs)} esempi")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        img_in = Image.open(os.path.join(self.input_dir, self.inputs[idx])).convert("L")
        img_gt = Image.open(os.path.join(self.target_dir, self.targets[idx])).convert("L")

        img_in = img_in.resize(self.img_size, resample=Image.BILINEAR)
        img_gt = img_gt.resize(self.img_size, resample=Image.NEAREST)

        img_in = self.to_tensor(img_in)
        img_gt = self.to_tensor(img_gt)

        return img_in, img_gt


# ============================================================
# MAIN TEST
# ============================================================
if __name__ == "__main__":
    root = "dataset/processed"

    ds_train = FingerprintBinaryDataset(
        root_dir=root,
        split="train",
        val_split=0.15,
        subset_size=200,
    )

    print(f"\n[RESULT] Lunghezza train: {len(ds_train)}")

    if len(ds_train) > 0:
        img, mask = ds_train[0]
        print(f"[CHECK] Immagine shape: {img.shape}")
        print(f"[CHECK] Mask shape:     {mask.shape}")
