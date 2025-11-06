import os
import sys
import torch
import multiprocessing as mp

if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.descriptors_deep import train_deep_descriptor


if __name__ == "__main__":
    # Imposta parametri direttamente qui
    dataset_dir = "dataset/DBII"
    save_path = "data/model/deep_descriptor.pth"
    epochs = 10
    batch_size = 16
    embedding_dim = 256
    lr = 1e-4

    # Seleziona device automaticamente
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Avvio training deep model su device: {device}")

    # Nota: use_amp solo su CUDA, mai su MPS o CPU
    model = train_deep_descriptor(
        dataset_dir=dataset_dir,
        save_path=save_path,
        epochs=epochs,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
        lr=lr,
        device=device,
        pretrained=True,
        use_amp=device
    )

    print(f"Training completato. Modello salvato in: {save_path}")
