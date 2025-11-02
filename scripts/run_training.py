# src/scripts/run_training.py
import os
import argparse
from pathlib import Path

# sistema import path in caso venga eseguito dalla root
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.descriptors_deep import main as train_main

def parse_args():
    p = argparse.ArgumentParser(description="Train fingerprint deep descriptor")
    p.add_argument("--dataset_dir", type=str, default=None, help="Directory con immagini processed (cartelle per soggetto)")
    p.add_argument("--save_path", type=str, default=None, help="Dove salvare il modello (file .pt)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default=None, help="cuda / cpu or None to auto")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # fallback: se non fornito, train_main userà src.config (se esiste) o i default
    saved = train_main(
        dataset_dir=args.dataset_dir,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        lr=args.lr,
        device=args.device
    )
    print(f"Training finito. Modello salvato in: {saved}")
