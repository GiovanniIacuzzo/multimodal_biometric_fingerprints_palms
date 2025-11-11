import torch
import os
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------
# Salvataggio e caricamento modello / checkpoint
# ------------------------------------------------
def save_model(obj, path: str):
    """
    Salva un modello o un checkpoint.
    - Se obj è un modello nn.Module, salva state_dict.
    - Se obj è un dict (checkpoint), salva direttamente il dict.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(obj, dict):
        torch.save(obj, path)
        print(f"Checkpoint saved to {path}")
    elif hasattr(obj, "state_dict"):
        torch.save(obj.state_dict(), path)
        print(f"Model state_dict saved to {path}")
    else:
        raise ValueError("save_model: oggetto non supportato. Passa nn.Module o dict.")

def load_model(model, path: str, device='cpu'):
    """
    Carica uno state_dict su un modello già istanziato.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} non trovato.")
    state = torch.load(path, map_location=device)
    # Se è checkpoint dict
    if isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
        print(f"Checkpoint loaded from {path}, epoch={state.get('epoch', 'N/A')}")
    else:
        model.load_state_dict(state)
        print(f"Model state_dict loaded from {path}")
    model.to(device)
    model.eval()
    return model

# ------------------------------------------------
# Visualizzazioni embeddings
# ------------------------------------------------
def plot_embeddings(embeddings, labels=None, title="Embeddings", save_path=None):
    """
    Plot 2D embeddings (t-SNE o PCA ridotte a 2D).
    embeddings: np.array (N, D)
    labels: opzionale, colore per classe
    """
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42)
    emb_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(8,6))
    if labels is not None:
        for lbl in np.unique(labels):
            idx = labels == lbl
            plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], label=str(lbl), alpha=0.6)
        plt.legend()
    else:
        plt.scatter(emb_2d[:,0], emb_2d[:,1], alpha=0.6)

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Plot saved to {save_path}")
    plt.show()

# ------------------------------------------------
# Funzioni generiche di preprocessing su batch
# ------------------------------------------------
def normalize_batch(batch_tensor, eps=1e-8):
    """
    Normalizza batch di tensori [B, D] a media 0, std 1.
    """
    mean = batch_tensor.mean(dim=0, keepdim=True)
    std = batch_tensor.std(dim=0, keepdim=True) + eps
    return (batch_tensor - mean) / std

# ------------------------------------------------
# Funzione per vedere immagini
# ------------------------------------------------
def show_image(img, title=None, cmap='gray'):
    """
    Mostra immagine singola o batch (prende la prima se batch)
    img: np.array o torch.Tensor
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.ndim == 4:  # batch
        img = img[0]
    if img.ndim == 3 and img.shape[0] in [1,3]:  # [C,H,W]
        img = np.transpose(img, (1,2,0))
        if img.shape[2] == 1:
            img = img[:,:,0]
    plt.imshow(img, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
