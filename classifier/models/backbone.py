import torch
import torch.nn as nn
import timm

class FingerprintViTBackbone(nn.Module):
    def __init__(self,
                 model_name="vit_base_patch16_224",
                 pretrained=True,
                 checkpoint_path=None,
                 embedding_dim=512,
                 freeze_backbone=False,
                 norm_layer=True,
                 dropout=0.2,
                 patch_size=16):
        super().__init__()
        self.output_dim = embedding_dim

        # Carica il modello timm senza head
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='')
        
        # adattamento per 1 canale se necessario
        try:
            pe = self.backbone.patch_embed
            if hasattr(pe, "proj") and pe.proj.in_channels != 1:
                w = pe.proj.weight
                new_conv = nn.Conv2d(1, w.shape[0], kernel_size=w.shape[2:], stride=pe.proj.stride, padding=pe.proj.padding, bias=False)
                # inizializza con la media dei canali rgb
                new_conv.weight.data.copy_(w.mean(dim=1, keepdim=True))
                pe.proj = new_conv
        except Exception:
            pass

        # Se ti è dato un checkpoint specifico
        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            self.backbone.load_state_dict(ckpt, strict=False)
            print(f"Loaded fingerprint‐specific checkpoint from {checkpoint_path}")

        self.norm = nn.LayerNorm(embedding_dim) if norm_layer else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.projector = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)  # shape (B, T, D)
        pooled = feats[:, 0]  # CLS token
        if self.projector is None:
            self.projector = nn.Linear(pooled.shape[1], self.output_dim).to(pooled.device)
        emb = self.projector(pooled)
        emb = self.norm(emb)
        emb = self.dropout(emb)
        return emb
