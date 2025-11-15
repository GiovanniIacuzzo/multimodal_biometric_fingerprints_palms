import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# ------------------------------------------------------------
# Pretrained ViT Encoder
# ------------------------------------------------------------
class PretrainedEncoder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, in_chans=1)
        
        if hasattr(self.vit, 'head'):
            self.vit.head = nn.Identity()
        elif hasattr(self.vit, 'head_dist'):
            self.vit.head_dist = nn.Identity()
        
        patch_size = self.vit.patch_embed.patch_size
        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        else:
            self.patch_size = patch_size
        
        self.embed_dim = self.vit.embed_dim

    def forward(self, x):
        x = self.vit.patch_embed(x)  # [B,C,H_p,W_p] o [B,N,C]
        
        if x.ndim == 4:
            B, C, H_p, W_p = x.shape
            x = x.flatten(2).transpose(1,2)  # [B, N, C]
        else:
            B, N, C = x.shape
            H_p = W_p = int(N ** 0.5)

        # Transformer blocks
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x, (H_p, W_p)

# ------------------------------------------------------------
# Decoder Transformer -> immagine binarizzata
# ------------------------------------------------------------
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, out_ch=1, patch_size=(16,16)):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size  # tuple (H_patch, W_patch)

        # MLP: embed_dim -> patch_size_H * patch_size_W
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_size[0] * patch_size[1])
        )
        self.out_conv = nn.Conv2d(1, out_ch, kernel_size=1)

    def forward(self, x, H_p, W_p, H_in=None, W_in=None):
        B, N, C = x.shape
        H_patch, W_patch = self.patch_size

        patches = self.mlp(x)  # [B, N, H_patch*W_patch]
        patches = patches.view(B, 1, H_p, W_p, H_patch, W_patch)
        patches = patches.permute(0, 1, 4, 2, 5, 3)  # [B,1,H_patch,H_p,W_patch,W_p]
        patches = patches.reshape(B, 1, H_p*H_patch, W_p*W_patch)

        out = self.out_conv(patches)
        if H_in is not None and W_in is not None:
            out = F.interpolate(out, size=(H_in, W_in), mode='bilinear', align_corners=False)
        out = torch.sigmoid(out)  # valori tra 0 e 1
        return out

# ------------------------------------------------------------
# Full model: encoder pre-trained + decoder
# ------------------------------------------------------------
class FingerprintTransUNet(nn.Module):
    def __init__(self, pretrained_model='vit_base_patch16_224', out_ch=1):
        super().__init__()
        self.encoder = PretrainedEncoder(model_name=pretrained_model)
        self.decoder = TransformerDecoder(
            embed_dim=self.encoder.embed_dim,
            out_ch=out_ch,
            patch_size=self.encoder.patch_size
        )

    def forward(self, x):
        B, _, H_in, W_in = x.shape
        x, (H_p, W_p) = self.encoder(x)
        out = self.decoder(x, H_p, W_p, H_in, W_in)
        return out
