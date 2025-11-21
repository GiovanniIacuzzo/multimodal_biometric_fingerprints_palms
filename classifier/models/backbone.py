import torch
import torch.nn as nn
import timm

class FingerprintViTBackbone(nn.Module):
    def __init__(self,
                 model_name="convnextv2_base.fcmae_ft_in22k_in1k",
                 pretrained=True,
                 checkpoint_path=None,
                 embedding_dim=256,
                 freeze_backbone=False,
                 use_l2norm=True):
        super().__init__()

        # ----------------------------------------------------
        # BACKBONE
        # ----------------------------------------------------
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=""      # niente pool finale
        )

        # ----------------------------------------------------
        # FIX INPUT CHANNELS (1 → RGB) SE MODELLO CNN
        # ----------------------------------------------------
        if hasattr(self.backbone, "conv_stem"):  # EfficientNet / ConvNeXt style
            conv = self.backbone.conv_stem
            if conv.weight.shape[1] != 1:
                new_conv = nn.Conv2d(
                    1, conv.out_channels,
                    kernel_size=conv.kernel_size,
                    stride=conv.stride,
                    padding=conv.padding,
                    bias=conv.bias is not None
                )
                # Copia pesi mediati
                new_conv.weight.data = conv.weight.data.mean(dim=1, keepdim=True)
                if conv.bias is not None:
                    new_conv.bias.data = conv.bias.data.clone()
                self.backbone.conv_stem = new_conv

        # ----------------------------------------------------
        # FIX INPUT CHANNELS (1 → RGB) SE MODELLO VIT / SWIN
        # ----------------------------------------------------
        if hasattr(self.backbone, "patch_embed"):
            pe = self.backbone.patch_embed
            if hasattr(pe, "proj"):
                proj = pe.proj
                if proj.in_channels != 1:
                    new_proj = nn.Conv2d(
                        1, proj.out_channels,
                        kernel_size=proj.kernel_size,
                        stride=proj.stride,
                        padding=proj.padding,
                        bias=False
                    )
                    new_proj.weight.data = proj.weight.data.mean(dim=1, keepdim=True)
                    pe.proj = new_proj

        # ----------------------------------------------------
        # INFER FEATURE DIM
        # ----------------------------------------------------
        test = torch.zeros(1, 1, 224, 224)
        with torch.no_grad():
            tmp = self.backbone.forward_features(test)
            if tmp.ndim == 4:      # CNN → B C H W
                feat_dim = tmp.shape[1]
            else:                   # Transformer → B N C
                feat_dim = tmp.shape[-1]

        # ----------------------------------------------------
        # PROIETTORE LINEARE
        # ----------------------------------------------------
        self.projector = nn.Linear(feat_dim, embedding_dim)
        self.use_l2norm = use_l2norm

        # ----------------------------------------------------
        # FREEZE BACKBONE
        # ----------------------------------------------------
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # ----------------------------------------------------
        # LOAD CHECKPOINT
        # ----------------------------------------------------
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            self.load_state_dict(ckpt, strict=False)
            print("Loaded checkpoint:", checkpoint_path)

    def forward(self, x):
        feats = self.backbone.forward_features(x)

        if feats.ndim == 3:   # Transformer
            pooled = feats[:, 0, :]  # cls token
        else:                  # CNN
            pooled = feats.mean(dim=[2, 3])

        emb = self.projector(pooled)

        if self.use_l2norm:
            emb = nn.functional.normalize(emb, p=2, dim=1)

        return emb
