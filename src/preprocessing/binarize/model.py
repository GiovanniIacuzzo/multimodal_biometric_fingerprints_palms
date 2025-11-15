import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Block conv → normalization → activation
# GroupNorm funziona molto meglio per batch piccoli (<8)
# ------------------------------------------------------------
def conv_block(in_ch, out_ch, dropout=0.0):
    layers = [
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.GroupNorm(num_groups=8, num_channels=out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))

    layers.extend([
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(num_groups=8, num_channels=out_ch),
        nn.LeakyReLU(0.1, inplace=True),
    ])
    return nn.Sequential(*layers)


# ------------------------------------------------------------
# Upsample + conv invece di ConvTranspose (più stabile)
# ------------------------------------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = conv_block(in_ch, out_ch, dropout=dropout)

    def forward(self, x, skip):
        x = self.up(x)
        # padding per sicurezza
        if x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2):
            x = F.pad(x, (0, skip.size(-1) - x.size(-1),
                          0, skip.size(-2) - x.size(-2)))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ------------------------------------------------------------
# U-Net profonda e stabile per fingerprint binarization
# ------------------------------------------------------------
class UNetFingerprint(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base=64):
        super().__init__()

        # Encoder
        self.enc1 = conv_block(in_channels, base)          # 64
        self.enc2 = conv_block(base, base * 2)             # 128
        self.enc3 = conv_block(base * 2, base * 4)         # 256
        self.enc4 = conv_block(base * 4, base * 8)         # 512

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = conv_block(base * 8, base * 16, dropout=0.2)  # 1024

        # Decoder
        self.dec4 = UpBlock(base * 16 + base * 8, base * 8, dropout=0.1)
        self.dec3 = UpBlock(base * 8 + base * 4, base * 4, dropout=0.1)
        self.dec2 = UpBlock(base * 4 + base * 2, base * 2)
        self.dec1 = UpBlock(base * 2 + base, base)

        # Output layer
        self.final = nn.Conv2d(base, out_channels, 1)

    def forward(self, x):
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        # Bottleneck
        b = self.bottleneck(self.pool(s4))

        # Decoder
        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        # Logits, NO sigmoid!
        return self.final(d1)
