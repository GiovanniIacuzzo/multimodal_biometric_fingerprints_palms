import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================
# BLOCCO CONV BASE
# ===========================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# ===========================================
# U-NET
# ===========================================
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        # Encoder
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev_ch = in_ch
        for f in features:
            self.downs.append(ConvBlock(prev_ch, f))
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = f

        # Bottleneck
        self.bottleneck = ConvBlock(features[-1], features[-1]*2)

        # Decoder
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        rev_features = list(reversed(features))
        prev_ch = features[-1]*2
        for f in rev_features:
            self.ups.append(nn.ConvTranspose2d(prev_ch, f, kernel_size=2, stride=2))
            self.up_convs.append(ConvBlock(prev_ch, f))
            prev_ch = f

        # Output
        self.final_conv = nn.Conv2d(prev_ch, out_ch, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for up, conv, skip in zip(self.ups, self.up_convs, skip_connections):
            x = up(x)
            # A volte serve tagliare dimensione per match
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = conv(x)

        return torch.sigmoid(self.final_conv(x))  # output 0..1 per binary skeleton
