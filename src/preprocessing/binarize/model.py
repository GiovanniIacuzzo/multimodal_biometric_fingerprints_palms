import torch
import torch.nn as nn

# ------------------------------------------------------------
# Convolutional block: 2x Conv + BN + ReLU
# ------------------------------------------------------------
def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

# ------------------------------------------------------------
# U-Net per binarizzazione fingerprint
# ------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base_ch=64):
        super().__init__()

        # ---------- Encoder ----------
        self.enc1 = conv_block(in_ch, base_ch)
        self.enc2 = conv_block(base_ch, base_ch*2)
        self.enc3 = conv_block(base_ch*2, base_ch*4)
        self.enc4 = conv_block(base_ch*4, base_ch*8)

        self.pool = nn.MaxPool2d(2)

        # ---------- Bottleneck ----------
        self.bottleneck = conv_block(base_ch*8, base_ch*16)

        # ---------- Decoder ----------
        self.upconv4 = nn.ConvTranspose2d(base_ch*16, base_ch*8, kernel_size=2, stride=2)
        self.dec4 = conv_block(base_ch*16, base_ch*8)

        self.upconv3 = nn.ConvTranspose2d(base_ch*8, base_ch*4, kernel_size=2, stride=2)
        self.dec3 = conv_block(base_ch*8, base_ch*4)

        self.upconv2 = nn.ConvTranspose2d(base_ch*4, base_ch*2, kernel_size=2, stride=2)
        self.dec2 = conv_block(base_ch*4, base_ch*2)

        self.upconv1 = nn.ConvTranspose2d(base_ch*2, base_ch, kernel_size=2, stride=2)
        self.dec1 = conv_block(base_ch*2, base_ch)

        # ---------- Output ----------
        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)
        self.sigmoid = nn.Identity()  # sigmoid applicata solo in inferenza/loss

    def forward(self, x):
        # ----- Encoder -----
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # ----- Bottleneck -----
        b = self.bottleneck(self.pool(e4))

        # ----- Decoder -----
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        return self.sigmoid(out)
