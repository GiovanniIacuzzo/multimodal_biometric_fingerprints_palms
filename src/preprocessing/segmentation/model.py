import torch
import torch.nn as nn

# ---------------------------------------------------------
# Basic conv block
# ---------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

# ---------------------------------------------------------
# U-Net++ Nested Block
# ---------------------------------------------------------
class NestedUNet(nn.Module):
    def __init__(self, num_labels=1, input_channels=3, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision

        filters = [64, 128, 256, 512, 1024]

        # Down
        self.conv0_0 = ConvBlock(input_channels, filters[0])
        self.pool0 = nn.MaxPool2d(2)

        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.pool1 = nn.MaxPool2d(2)

        self.conv2_0 = ConvBlock(filters[1], filters[2])
        self.pool2 = nn.MaxPool2d(2)

        self.conv3_0 = ConvBlock(filters[2], filters[3])
        self.pool3 = nn.MaxPool2d(2)

        self.conv4_0 = ConvBlock(filters[3], filters[4])

        # Up (nested)
        self.up1_0 = ConvBlock(filters[0] + filters[1], filters[0])
        self.up2_0 = ConvBlock(filters[1] + filters[2], filters[1])
        self.up3_0 = ConvBlock(filters[2] + filters[3], filters[2])

        self.up1_1 = ConvBlock(filters[0]*2 + filters[1], filters[0])
        self.up2_1 = ConvBlock(filters[1]*2 + filters[2], filters[1])

        self.up1_2 = ConvBlock(filters[0]*3 + filters[1], filters[0])

        # Up-sampling
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # Final
        self.final = nn.Conv2d(filters[0], num_labels, kernel_size=1)

    def forward(self, x):
        # encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool0(x0_0))
        x2_0 = self.conv2_0(self.pool1(x1_0))
        x3_0 = self.conv3_0(self.pool2(x2_0))
        x4_0 = self.conv4_0(self.pool3(x3_0))

        # decoder nested
        x0_1 = self.up1_0(torch.cat([x0_0, self.up(x1_0)], 1))
        x1_1 = self.up2_0(torch.cat([x1_0, self.up(x2_0)], 1))
        x2_1 = self.up3_0(torch.cat([x2_0, self.up(x3_0)], 1))

        x0_2 = self.up1_1(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        x1_2 = self.up2_1(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))

        x0_3 = self.up1_2(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        out = self.final(x0_3)
        return out


# ---------------------------------------------------------
# Wrapper per compatibilità col tuo training loop
# ---------------------------------------------------------
class FingerprintSegmentationModel(nn.Module):
    def __init__(self, num_labels=1, image_size=(224,224), pretrained_model=None):
        """
        pretrained_model ignorato -> così il config rimane compatibile.
        """
        super().__init__()
        self.image_size = image_size
        self.model = NestedUNet(num_labels=num_labels, input_channels=3)

    def forward(self, x):
        return self.model(x)
