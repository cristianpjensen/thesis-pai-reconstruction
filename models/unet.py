"""Original U-net implementation in (Ronneberger et al., 2015)."""

import torch
import torchvision
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.initial_conv = DoubleConv2d(1, 64)
        self.down1 = Downsample(64, 128)
        self.down2 = Downsample(128, 256)
        self.down3 = Downsample(256, 512)
        self.down4 = Downsample(512, 1024)

        self.up1 = Upsample(1024, 512)
        self.up2 = Upsample(512, 256)
        self.up3 = Upsample(256, 128)
        self.up4 = Upsample(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x1 = self.initial_conv(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.out(x)


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Downsample(nn.Module):
    """First downsample, then double convolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2d(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Upsample(nn.Module):
    """First, upsample, then double convolution.

    In the `forward` function, the first argument must be the layer being
    upsampled and the second argument must be the skip-connection layer.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.double_conv = DoubleConv2d(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        x_diff = x2.shape[1] - x1.shape[1]
        y_diff = x2.shape[2] - x1.shape[2]

        # Crop skip-connection image and concatenate
        x2_cropped = torchvision.transforms.functional.crop(
            x2, x_diff // 2, y_diff // 2, x1.shape[1], x1.shape[2])

        x = torch.cat([x2_cropped, x1], dim=0)
        return self.double_conv(x)


if __name__ == "__main__":
    x = torch.rand(1, 572, 572)

    u_net = UNet()
    y = u_net(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
