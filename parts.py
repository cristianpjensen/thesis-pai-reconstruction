"""Implementation of various parts from the pix2pix paper (Isola et al., 2018).

Ck (Convolution-BatchNorm-ReLU with k filters): Downsample and Upsample.
CDk (Convolution-BatchNorm-Dropout-ReLU with k filters): UpsampleDropout.

"""

import torch.nn as nn


class Downsample(nn.Module):
    """Convolution-BatchNorm-ReLU encoder layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        padding: int = 1
    ):
        super().__init__()

        self.down = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.down(x)


class UpsampleDropout(nn.Module):
    """Convolution-BatchNorm-Dropout-ReLU layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.up(x)


class Upsample(nn.Module):
    """Convolution-BatchNorm-ReLU decoder layer."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.up(x)
