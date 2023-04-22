"""Implementation of various parts from the pix2pix paper (Isola et al., 2018).

Ck (Convolution-BatchNorm-ReLU with k filters): Downsample and Upsample.
CDk (Convolution-BatchNorm-Dropout-ReLU with k filters): Upsample with
`dropout=True`.

"""

import torch.nn as nn


class Downsample(nn.Module):
    """Convolution-BatchNorm-ReLU encoder layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        size: int = 4,
        stride: int = 2,
        padding: int = 1,
        batchnorm: bool = True,
    ):
        super().__init__()

        if batchnorm:
            self.down = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
            )
        else:
            self.down = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.LeakyReLU(0.2),
            )

        self.down.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0., 0.02)

    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):
    """Convolution-BatchNorm-ReLU decoder layer with optional dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        size: int = 4,
        stride: int = 2,
        padding: int = 1,
        dropout: bool = False,
    ):
        super().__init__()

        if dropout:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.Dropout(0.5),
                nn.ReLU(),
            )
        else:
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

        self.up.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0., 0.02)

    def forward(self, x):
        return self.up(x)
