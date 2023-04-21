"""Implementation of image-to-image translation models from pix2pix
(Isola et al., 2018)."""

import torch
import torch.nn as nn
from models.layers.unet import Downsample, Upsample


class GeneratorUNet(nn.Module):
    """The performance of this generator is the baseline to which we compare
    other models."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super().__init__()

        # According to the baseline, the batchnorm should be omitted in the
        # first layer, but according to the pix2pix paper it should be omitted
        # in the last layer.

        # Input: 256 (pixels), 3 (channels)
        self.down1 = Downsample(in_channels, 64, batchnorm=False)  # 128, 64
        self.down2 = Downsample(64, 128)   # 64, 128
        self.down3 = Downsample(128, 256)  # 32, 256
        self.down4 = Downsample(256, 512)  # 16, 512
        self.down5 = Downsample(512, 512)  # 8, 512
        self.down6 = Downsample(512, 512)  # 4, 512
        self.down7 = Downsample(512, 512)  # 2, 512
        self.down8 = Downsample(512, 512)  # 1, 512

        # Skip-connections are added here, so the amount of channels of the
        # output is doubled.
        self.up8 = Upsample(512, 512, dropout=True)   # 2, 1024
        self.up7 = Upsample(1024, 512, dropout=True)  # 4, 1024
        self.up6 = Upsample(1024, 512, dropout=True)  # 8, 1024
        self.up5 = Upsample(1024, 512)  # 16, 1024
        self.up4 = Upsample(1024, 256)  # 32, 512
        self.up3 = Upsample(512, 128)   # 64, 256
        self.up2 = Upsample(256, 64)    # 128, 128
        self.up1 = nn.ConvTranspose2d(
            128,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.out = nn.Tanh()

        nn.init.normal_(self.up1.weight, 0., 0.02)

    def forward(self, x):
        """
        `e`: encoder layers
        `d`: decoder layers
        """

        # Encoder.
        e1 = self.down1(x)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)
        e5 = self.down5(e4)
        e6 = self.down6(e5)
        e7 = self.down7(e6)
        e8 = self.down8(e7)

        # Decoder with skip-connections.
        d8 = self.up8(e8)
        d8 = torch.cat((d8, e7), dim=1)
        d7 = self.up7(d8)
        d7 = torch.cat((d7, e6), dim=1)
        d6 = self.up6(d7)
        d6 = torch.cat((d6, e5), dim=1)
        d5 = self.up5(d6)
        d5 = torch.cat((d5, e4), dim=1)
        d4 = self.up4(d5)
        d4 = torch.cat((d4, e3), dim=1)
        d3 = self.up3(d4)
        d3 = torch.cat((d3, e2), dim=1)
        d2 = self.up2(d3)
        d2 = torch.cat((d2, e1), dim=1)
        d1 = self.up1(d2)

        return self.out(d1)


class GeneratorEncoderDecoder(nn.Module):
    """Encoder-Decoder generator to measure how much the skip-connections make
    a difference."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        # Input: 256 (pixels), 3 (channels)
        self.net = nn.Sequential(
            Downsample(in_channels, 64, batchnorm=False),  # 128, 64
            Downsample(64, 128),   # 64, 128
            Downsample(128, 256),  # 32, 256
            Downsample(256, 512),  # 16, 512
            Downsample(512, 512),  # 8, 512
            Downsample(512, 512),  # 4, 512
            Downsample(512, 512),  # 2, 512
            Downsample(512, 512),  # 1, 512
            Upsample(512, 512, dropout=True),  # 2, 512
            Upsample(512, 512, dropout=True),  # 4, 512
            Upsample(512, 512, dropout=True),  # 8, 512
            Upsample(512, 512),  # 16, 512
            Upsample(512, 256),  # 32, 256
            Upsample(256, 128),  # 64, 128
            Upsample(128, 64),  # 128, 64
            nn.ConvTranspose2d(
                64,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.Tanh(),
        )

        self.net.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0., 0.02)

    def forward(self, x):
        return self.net(x)


class Patch70Discriminator(nn.Module):
    """Discriminator used for the baseline. Discriminators with other patch
    sizes can be found in the git history if necessary."""

    def __init__(self, in_channels: int):
        super().__init__()

        # The discriminator has to distinguish between the real label and the
        # prediction by the generator. So, it has two images as input, thus the
        # channels are doubled.

        # Input: 256 (pixels), 6 (channels)
        self.net = nn.Sequential(
            Downsample(in_channels * 2, 64, batchnorm=False),  # 128, 64
            Downsample(64, 128),   # 64, 128
            Downsample(128, 256),  # 32, 256
            nn.ZeroPad2d(1),       # 34, 256
            Downsample(256, 512, stride=1, padding=0),  # 31, 512
            nn.ZeroPad2d(1),  # 33, 512
            nn.Conv2d(
                512,
                1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),  # 30, 1
        )

        self.net.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0., 0.02)

    def forward(self, x, y):
        xy_concat = torch.cat((x, y), dim=1)
        return self.net(xy_concat)


if __name__ == "__main__":
    # Testing
    generator = GeneratorUNet(3, 3)

    x = torch.rand((2, 3, 256, 256))
    y = generator(x)
    print("256, 3:", x.shape)
    print("256, 3:", y.shape)

    discriminator = Patch70Discriminator(3)
    z = discriminator(x, y)
    print("30, 1:", z.shape)
