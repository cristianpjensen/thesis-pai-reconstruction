"""Implementation of image-to-image translation models from pix2pix
(Isola et al., 2018)."""

import torch
import torch.nn as nn
from models.layers.pix2pix import Downsample, Upsample, UpsampleDropout


class GeneratorEncoderDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.enc = nn.Sequential(
            Downsample(in_channels, 64),
            Downsample(64, 128),
            Downsample(128, 256),
            Downsample(256, 512),
            Downsample(512, 512),
            Downsample(512, 512),
            Downsample(512, 512),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.dec = nn.Sequential(
            UpsampleDropout(512, 512),
            UpsampleDropout(512, 512),
            UpsampleDropout(512, 512),
            Upsample(512, 512),
            Upsample(512, 256),
            Upsample(256, 128),
            Upsample(128, 64),
            nn.ConvTranspose2d(
                64,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=False),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.enc(x)
        decoded = self.dec(encoded)
        return decoded


class GeneratorUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()

        self.down1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.down1 = Downsample(in_channels, 64)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.down4 = Downsample(256, 512)
        self.down5 = Downsample(512, 512)
        self.down6 = Downsample(512, 512)
        self.down7 = Downsample(512, 512)
        self.down8 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
        )

        self.up1 = UpsampleDropout(512, 512)
        self.up2 = UpsampleDropout(1024, 512)
        self.up3 = UpsampleDropout(1024, 512)
        self.up4 = Upsample(1024, 512)
        self.up5 = Upsample(1024, 256)
        self.up6 = Upsample(512, 128)
        self.up7 = Upsample(256, 64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=False),
        )

        self.out = nn.Tanh()

    def forward(self, x):
        # Encoder
        # `f` is the feature vector that results from the encoder.
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        f = self.down8(d7)

        # Decoder
        u1_ = self.up1(f)
        u1 = torch.cat([u1_, d7], dim=1)
        u2_ = self.up2(u1)
        u2 = torch.cat([u2_, d6], dim=1)
        u3_ = self.up3(u2)
        u3 = torch.cat([u3_, d5], dim=1)
        u4_ = self.up4(u3)
        u4 = torch.cat([u4_, d4], dim=1)
        u5_ = self.up5(u4)
        u5 = torch.cat([u5_, d3], dim=1)
        u6_ = self.up6(u5)
        u6 = torch.cat([u6_, d2], dim=1)
        u7_ = self.up7(u6)
        u7 = torch.cat([u7_, d1], dim=1)
        y = self.up8(u7)

        return self.out(y)


class PixelDiscriminator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.dis = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        xy_concat = torch.cat([x, y], dim=1)
        return self.dis(xy_concat)


class Patch16Discriminator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.dis = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            Downsample(64, 128, stride=1),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        xy_concat = torch.cat([x, y], dim=1)
        return self.dis(xy_concat)


class Patch70Discriminator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.dis = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            Downsample(64, 128),
            Downsample(128, 256),
            Downsample(256, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        xy_concat = torch.cat([x, y], dim=1)
        return self.dis(xy_concat)


class Patch286Discriminator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.dis = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            Downsample(64, 128),
            Downsample(128, 256),
            Downsample(256, 512),
            Downsample(512, 512),
            Downsample(512, 512),
            Downsample(512, 512, stride=1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, y):
        xy_concat = torch.cat([x, y], dim=1)
        return self.dis(xy_concat)
