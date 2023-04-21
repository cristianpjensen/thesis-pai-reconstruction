import torch
import torch.nn as nn
from models.layers.pix2pix import Downsample, Upsample, UpsampleDropout


class Pix2PixAttentionUNet(nn.Module):
    """Generator from pix2pix, modified to use attention skip-connections."""

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

        self.up7 = UpsampleDropout(512, 512)
        self.att7 = AttentionBlock(512, 512, 512)
        self.up6 = UpsampleDropout(1024, 512)
        self.att6 = AttentionBlock(512, 512, 512)
        self.up5 = UpsampleDropout(1024, 512)
        self.att5 = AttentionBlock(512, 512, 512)
        self.up4 = Upsample(1024, 512)
        self.att4 = AttentionBlock(512, 512, 512)
        self.up3 = Upsample(1024, 256)
        self.att3 = AttentionBlock(512, 512, 256)
        self.up2 = Upsample(512, 128)
        self.att2 = AttentionBlock(256, 256, 128)
        self.up1 = Upsample(256, 64)
        self.att1 = AttentionBlock(128, 128, 64)

        self.out = nn.Sequential(
            nn.Conv2d(
                128,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=False),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        `e`: encoder layers
        `d`: decoder layers
        `s`: skip-connection with attention from encoder to decoder layers
        """

        e1 = self.down1(x)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)
        e5 = self.down5(e4)
        e6 = self.down6(e5)
        e7 = self.down7(e6)
        d8 = self.down8(e7)

        d7 = self.up7(d8)
        s7 = self.att7(gate=d8, skip_connection=e7)
        d7 = torch.cat((d7, s7), dim=1)

        d6 = self.up6(d7)
        s6 = self.att6(gate=d7, skip_connection=e6)
        d6 = torch.cat((d6, s6), dim=1)

        d5 = self.up5(d6)
        s5 = self.att5(gate=d6, skip_connection=e5)
        d5 = torch.cat((d5, s5), dim=1)

        d4 = self.up4(d5)
        s4 = self.att4(gate=d5, skip_connection=e4)
        d4 = torch.cat((d4, s4), dim=1)

        d3 = self.up3(d4)
        s3 = self.att3(gate=d4, skip_connection=e3)
        d3 = torch.cat((d3, s3), dim=1)

        d2 = self.up2(d3)
        s2 = self.att2(gate=d3, skip_connection=e2)
        d2 = torch.cat((d2, s2), dim=1)

        d1 = self.up1(d2)
        s1 = self.att1(gate=d2, skip_connection=e1)
        d1 = torch.cat((d1, s1), dim=1)

        return self.out(d1)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        g_channels: int,
        x_channels: int,
        num_coefficients
    ):
        super().__init__()

        self.w_gate = nn.Sequential(
            nn.Conv2d(
                g_channels,
                num_coefficients,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(num_coefficients),
        )

        self.w_x = nn.Sequential(
            nn.Conv2d(
                x_channels,
                num_coefficients,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(num_coefficients),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(
                num_coefficients,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        g1 = self.w_gate(gate)
        x1 = self.w_x(skip_connection)
        additive_att = self.relu(x1 + g1)
        att = self.psi(additive_att)
        return skip_connection * att
