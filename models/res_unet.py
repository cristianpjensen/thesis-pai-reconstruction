import torch
import torch.nn as nn
from typing import Literal
from .wrapper import UnetWrapper


ResType = Literal["18", "50", "v2", "next"]


class ResUnetGAN(UnetWrapper):
    """Implementation of residual U-net.

    :param in_channels: Input channels that can vary if the images are
        grayscale or color.
    :param out_channels: Input channels that can vary if the images are
        grayscale or color.
    :param res_type: Which residual block to use.
    :param channel_mults: Channel multiples that define the depth and width of
        the U-net architecture.
    :param dropout: Dropout percentage used in some of the decoder blocks.
    :param loss_type: Loss type. One of "gan", "ssim", "psnr", "mse",
        "ssim+psnr".

    :input: [N x in_channels x H x W]
    :output: [N x out_channels x H x W]

    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        res_type: ResType = "18",
        channel_mults: tuple[int] = (1, 2, 4, 8, 8, 8, 8, 8),
        dropout: float = 0.5,
        loss_type: Literal["gan", "ssim", "psnr", "ssim+psnr", "mse"] = "gan",
    ):
        unet = ResUnet(
            in_channels,
            out_channels,
            res_type,
            channel_mults=channel_mults,
            dropout=dropout,
        )

        super().__init__(unet, loss_type=loss_type)

        self.example_input_array = torch.Tensor(2, in_channels, 256, 256)
        self.save_hyperparameters()


class ResidualBlock18(nn.Module):
    """Residual block as used in ResNet-18 and ResNet-34 (He et al. 2015)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

        self.out = nn.ReLU()

    def forward(self, x):
        return self.out(self.conv_block(x) + self.conv_skip(x))


class ResidualBlock50(nn.Module):
    """Residual block as used in ResNet-50, ResNet-101, and ResNet-152
    (He et al. 2015)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        bottleneck = in_channels // 4

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck, kernel_size=1),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(),
            nn.Conv2d(bottleneck, bottleneck, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(),
            nn.Conv2d(bottleneck, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

        self.out = nn.ReLU()

    def forward(self, x):
        return self.out(self.conv_block(x) + self.conv_skip(x))


class ResidualBlockV2(nn.Module):
    """Residual block as used in ResNet V2 (He et al. 2016)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.conv_skip = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class ResidualBlockNeXt(nn.Module):
    """Residual block as used in ResNeXt (Xie et al. 2017)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cardinality: int = 32,
        bottleneck: int = 4,
    ):
        super().__init__()

        branches = []

        for _ in range(cardinality):
            branch = nn.Sequential(
                nn.Conv2d(in_channels, bottleneck, kernel_size=1),
                nn.BatchNorm2d(bottleneck),
                nn.ReLU(),
                nn.Conv2d(bottleneck, bottleneck, kernel_size=3, padding=1),
                nn.BatchNorm2d(bottleneck),
                nn.ReLU(),
                nn.Conv2d(bottleneck, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )

            branches.append(branch)

        self.branches = nn.ModuleList(branches)
        self.branches_out = nn.ReLU()

        self.conv_skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        branches_sum = self.branches[0](x)
        for branch in self.branches[1:]:
            branches_sum += branch(x)

        return self.branches_out(branches_sum) + self.conv_skip(x)


res_blocks: dict[ResType, nn.Module] = {
    "18": ResidualBlock18,
    "50": ResidualBlock50,
    "v2": ResidualBlockV2,
    "next": ResidualBlockNeXt,
}


class EncoderBlock(nn.Module):
    """Encoder block that downsamples the input by 2.

    :param in_channels: Input channels.
    :param out_channels: Output channels.
    :param res_type: Which residual block to use.

    :input: [N x in_channels x H x W]
    :output: [N x out_channels x (H / 2) x (W / 2)]

    """

    def __init__(self, in_channels: int, out_channels: int, res_type: ResType):
        super().__init__()

        self.encode = nn.Sequential(
            res_blocks[res_type](in_channels, out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.encode(x)


class DecoderBlock(nn.Module):
    """Decoder block that upsamples the input by 2.

    :param in_channels: Input channels.
    :param out_channels: Output channels.
    :param dropout: Dropout percentage.
    :param res_type: Which residual block to use.

    :input: [N x in_channels x H x W]
    :output: [N x out_channels x (H * 2) x (W * 2)]

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        res_type: ResType,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.decode = nn.Sequential(
            res_blocks[res_type](in_channels, out_channels),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, x):
        return self.decode(x)


class ResUnet(nn.Module):
    """U-net used as the generator in pix2pix GAN.

    :param in_channels: Input channels that can vary if the images are
        grayscale or color.
    :param out_channels: Input channels that can vary if the images are
        grayscale or color.
    :param res_type: Which residual block to use.
    :param channel_mults: Channel multiples that define the depth and width of
        the U-net architecture.
    :param dropout: Dropout percentage used in some of the decoder blocks.

    :input: [N x in_channels x H x W]
    :output: [N x out_channels x H x W]

    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        res_type: ResType = "18",
        channel_mults: tuple[int] = (1, 2, 4, 8, 8, 8, 8, 8),
        dropout: float = 0.5,
    ):
        super().__init__()

        self.in_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        in_channels = 64

        # Encoder blocks
        encoders = []
        for level, mult in enumerate(channel_mults):
            channels = mult * 64
            encoders.append(EncoderBlock(in_channels, channels, res_type))
            in_channels = channels

        self.encoders = nn.ModuleList(encoders)

        # Decoder blocks
        decoders = []
        for level, mult in reversed(list(enumerate(channel_mults[:-1]))):
            channels = mult * 64

            decoders.append(
                DecoderBlock(
                    in_channels,
                    channels,
                    res_type,
                    # Only dropout in the lowest three decoder blocks that are
                    # at the widest part
                    dropout=dropout if (
                        mult == max(channel_mults) and
                        level > len(channel_mults) - 5
                    ) else 0,
                )
            )

            in_channels = channels * 2

        decoders.append(
            DecoderBlock(
                in_channels,
                channel_mults[0] * 64,
                res_type,
            )
        )

        self.decoders = nn.ModuleList(decoders)
        self.out = nn.Sequential(
            nn.Conv2d(
                channel_mults[0] * 64,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.in_conv(x.type(torch.float32))

        skips = []
        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)

        # Remove last feature map, since that should not be used in
        # skip-connection
        skips.pop()

        for index, decoder in enumerate(self.decoders):
            if index != 0:
                h = torch.cat([h, skips.pop()], dim=1)

            h = decoder(h)

        return self.out(h)
