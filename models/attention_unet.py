import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from .wrapper import UnetWrapper
from .pix2pix import EncoderBlock, DecoderBlock


class AttentionUnetGAN(UnetWrapper):
    """The same model as pix2pix modified to use attention in the skip
    connections (Oktay et al. 2018).

    :param in_channels: Input channels that can vary if the images are
        grayscale or color.
    :param out_channels: Input channels that can vary if the images are
        grayscale or color.
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
        channel_mults: tuple[int] = (1, 2, 4, 8, 8, 8, 8, 8),
        dropout: float = 0.5,
        loss_type: Literal["gan", "ssim", "psnr", "ssim+psnr", "mse"] = "gan",
    ):
        unet = AttentionUnet(
            in_channels,
            out_channels,
            channel_mults=channel_mults,
            dropout=dropout,
        )

        super().__init__(unet, loss_type=loss_type)

        self.example_input_array = torch.Tensor(2, in_channels, 256, 256)
        self.save_hyperparameters()


class AttentionBlock(nn.Module):
    """Attention block used in the skip connections of the attention U-net.

    :param input_channels: Amount of channels that the encoder layer that is
        being skipped has.
    :param signal_channels: Amount of channels that the signal has, which is
        the output of the previous decoder layer.
    :param attention_channels: Amount of channels that the input and signal are
        mapped to.

    :input x: [N x input_channels x H x W]
    :input signal: [N x signal_channels x H x W]
    :output: [N x input_channels x H x W]

    """

    def __init__(
        self,
        input_channels: int,
        signal_channels: int,
        attention_channels: int,
    ):
        super().__init__()

        self.input_gate = nn.Sequential(
            nn.Conv2d(input_channels, attention_channels, kernel_size=1),
            nn.BatchNorm2d(attention_channels),
        )

        self.signal_gate = nn.Sequential(
            nn.Conv2d(signal_channels, attention_channels, kernel_size=1),
            nn.BatchNorm2d(attention_channels),
        )

        self.attention = nn.Sequential(
            nn.Conv2d(attention_channels, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU()

    def forward(self, x, signal):
        h_input = self.input_gate(x)
        h_signal = self.signal_gate(signal)
        h = self.relu(h_signal + h_input)
        attention = self.attention(h)

        return x * attention


class AttentionUnet(nn.Module):
    """U-net with attention used in the skip-connections.

    :param in_channels: Input channels that can vary if the images are
        grayscale or color.
    :param out_channels: Input channels that can vary if the images are
        grayscale or color.
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
        channel_mults: tuple[int] = (1, 2, 4, 8, 8, 8, 8, 8),
        dropout: float = 0.5,
    ):
        super().__init__()

        # Encoder blocks
        encoders = [
            nn.Conv2d(
                in_channels,
                channel_mults[0] * 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
        ]
        in_channels = channel_mults[0] * 64
        for level, mult in enumerate(channel_mults[1:], 1):
            channels = mult * 64

            encoders.append(
                EncoderBlock(
                    in_channels,
                    channels,
                    norm=level != len(channel_mults) - 1,
                )
            )

            in_channels = channels

        self.encoders = nn.ModuleList(encoders)

        # Decoder and attention blocks
        decoders = []
        attention_blocks = []
        for level, mult in reversed(list(enumerate(channel_mults[:-1]))):
            channels = mult * 64

            decoders.append(
                DecoderBlock(
                    in_channels,
                    channels,
                    # Only dropout in the lowest three decoder blocks that are
                    # at the widest part
                    dropout=dropout if (
                        mult == max(channel_mults) and
                        level > len(channel_mults) - 5
                    ) else 0,
                )
            )
            attention_blocks.append(
                AttentionBlock(channels, channels, channels // 2)
            )

            in_channels = channels * 2

        decoders.append(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
        )

        self.decoders = nn.ModuleList(decoders)
        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.out = nn.Tanh()

    def forward(self, x):
        h = x.type(torch.float32)

        feats = []
        for encoder in self.encoders:
            h = encoder(h)
            feats.append(h)

        # Remove last feature map, since that should not be used in
        # skip-connection
        feats.pop()

        for index, decoder in enumerate(self.decoders):
            if index != 0:
                attention = self.attention_blocks[index - 1]
                s = attention(feats.pop(), h)
                h = torch.cat([h, s], dim=1)

            h = decoder(h)

        return self.out(h)
