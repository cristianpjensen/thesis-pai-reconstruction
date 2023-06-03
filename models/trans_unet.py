import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import math
from .gan import GAN, Discriminator


class TransUnetGAN(GAN):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        image_size: int = 256,
        channel_mults: tuple[int] = (1, 2, 4, 8),
        patch_size: int = 2,
        num_heads: int = 8,
        dropout: float = 0.5,
        l1_lambda: int = 50,
    ):
        generator = TransUnet(
            in_channels,
            out_channels,
            image_size=256,
            channel_mults=channel_mults,
            patch_size=patch_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        discriminator = Discriminator(in_channels)

        super().__init__(generator, discriminator, l1_lambda=l1_lambda)

        self.example_input_array = torch.Tensor(2, in_channels, 256, 256)
        self.save_hyperparameters()


class TransUnet(nn.Module):
    """Trans U-net.

    :param in_channels: Input channels that can vary if the images are
        grayscale or color.
    :param out_channels: Input channels that can vary if the images are
        grayscale or color.
    :param image_size: Input image size.
    :param channel_mults: Define how deep and wide the U-net is.

    :input: [N x in_channels x image_size x image_size]
    :output: [N x out_channels x image_size x image_size]

    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        image_size: int = 256,
        channel_mults: tuple[int] = (1, 2, 4, 8),
        patch_size: int = 16,
        num_heads: int = 8,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.in_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        in_channels = 64

        # Encoder blocks
        encoders = []
        for level, mult in enumerate(channel_mults):
            channels = mult * 64
            encoders.append(EncoderBlock(in_channels, channels))
            in_channels = channels

        self.encoders = nn.ModuleList(encoders)

        # Vision transformer bottleneck
        self.vit_bottleneck = VisionTransformer(
            channels=channel_mults[-1] * 64,
            input_size=image_size // (2 ** len(channel_mults)),
            patch_size=patch_size,
            num_heads=num_heads,
            dropout=dropout,
            transformer_layers=12,
        )

        # Decoder blocks
        decoders = []
        for level, mult in reversed(list(enumerate(channel_mults[:-1]))):
            channels = mult * 64
            decoders.append(DecoderBlock(in_channels, channels))
            in_channels = channels * 2

        decoders.append(DecoderBlock(in_channels, 64))
        self.decoders = nn.ModuleList(decoders)

        self.out = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.in_conv(x.type(torch.float32))

        skips = []
        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)

        skips.pop()

        h = self.vit_bottleneck(h)

        for index, decoder in enumerate(self.decoders):
            if index != 0:
                h = torch.cat([h, skips.pop()], dim=1)

            h = decoder(h)

        return self.out(h)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        channels: int,
        input_size: int,
        patch_size: int = 16,
        num_heads: int = 8,
        dropout: float = 0.5,
        transformer_layers: int = 12,
    ):
        super().__init__()

        patch_dim = channels * patch_size * patch_size
        num_patches = (input_size ** 2) // (patch_size ** 2)

        self.to_patch_embedding = nn.Sequential(
            # Get flattened patches
            Rearrange(
                "n c (h p1) (w p2) -> n (h w) (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, patch_dim),
            nn.LayerNorm(patch_dim),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, patch_dim)
        )

        trans_enc_layer = nn.TransformerEncoderLayer(
            patch_dim,
            num_heads,
            dropout=dropout,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            trans_enc_layer,
            transformer_layers,
        )

        self.to_image = Rearrange(
            "n (h w) (p1 p2 c) -> n c (h p1) (w p2)",
            h=int(math.sqrt(num_patches)),
            w=int(math.sqrt(num_patches)),
            p1=patch_size,
            p2=patch_size,
        )

    def forward(self, x):
        patch_emb = self.to_patch_embedding(x)
        patch_emb += self.pos_embedding
        patch_emb = self.transformer(patch_emb)
        return self.to_image(patch_emb)


class EncoderBlock(nn.Module):
    """Encoder block that downsamples the input by 2. Basically just a residual
    block as used in ResNet-50, ResNet-101, and ResNet-152 with a downsample.

    :param in_channels: Input channels.
    :param out_channels: Output channels.

    :input: [N x in_channels x image_size x image_size]
    :output: [N x out_channels x image_size x image_size]

    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        bottleneck = in_channels // 4

        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(),
            nn.Conv2d(
                bottleneck,
                bottleneck,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(),
            nn.Conv2d(bottleneck, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.skip = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.out = nn.ReLU()

    def forward(self, x):
        return self.out(self.decode(x) + self.skip(x))


class DecoderBlock(nn.Module):
    """Decoder block that upsamples the input by 2.

    :param in_channels: Input channels.
    :param out_channels: Output channels.

    :input: [N x in_channels x H x W]
    :output: [N x out_channels x H x W]

    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, x):
        return self.decode(x)
