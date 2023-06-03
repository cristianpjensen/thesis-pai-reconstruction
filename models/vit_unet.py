import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
import math
from .gan import GAN, Discriminator


class ViTUnetGAN(GAN):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        l1_lambda: int = 50,
    ):
        generator = ViTUnet(in_channels, out_channels, image_size=256)
        discriminator = Discriminator(in_channels)

        super().__init__(generator, discriminator, l1_lambda=l1_lambda)

        self.example_input_array = torch.Tensor(2, in_channels, 256, 256)
        self.save_hyperparameters()


class ViTUnet(nn.Module):
    """Vision transformer U-net.

    :param in_channels: Input channels that can vary if the images are
        grayscale or color.
    :param out_channels: Input channels that can vary if the images are
        grayscale or color.

    :input: [N x in_channels x image_size x image_size]
    :output: [N x out_channels x image_size x image_size]

    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        image_size: int = 256,
        patch_size: tuple[int] = (16, 8, 4, 2),
        num_heads: tuple[int] = (2, 4, 8, 16),
        channel_mults: tuple[int] = (1, 2, 4, 8),
        mlp_ratio: int = 2,
        dropout: float = 0.5,
        transformer_layers: int = 2,
    ):
        super().__init__()

        # Encoder blocks
        encoders = []
        for level, mult in enumerate(channel_mults):
            channels = mult * 8
            encoders.append(
                EncoderBlock(
                    in_channels,
                    channels,
                    image_size=image_size // (2 ** level),
                    patch_size=patch_size[level],
                    num_heads=num_heads[level],
                    mlp_ratio=mlp_ratio,
                    transformer_layers=transformer_layers,
                    norm=level > 0,
                )
            )
            in_channels = channels

        self.encoders = nn.ModuleList(encoders)

        # Decoder blocks
        decoders = []
        for level, mult in reversed(list(enumerate(channel_mults[:-1], 1))):
            channels = mult * 8
            decoders.append(
                DecoderBlock(
                    in_channels,
                    channels,
                    image_size=image_size // (2 ** (level + 1)),
                    patch_size=patch_size[level],
                    num_heads=num_heads[level],
                    mlp_ratio=mlp_ratio,
                    transformer_layers=transformer_layers,
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
                out_channels,
                image_size=image_size // 2,
                patch_size=patch_size[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratio,
                transformer_layers=transformer_layers,
                dropout=0,
            )
        )

        self.decoders = nn.ModuleList(decoders)
        self.out = nn.Tanh()

    def forward(self, x):
        h = x.type(torch.float32)

        skips = []
        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)

        skips.pop()

        for index, decoder in enumerate(self.decoders):
            if index != 0:
                h = torch.cat([h, skips.pop()], dim=1)

            h = decoder(h)

        return self.out(h)


class EncoderBlock(nn.Module):
    """Encoder block that downsamples the input by 2.

    :param in_channels: Input channels.
    :param out_channels: Output channels.
    :param image_size: Input image size.
    :param patch_size: Patch size.
    :param num_heads: Number of attention heads.
    :param mlp_ratio: Multiplicator for the hidden dimensionality of the
        linear projection to embed the patches.
    :param transformer_layers: Amount of layers in the transformer encoder.
    :param norm: Whether to apply batch normalization or not.

    :input: [N x in_channels x image_size x image_size]
    :output: [N x out_channels x image_size x image_size]

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: int,
        patch_size: int,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        transformer_layers: int = 3,
        norm: bool = True,
    ):
        super().__init__()

        self.decode = nn.Sequential(
            nn.BatchNorm2d(in_channels) if norm else nn.Identity(),
            VisionTransformerBlock(
                in_channels,
                image_size,
                patch_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                transformer_layers=transformer_layers,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels) if norm else nn.Identity(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.decode(x)


class DecoderBlock(nn.Module):
    """Decoder block that upsamples the input by 2.

    :param in_channels: Input channels.
    :param out_channels: Output channels.
    :param image_size: Input image size.
    :param patch_size: Patch size.
    :param num_heads: Number of attention heads.
    :param mlp_ratio: Multiplicator for the hidden dimensionality of the
        linear projection to embed the patches.
    :param transformer_layers: Amount of layers in the transformer encoder.
    :param dropout: Dropout percentage.

    :input: [N x in_channels x image_size x image_size]
    :output: [N x out_channels x image_size x image_size]

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: int,
        patch_size: int,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        transformer_layers: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.decode = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            VisionTransformerBlock(
                in_channels,
                image_size,
                patch_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                transformer_layers=transformer_layers,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, x):
        return self.decode(x)


class PositionalEncoding(nn.Module):
    def __init__(self, temperature: int = 10_000):
        super().__init__()

        self.temperature = temperature

    def forward(self, patches):
        _, h, w, dim = patches.shape
        device = patches.device

        y, x = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )

        omega = torch.arange(dim // 4, device=device) / (dim // 4 - 1)
        omega = 1 / (self.temperature ** omega)

        y = y.flatten().unsqueeze(1) * omega.unsqueeze(0)
        x = x.flatten().unsqueeze(1) * omega.unsqueeze(0)

        pe = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)

        return rearrange(patches, "n ... d -> n (...) d") + pe


class VisionTransformerBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        image_size: int,
        patch_size: int,
        num_heads: int = 12,
        mlp_ratio: int = 4,
        transformer_layers: int = 3,
    ):
        super().__init__()

        patch_dim = channels * patch_size * patch_size
        hidden_dim = int(patch_dim * mlp_ratio)
        image_patch_width = int(math.sqrt((image_size ** 2) / (patch_size ** 2)))

        trans_enc_layer = nn.TransformerEncoderLayer(
            patch_dim,
            num_heads,
            activation="gelu",
        )

        self.block = nn.Sequential(
            # Get flattened patches
            Rearrange(
                "n c (h p1) (w p2) -> n h w (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            ),

            # Embed patches with MLP
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, hidden_dim),
            nn.Linear(hidden_dim, patch_dim),
            nn.LayerNorm(patch_dim),

            # Encode with transformer
            PositionalEncoding(),
            nn.TransformerEncoder(trans_enc_layer, transformer_layers),

            # Reshape back to images
            Rearrange(
                "n (h w) (p1 p2 c) -> n c (h p1) (w p2)",
                h=image_patch_width,
                w=image_patch_width,
                p1=patch_size,
                p2=patch_size,
            )
        )

    def forward(self, x):
        return self.block(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    vit_unet = ViTUnet(3, 3, 256)

    print("params:", count_parameters(vit_unet))

    y = vit_unet(x)

    print(y.shape)
