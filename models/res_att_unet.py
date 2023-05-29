import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim,
)
import pytorch_lightning as pl
import math
from typing import Literal, Optional
from .utils import denormalize


class ModernUnetGAN(pl.LightningModule):
    """GAN.

    :param in_channels: Channels of input images.
    :param out_channels: Channels of output images.
    :param num_res_blocks: Amount of residual blocks per layer.
    :param channel_mults: Channel multiples that define the depth and width of
        the U-net.
    :param att_mults: At which multipliers, attention blocks should be added.
    :param num_heads: Number of heads in attention blocks.
    :param dropout: Dropout percentage. Only used in the layers with maximum
        channel multiplication.
    :param l1_lambda: Weight given to the L1 loss over the discriminator loss.

    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_res_blocks: int = 3,
        channel_mults: tuple[int] = (1, 2, 4, 8,),
        att_mults: tuple[int] = (8,),
        num_heads: int = 4,
        dropout: float = 0.2,
        l1_lambda: float = 50,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(2, in_channels, 256, 256)
        self.automatic_optimization = False

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.l1_lambda = l1_lambda

        self.generator = UNet(
            in_channels,
            out_channels,
            num_res_blocks=num_res_blocks,
            channel_mults=channel_mults,
            attention_mults=att_mults,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.discriminator = Discriminator(
            in_channels,
            num_heads=num_heads,
        )

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x H x W]

        """

        return self.generator(x)

    def generator_loss(
        self,
        pred: torch.Tensor,
        pred_label: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loss function for generator.

        :param pred: Predicted image by generator.
        :param pred_label: Predicted label of generated image by discriminator.
        :param target: Target image.
        :returns: Loss.

        """

        # We want to fool the discriminator into predicting all ones.
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_label,
            torch.ones_like(pred_label),
        )

        l1_loss = F.l1_loss(pred, target)

        return bce_loss + self.l1_lambda * l1_loss

    def discriminator_loss(
        self,
        pred_label: torch.Tensor,
        target_label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loss function for discriminator.

        :param pred_label: predicted label of generated image by discriminator.
        :param target_label: predicted label of real target image by
            discriminator.
        :returns: Loss.

        """

        # The discriminator should predict all zeros for "fake" images.
        pred_loss = F.binary_cross_entropy_with_logits(
            pred_label,
            torch.zeros_like(pred_label),
        )

        # The discriminator should predict all ones for "real" images.
        target_loss = F.binary_cross_entropy_with_logits(
            target_label,
            torch.ones_like(pred_label),
        )

        return pred_loss + target_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.generator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.999),
            eps=1e-7,
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=2e-4,
            betas=(0.5, 0.999),
            eps=1e-7,
        )

        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        # Train discriminator.
        self.toggle_optimizer(opt_d)

        input, target = batch
        pred = self.forward(input)

        target_label = self.discriminator(input, target)
        pred_label = self.discriminator(input, pred)
        d_loss = self.discriminator_loss(pred_label, target_label)

        self.log("d_loss", d_loss, prog_bar=True)

        self.discriminator.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        opt_d.step()

        self.untoggle_optimizer(opt_d)

        # Train generator.
        self.toggle_optimizer(opt_g)

        pred = self.forward(input)
        pred_label = self.discriminator(input, pred)
        g_loss = self.generator_loss(pred, pred_label, target)

        g_ssim = ssim(denormalize(pred), denormalize(target), data_range=1.0)
        g_psnr = psnr(denormalize(pred), denormalize(target), data_range=1.0)

        self.log("g_loss", g_loss, prog_bar=True)
        self.log("train_ssim", g_ssim, prog_bar=True)
        self.log("train_psnr", g_psnr, prog_bar=True)

        self.generator.zero_grad(set_to_none=True)
        self.manual_backward(g_loss)
        opt_g.step()

        self.untoggle_optimizer(opt_g)

    def validation_step(self, batch, batch_idx):
        input, target = batch
        pred = self.forward(input)

        g_ssim = ssim(denormalize(pred), denormalize(target), data_range=1.0)
        g_psnr = psnr(denormalize(pred), denormalize(target), data_range=1.0)

        self.log("val_ssim", g_ssim, prog_bar=True)
        self.log("val_psnr", g_psnr, prog_bar=True)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        inner_channels: int = 64,
        channel_mults: list[int] = [1, 2, 4, 8],
        attention_mults: list[int] = [8],
        num_heads: int = 4,
    ):
        super().__init__()

        downs = [nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)]
        in_channels = 64
        input_block_channels = [64]
        for level, mult in enumerate(channel_mults):
            channels = mult * inner_channels

            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        in_channels,
                        channels,
                        dropout=dropout,
                    )
                ]

                if mult in attention_mults:
                    layers.append(
                        AttentionBlock(
                            channels,
                            num_heads=num_heads,
                        )
                    )

                downs.append(nn.Sequential(*layers))
                input_block_channels.append(channels)
                in_channels = channels

            if level != len(channel_mults) - 1:
                downs.append(
                    ResBlock(
                        channels,
                        channels,
                        dropout=dropout,
                        operation="down",
                    )
                )

                input_block_channels.append(channels)

        self.downs = nn.ModuleList(downs)

        self.mid = nn.Sequential(
            ResBlock(
                in_channels,
                in_channels,
                dropout=dropout,
            ),
            AttentionBlock(
                in_channels,
                num_heads=num_heads,
            ),
            ResBlock(
                in_channels,
                in_channels,
                dropout=dropout,
            ),
        )

        ups = []
        for level, mult in list(enumerate(channel_mults))[::-1]:
            channels = mult * inner_channels

            for i in range(num_res_blocks + 1):
                skip_channels = input_block_channels.pop()

                layers = [
                    ResBlock(
                        in_channels + skip_channels,
                        channels,
                        dropout=dropout,
                    )
                ]

                if mult in attention_mults:
                    layers.append(
                        AttentionBlock(
                            channels,
                            num_heads=num_heads,
                        )
                    )

                in_channels = channels

                if level > 0 and i == num_res_blocks:
                    layers.append(
                        ResBlock(
                            channels,
                            channels,
                            dropout=dropout,
                            operation="up",
                        )
                    )

                ups.append(nn.Sequential(*layers))

        self.ups = nn.ModuleList(ups)

        self.out = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            zero_module(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
        )

    def forward(self, x):
        skips = []
        h = x.type(torch.float32)

        for layer in self.downs:
            h = layer(h)
            skips.append(h)

        h = self.mid(h)

        for layer in self.ups:
            h = torch.cat([h, skips.pop()], dim=1)
            h = layer(h)

        return self.out(h.type(x.dtype))


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()

        self.net = nn.Sequential(
            ResBlock(in_channels * 2, 64),
            ResBlock(64, 64),
            ResBlock(64, 64, operation="down"),

            ResBlock(64, 128),
            ResBlock(128, 128),
            ResBlock(128, 128, operation="down"),

            ResBlock(128, 256),
            AttentionBlock(256, num_heads=num_heads),
            ResBlock(256, 256),
            AttentionBlock(256, num_heads=num_heads),
            ResBlock(256, 256, operation="down"),

            ResBlock(256, 512),
            AttentionBlock(512, num_heads=num_heads),
            ResBlock(512, 512),
            AttentionBlock(512, num_heads=num_heads),
            ResBlock(512, 512, operation="down"),

            nn.Conv2d(512, 1, kernel_size=3, padding=1),
        )

    def forward(self, x, y):
        """
        :param x: [N x in_channels x H x W]
        :param y: [N x in_channels x H x W]
        :returns: [N x 1 x OUT x OUT]

        """

        xy_concat = torch.cat([x, y], dim=1)
        return self.net(xy_concat)


class ResBlock(nn.Module):
    """Residual block with skip-connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        operation: Optional[Literal["up", "down"]] = None,
    ):
        super().__init__()

        self.operation = operation

        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            self.operation_module(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            zero_module(
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            ),
        )

        self.skip_connection = nn.Sequential(
            self.operation_module(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
            ),
        )

    def operation_module(self):
        if self.operation == "up":
            return nn.Upsample(scale_factor=2, mode="nearest")

        if self.operation == "down":
            return nn.AvgPool2d(2)

        return nn.Identity()

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x H x W]

        """

        return self.net(x) + self.skip_connection(x)


class AttentionBlock(nn.Module):
    """
    Attention block that allows spatial positions to attend to each other.

    :param channels: Input and output channels.
    :param num_heads: Amount of heads.

    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
    ):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.channels_per_head = self.channels // self.num_heads

        self.qkv_conv = nn.Sequential(
            nn.BatchNorm1d(channels),
            nn.Conv1d(
                channels,
                channels * 3,
                kernel_size=1,
            ),
        )

        self.out = zero_module(nn.Conv1d(channels, channels, kernel_size=1))

    def forward(self, x):
        """
        :param x: [N x channels x H x W]
        :returns: [N x channels x H x W]

        """

        # Reshape so all pixels are one after another
        n, c, h, w = x.shape
        batch_heads = n * self.num_heads
        x = x.reshape(n, c, -1)

        # Compute query, key, and value
        qkv = self.qkv_conv(x)
        query, key, value = qkv.chunk(3, dim=1)

        # Reshape to divide channels by heads
        scale = 1 / math.sqrt(self.channels_per_head)
        query_heads = query.reshape(batch_heads, self.channels_per_head, h * w)
        key_heads = key.reshape(batch_heads, self.channels_per_head, h * w)
        value_heads = value.reshape(batch_heads, self.channels_per_head, h * w)

        scores = torch.einsum(
            "bct,bcs->bts",
            query_heads * scale,
            key_heads * scale,
        )
        attention = F.softmax(scores.float(), dim=-1).type(scores.dtype)
        weighted = torch.einsum("bts,bcs->bct", attention, value_heads)
        weighted = weighted.reshape(n, c, -1)

        return (x + self.out(weighted)).reshape(n, -1, h, w)


def zero_module(module: nn.Module):
    for p in module.parameters():
        p.detach().zero_()

    return module
