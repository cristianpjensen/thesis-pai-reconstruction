"""Implementation of TransGAN: Two Pure Transformers Can Make One Strong GAN,
and That Can Scale Up (Jiang et al., 2021)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim,
)
import pytorch_lightning as pl


class TransGAN(pl.LightningModule):
    def __init__(self, l1_lambda: float):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(16, 3, 256, 256)
        self.automatic_optimization = False

        self.generator = TransformerUNet()
        self.discriminator = TransformerDiscriminator()

        self.l1_lambda = l1_lambda

    def forward(self, x):
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
        self.discriminator.zero_grad(set_to_none=True)

        input, target = batch
        pred = self.forward(input)

        target_label = self.discriminator(input, target)
        pred_label = self.discriminator(input, pred)
        d_loss = self.discriminator_loss(pred_label, target_label)
        self.log("d_loss", d_loss, prog_bar=True)

        self.manual_backward(d_loss)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        # Train generator.
        self.toggle_optimizer(opt_g)
        self.generator.zero_grad(set_to_none=True)

        pred = self.forward(input)
        pred_label = self.discriminator(input, pred)
        g_loss = self.generator_loss(pred, pred_label, target)
        g_psnr = psnr(pred, target, data_range=1.0)
        g_ssim = ssim(pred, target, data_range=1.0)
        self.log("g_loss", g_loss, prog_bar=True)
        self.log("train_ssim", g_ssim, prog_bar=True)
        self.log("train_psnr", g_psnr, prog_bar=True)

        self.manual_backward(g_loss)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

    def validation_step(self, batch, batch_idx):
        input, target = batch
        pred = self.forward(input)

        g_psnr = psnr(pred, target, data_range=1.0)
        g_ssim = ssim(pred, target, data_range=1.0)

        self.log("val_ssim", g_ssim, prog_bar=True)
        self.log("val_psnr", g_psnr, prog_bar=True)


class TransformerUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        self.down1 = Downsample(in_channels, 64, 256, 8)
        self.down2 = Downsample(64, 128, 128, 8)
        self.down3 = Downsample(128, 256, 64, 8)
        self.down4 = Downsample(256, 512, 32, 8)
        self.down5 = Downsample(512, 512, 16, 8)
        self.down6 = Downsample(512, 512, 8, 8)
        self.down7 = Downsample(512, 512, 4, 4)
        self.down8 = Downsample(512, 512, 2, 2)

        # Skip-connections are added here, so the amount of channels of the
        # output is doubled.
        self.up8 = Upsample(512, 512, 1, 1)
        self.up7 = Upsample(1024, 512, 2, 2)
        self.up6 = Upsample(1024, 512, 4, 4)
        self.up5 = Upsample(1024, 512, 8, 8)
        self.up4 = Upsample(1024, 256, 16, 8)
        self.up3 = Upsample(512, 128, 32, 8)
        self.up2 = Upsample(256, 64, 64, 8)
        self.up1 = Upsample(128, 64, 128, 8)

        self.out = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        """
        :param x: [N x C x H x W]
        :returns: [N x C x H x W]

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


class TransformerDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.stage1 = nn.Sequential(
            Downsample()
        )

        self.net = nn.Sequential(
            Downsample(in_channels * 2, 64, 256, 8),
            Downsample(64, 128, 128, 8),
            Downsample(128, 256, 64, 8),
            Downsample(256, 512, 32, 8),
            TransformerBlocks(512, 1, 16, 16),
        )

    def forward(self, x, y):
        xy_concat = torch.cat((x, y), dim=1)
        return self.net(xy_concat)


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: int,
        window_size: int,
        block_depth: int = 2,
    ):
        super().__init__()

        self.up = nn.Sequential(
            TransformerBlocks(
                in_channels,
                out_channels,
                image_size,
                window_size,
                depth=block_depth,
            ),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x (2 * H) x (2 * W)]

        """

        return self.up(x)


class Downsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: int,
        window_size: int,
        block_depth: int = 3,
    ):
        super().__init__()

        self.down = nn.Sequential(
            TransformerBlocks(
                in_channels,
                out_channels,
                image_size,
                window_size,
                depth=block_depth,
            ),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x (H / 2) x (W / 2)]

        """

        return self.down(x)


class TransformerBlocks(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: int,
        window_size: int,
        depth: int = 2,
    ):
        super().__init__()

        blocks = [
            TransformerBlock(
                in_channels,
                out_channels,
                image_size,
                window_size,
            ),
        ]

        blocks.extend([
            TransformerBlock(
                out_channels,
                out_channels,
                window_size,
                image_size,
            ) for _ in range(depth)
        ])

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x H x W]

        """

        return self.blocks(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        window_size: int,
        image_size: int,
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_channels) if in_channels > 32 else nn.Identity(),
            GridSelfAttention(in_channels, window_size),
        )

        dim = image_size * image_size
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, in_channels) if in_channels > 32 else nn.Identity(),
            MLP(dim, 4 * dim, dim),
            nn.GELU(),
        )

        self.out = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x H x W]

        """

        b, c, h, w = x.shape

        out = self.block1(x)
        out = out.reshape(b, c, -1)

        out = self.block2(out)
        out = out.reshape(b, c, h, w)

        return self.out(out)


class GridSelfAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        window_size: int,
        dropout: float = 0,
    ):
        super().__init__()

        self.window_size = window_size

        self.scale = channels ** -0.5
        self.qkv = nn.Linear(channels, channels * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.out = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        :param x: [N x C x H x W]
        :returns: [N x C x H x W]

        """

        w_size = self.window_size

        # Partition into grid of vectors:
        # [(N * num_windows) x (w_size * w_size) x C]
        n, c, h, w = x.shape
        windows = x.view(n, c, h // w_size, w_size, w // w_size, w_size)
        windows = windows.permute(0, 2, 4, 3, 5, 1)
        windows = windows.reshape(-1, w_size * w_size, c)

        # Do self-attention per window
        qkv = self.qkv(windows)
        query, key, value = qkv.chunk(3, dim=-1)

        scores = torch.bmm(query.transpose(1, 2), key) * self.scale
        attention = F.softmax(scores, dim=-1)
        attention = self.attn_drop(attention)

        weighted = torch.bmm(attention, value.transpose(1, 2)).transpose(1, 2)
        out = self.out(weighted)

        # Reconstruct image from windows: [N x C x H x W]
        out = out.view(n, h // w_size, w // w_size, w_size, w_size, c)
        out = out.permute(0, 5, 1, 3, 2, 4)
        out = out.reshape(n, c, h, w)

        return out


class Mlp(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0,
    ):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        :param x: [N x C x in_dim]
        :returns: [N x C x out_dim]

        """

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


if __name__ == "__main__":
    gen = TransformerUNet(3, 3)

    x = torch.randn((2, 3, 256, 256))
    y = gen(x)
    print(y.shape)
