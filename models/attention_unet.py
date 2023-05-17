"""Implementation of image-to-image translation models from pix2pix
(Isola et al., 2018)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim,
)
import pytorch_lightning as pl


class AttentionUNet(pl.LightningModule):
    def __init__(self, l1_lambda: float):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(32, 3, 256, 256)
        self.automatic_optimization = False

        self.generator = UNet()
        self.discriminator = Discriminator()

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


class UNet(nn.Module):
    """
    U-net with attention skip-connections.

    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super().__init__()

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)

        self.up4 = UpConv(512, 256)
        self.att4 = AttentionBlock(256, 256, 128)
        self.up_conv4 = ConvBlock(512, 256)

        self.up3 = UpConv(256, 128)
        self.att3 = AttentionBlock(128, 128, 64)
        self.up_conv3 = ConvBlock(256, 128)

        self.up2 = UpConv(128, 64)
        self.att2 = AttentionBlock(64, 64, 32)
        self.up_conv2 = ConvBlock(128, 64)

        self.out = ConvBlock(64, 3)

    def forward(self, x):
        """
        e: encoder layers
        d: decoder layers
        """

        e1 = self.conv1(x)

        e2 = F.max_pool2d(e1, 2)
        e2 = self.conv2(e2)

        e3 = F.max_pool2d(e2, 2)
        e3 = self.conv3(e3)

        e4 = F.max_pool2d(e3, 2)
        e4 = self.conv4(e4)

        d4 = self.up4(e4)
        s3 = self.att4(e3, d4)
        d4 = torch.cat([s3, d4], dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        s2 = self.att3(e2, d3)
        d3 = torch.cat([s2, d3], dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        s1 = self.att2(e1, d2)
        d2 = torch.cat([s1, d2], dim=1)
        d2 = self.up_conv2(d2)

        return self.out(d2)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.net = nn.Sequential(
            ConvBlock(in_channels * 2, 64),
            nn.MaxPool2d(2),
            ConvBlock(64, 128),
            nn.MaxPool2d(2),
            ConvBlock(128, 256),
            nn.MaxPool2d(2),
            ConvBlock(256, 512),
            ConvBlock(512, 1),
        )

        self.net.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0., 0.02)

    def forward(self, x, y):
        xy_concat = torch.cat((x, y), dim=1)
        return self.net(xy_concat)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.conv.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0., 0.02)

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.up.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0., 0.02)

    def forward(self, x):
        return self.up(x)


class AttentionBlock(nn.Module):
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

        self.input_gate.apply(self.init_weights)
        self.signal_gate.apply(self.init_weights)
        self.attention.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0., 0.02)

    def forward(self, x, signal):
        h_input = self.input_gate(x)
        h_signal = self.signal_gate(signal)
        h = F.relu(h_signal + h_input)
        attention = self.attention(h)

        return x * attention
