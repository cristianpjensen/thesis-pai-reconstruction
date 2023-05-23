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
from .utils import denormalize


class Pix2Pix(pl.LightningModule):
    """Implementation of the Pix2Pix image-to-image translation GAN.

    :param in_channels: Channels of input images.
    :param out_channels: Channels of output images.
    :param channel_mults: Channel multiples that define the depth and width of
        the U-net.
    :param dropout: Dropout percentage. Only used in the layers with maximum
        channel multiplication.
    :param l1_lambda: Weight given to the L1 loss over the discriminator loss.

    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channel_mults: tuple[int] = (1, 2, 4, 8, 8, 8, 8, 8),
        dropout: float = 0.5,
        l1_lambda: float = 50,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(32, in_channels, 256, 256)
        self.automatic_optimization = False

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.l1_lambda = l1_lambda

        self.generator = UNet(
            in_channels,
            out_channels,
            channel_mults=channel_mults,
            dropout=dropout,
        )
        self.discriminator = Discriminator(in_channels)

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
        in_channels: int = 3,
        out_channels: int = 3,
        channel_mults: tuple[int] = (1, 2, 4, 8, 8, 8, 8, 8),
        dropout: float = 0.5,
    ):
        super().__init__()

        downs = []
        for index, mult in enumerate(channel_mults):
            channels = mult * 64

            downs.append(
                Downsample(
                    in_channels,
                    channels,
                    batchnorm=index != 0,
                )
            )

            in_channels = channels

        self.downs = nn.ModuleList(downs)

        ups = []
        for index, mult in reversed(list(enumerate(channel_mults[:-1]))):
            channels = mult * 64

            ups.append(
                Upsample(
                    in_channels,
                    channels,
                    dropout=dropout if index < 3 else 0,
                )
            )

            # Multiply by 2 for the skip-connections.
            in_channels = channels * 2

        ups.append(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            )
        )

        self.ups = nn.ModuleList(ups)
        self.out = nn.Tanh()

        self.ups.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0., 0.02)

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x H x W]
        """

        h = x.type(torch.float32)

        feats = []
        for layer in self.downs:
            h = layer(h)
            feats.append(h)

        # Remove last feature map, since that should not be used in
        # skip-connection.
        feats.pop()

        for index, layer in enumerate(self.ups):
            if index != 0:
                h = torch.cat([h, feats.pop()], dim=1)

            h = layer(h)

        return self.out(h)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3):
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
        """
        :param x: [N x in_channels x H x W]
        :param y: [N x in_channels x H x W]
        :returns: [1 x OUT x OUT]

        """

        xy_concat = torch.cat((x, y), dim=1)
        return self.net(xy_concat)


class Downsample(nn.Module):
    """Convolution-BatchNorm-ReLU encoder layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        size: int = 4,
        stride: int = 2,
        padding: int = 1,
        batchnorm: bool = True,
    ):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity(),
            nn.LeakyReLU(0.2),
        )

        self.down.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0., 0.02)

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x H / 2 x W / 2]

        """

        return self.down(x)


class Upsample(nn.Module):
    """Convolution-BatchNorm-ReLU decoder layer with optional dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        size: int = 4,
        stride: int = 2,
        padding: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.ReLU(),
        )

        self.up.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0., 0.02)

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x H * 2 x W * 2]

        """

        return self.up(x)
