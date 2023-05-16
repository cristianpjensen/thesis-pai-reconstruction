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


class Pix2Pix(pl.LightningModule):
    def __init__(self, l1_lambda: float):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(32, 3, 256, 256)
        self.automatic_optimization = False

        self.generator = GeneratorUNet()
        self.discriminator = Patch70Discriminator()

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


class GeneratorUNet(nn.Module):
    """The performance of this generator is the baseline to which we compare
    other models."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super().__init__()

        # According to the baseline, the batchnorm should be omitted in the
        # first layer, but according to the pix2pix paper it should be omitted
        # in the last layer.

        # Input: 256 (pixels), 3 (channels)
        self.down1 = Downsample(in_channels, 64, batchnorm=False)  # 128, 64
        self.down2 = Downsample(64, 128)   # 64, 128
        self.down3 = Downsample(128, 256)  # 32, 256
        self.down4 = Downsample(256, 512)  # 16, 512
        self.down5 = Downsample(512, 512)  # 8, 512
        self.down6 = Downsample(512, 512)  # 4, 512
        self.down7 = Downsample(512, 512)  # 2, 512
        self.down8 = Downsample(512, 512)  # 1, 512

        # Skip-connections are added here, so the amount of channels of the
        # output is doubled.
        self.up8 = Upsample(512, 512, dropout=True)   # 2, 1024
        self.up7 = Upsample(1024, 512, dropout=True)  # 4, 1024
        self.up6 = Upsample(1024, 512, dropout=True)  # 8, 1024
        self.up5 = Upsample(1024, 512)  # 16, 1024
        self.up4 = Upsample(1024, 256)  # 32, 512
        self.up3 = Upsample(512, 128)   # 64, 256
        self.up2 = Upsample(256, 64)    # 128, 128
        self.up1 = nn.ConvTranspose2d(
            128,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.out = nn.Tanh()

        nn.init.normal_(self.up1.weight, 0., 0.02)

    def forward(self, x):
        """
        e: encoder layers
        d: decoder layers
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


class Patch70Discriminator(nn.Module):
    """Discriminator used for the baseline. Discriminators with other patch
    sizes can be found in the git history if necessary."""

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

        if batchnorm:
            self.down = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
            )
        else:
            self.down = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.LeakyReLU(0.2),
            )

        self.down.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0., 0.02)

    def forward(self, x):
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
        dropout: bool = False,
    ):
        super().__init__()

        if dropout:
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
                nn.Dropout(0.5),
                nn.ReLU(),
            )
        else:
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
                nn.ReLU(),
            )

        self.up.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0., 0.02)

    def forward(self, x):
        return self.up(x)
