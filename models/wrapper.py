import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Literal
from .utils import denormalize, init_weights, ssim, psnr, rmse


class UnetWrapper(pl.LightningModule):
    """U-net wrapper with different loss functions.

    :param unet: U-net model.
    :param loss_type: Loss function for the U-net. One of "gan", "ssim",
        "psnr", "mse", "ssim+psnr".

    :input: [N x C x H x W]
    :output: [N x C x H x W]

    """

    def __init__(
        self,
        unet: nn.Module,
        loss_type: Literal["gan", "ssim", "psnr", "ssim+psnr" "mse"] = "gan",
    ):
        super().__init__()
        self.automatic_optimization = False

        self.unet = unet
        self.loss_type = loss_type

        self.discriminator = None
        if loss_type == "gan":
            self.discriminator = Discriminator()
            self.discriminator.apply(init_weights)

        self.unet.apply(init_weights)

    def forward(self, x):
        return self.unet(x)

    def loss(self, x, pred, target):
        if self.loss_type == "gan":
            pred_label = self.discriminator(x, pred)
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_label,
                torch.ones_like(pred_label),
            )
            l1_loss = F.l1_loss(pred, target)

            return bce_loss + 50 * l1_loss

        if self.loss_type == "ssim":
            return -ssim(denormalize(pred), denormalize(target))

        if self.loss_type == "psnr":
            return -psnr(denormalize(pred), denormalize(target))

        if self.loss_type == "ssim+psnr":
            return -(
                30 * ssim(denormalize(pred), denormalize(target)) +
                psnr(denormalize(pred), denormalize(target))
            )

        if self.loss_type == "mse":
            return F.mse_loss(pred, target)

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

        # The discriminator should predict all zeros for "fake" images
        pred_loss = F.binary_cross_entropy_with_logits(
            pred_label,
            torch.zeros_like(pred_label),
        )

        # The discriminator should predict all ones for "real" images
        target_loss = F.binary_cross_entropy_with_logits(
            target_label,
            torch.ones_like(pred_label),
        )

        return pred_loss + target_loss

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(
            self.unet.parameters(),
            lr=2e-4,
            betas=(0.5, 0.999),
            eps=1e-7,
        )

        if self.discriminator is not None:
            opt_d = torch.optim.Adam(
                self.discriminator.parameters(),
                lr=2e-4,
                betas=(0.5, 0.999),
                eps=1e-7,
            )

            return opt_g, opt_d

        return opt_g

    def training_step(self, batch, batch_idx):
        x, target, _ = batch

        if self.loss_type == "gan":
            opt_d = self.optimizers()[1]

            # Train discriminator.
            self.toggle_optimizer(opt_d)

            pred = self.unet(x)

            target_label = self.discriminator(x, target)
            pred_label = self.discriminator(x, pred)
            d_loss = self.discriminator_loss(pred_label, target_label)

            self.log("d_loss", d_loss, prog_bar=True)

            self.discriminator.zero_grad(set_to_none=True)
            self.manual_backward(d_loss)
            opt_d.step()

            self.untoggle_optimizer(opt_d)

        opt_g = self.optimizers()
        if isinstance(opt_g, list):
            opt_g = opt_g[0]

        # Train U-net
        self.toggle_optimizer(opt_g)

        pred = self.unet(x)
        loss = self.loss(x, pred, target)

        den_pred = denormalize(pred)
        den_target = denormalize(target)

        self.log("loss", loss, prog_bar=True)
        self.log("train_ssim", ssim(den_pred, den_target), prog_bar=True)
        self.log("train_psnr", psnr(den_pred, den_target), prog_bar=True)
        self.log("train_rmse", rmse(den_pred, den_target), prog_bar=True)

        self.unet.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        opt_g.step()

        self.untoggle_optimizer(opt_g)

    def validation_step(self, batch, batch_idx):
        wandb_logger = self.loggers[1]

        x, target, _ = batch
        pred = self.forward(x)

        for y in pred:
            wandb_logger.log_image(
                key="predictions",
                images=[denormalize(y)],
            )

        den_pred = denormalize(pred)
        den_target = denormalize(target)

        self.log("val_ssim", ssim(den_pred, den_target), prog_bar=True)
        self.log("val_psnr", psnr(den_pred, den_target), prog_bar=True)
        self.log("val_rmse", rmse(den_pred, den_target), prog_bar=True)


class DiscriminatorBlock(nn.Module):
    """An encoder block that is used in the discriminator.

    :param in_channels: Input channels.
    :param out_channels: Output channels.
    :param norm: Whether to use normalization or not.

    :input: [N x in_channels x H x W]
    :output: [N x out_channels x (H / 2) x (W / 2)]

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: bool = False,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.InstanceNorm2d(out_channels) if norm else nn.Identity(),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.block(x)


class Discriminator(nn.Module):
    """Discriminator that distinguishes between real and fake images. This
        particular discriminator is used in all GANs in this repository.

    :param in_channels: Input channels for both the input and conditional
        image.

    :input x: [N x in_channels x H x W]
    :input y: [N x in_channels x H x W]
    :output: [1 x OUT x OUT]

    """

    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.discriminator = nn.Sequential(
            DiscriminatorBlock(in_channels * 2, 64, norm=False),
            DiscriminatorBlock(64, 128),
            DiscriminatorBlock(128, 256),
            DiscriminatorBlock(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False),
        )

    def forward(self, x, y):
        h = torch.cat([x, y], dim=1)
        return self.discriminator(h)
