"""Implementation of Palette: Image-to-Image Diffusion Models (Saharia et al.,
2022)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import write_png
from torchmetrics.functional import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim,
)
import pytorch_lightning as pl
from tqdm import tqdm
import os
from .guided_diffusion.unet import UNet
from .utils import denormalize, to_int


class Palette(pl.LightningModule):
    """
    Palette image-to-image diffusion model.

    :param in_channels: Input channels.
    :param out_channels: Output channels.
    :param inner_channels: Channel multiple, make sure it is a power of 2.
    :param channel_mults: Channel multipliers for each level of the U-net.
    :param num_res_blocks: Amount of residual blocks per layer.
    :param attention_res: Channel multipliers at which an attention layer
        should be added after the residual blocks.
    :param num_heads: Number of heads used by all attention layers.
    :param dropout: Dropout percentage.

    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        inner_channels: int = 64,
        channel_mults: tuple[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_res: tuple[int] = (8,),
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.unet = UNet(
            in_channel=in_channels * 2,
            out_channel=out_channels,
            inner_channel=inner_channels,
            channel_mults=channel_mults,
            res_blocks=num_res_blocks,
            attn_res=attention_res,
            num_heads=num_heads,
            dropout=dropout,
            conv_resample=True,
            image_size=256,
        )

        # Training scheduler
        self.diffusion = DiffusionModel(1e-6, 0.01, 2000, device=self.device)
        self.diffusion_inf = DiffusionModel(1e-4, 0.09, 1000, device=self.device)

    def forward(self, x, output_process=False):
        batch_size = x.shape[0]

        y_t = torch.randn_like(x)
        process_array = torch.unsqueeze(y_t, dim=1)
        for i in tqdm(reversed(range(self.diffusion_inf.timesteps))):
            t = torch.full((batch_size,), i, device=x.device)
            y_t = self.diffusion_inf.backward(x, y_t, t, self.unet)

            if (
                output_process
                and i % (self.diffusion_inf.timesteps // 7) == 0
            ):
                process_array = torch.cat(
                    [process_array, y_t.unsqueeze(1)],
                    dim=1,
                )

        if output_process:
            return y_t, process_array

        return y_t

    def loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loss function of Palette.

        :param pred: Predicted noise.
        :param target: Actual noise.
        :returns: Loss.

        """

        return F.l1_loss(pred, target)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=10000,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        x, y_0 = batch

        # Sample from p(gamma)
        t = torch.randint(
            0,
            self.diffusion.timesteps,
            size=(y_0.shape[0],),
            device=y_0.device,
        )
        y_t, noise, gamma = self.diffusion.forward(y_0, t)

        # Predict the added noise and compute loss
        noise_pred = self.unet(x, y_t, gamma)
        loss = self.loss(noise_pred, noise)

        self.log("loss", loss, prog_bar=True)

        return loss

    def on_validation_start(self):
        # Make dirs to save log video and output to
        epoch_dir = os.path.join(
            self.logger.log_dir,
            str(self.current_epoch + 1),
        )

        if not os.path.exists(epoch_dir):
            os.mkdir(epoch_dir)

    def validation_step(self, batch, batch_idx):
        x, y_0 = batch
        batch_size = x.shape[0]

        y_pred, process_array = self.forward(x, output_process=True)

        # Write diffusion processes of the model.
        for ind, process in enumerate(process_array):
            process_images = process.chunk(9)
            process = torch.cat([
                torch.cat([
                    process_images[0].squeeze(0),
                    process_images[1].squeeze(0),
                    process_images[2].squeeze(0),
                ], dim=2),
                torch.cat([
                    process_images[3].squeeze(0),
                    process_images[4].squeeze(0),
                    process_images[5].squeeze(0),
                ], dim=2),
                torch.cat([
                    process_images[6].squeeze(0),
                    process_images[7].squeeze(0),
                    process_images[8].squeeze(0),
                ], dim=2),
            ], dim=1)

            index = batch_size * batch_idx + ind

            write_png(
                to_int(denormalize(process)).cpu(),
                os.path.join(
                    self.logger.log_dir,
                    str(self.current_epoch + 1),
                    f"process_{index}.png",
                ),
                compression_level=0,
            )

        # Write outputs of the model.
        for ind, y_tx in enumerate(y_pred):
            index = batch_size * batch_idx + ind
            write_png(
                to_int(denormalize(y_tx)).cpu(),
                os.path.join(
                    self.logger.log_dir,
                    str(self.current_epoch + 1),
                    f"output_{index}.png",
                ),
                compression_level=0,
            )

        self.log(
            "val_ssim",
            ssim(denormalize(y_pred), denormalize(y_0), data_range=1.0),
            prog_bar=True,
        )
        self.log(
            "val_psnr",
            psnr(denormalize(y_pred), denormalize(y_0), data_range=1.0),
            prog_bar=True,
        )


class DiffusionModel(nn.Module):
    def __init__(self, start, end, timesteps: int, device="cpu"):
        super().__init__()

        self.timesteps = timesteps
        betas = torch.linspace(start, end, timesteps).to(device)

        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("gammas", torch.cumprod(self.alphas, axis=0))
        self.register_buffer(
            "gammas_prev",
            torch.cat([
                torch.ones((1,), device=self.gammas.device),
                self.gammas[:-1],
            ]),
        )

    def forward(self, y_0, t):
        """
        :param y_0: [N x C x H x W]
        :param y: [N]
        :returns: y_noised [N x C x H x W], noise [N x C x H x W], gamma [N]

        """

        noise = torch.randn_like(y_0) * (t > 0).view(-1, 1, 1, 1)
        gamma_prev = self.get_value(self.gammas_prev, t)
        gamma_cur = self.get_value(self.gammas, t)
        gamma = (gamma_cur-gamma_prev) * torch.rand_like(gamma_cur) + gamma_prev

        mean = torch.sqrt(gamma) * y_0
        variance = torch.sqrt(1 - gamma) * noise

        return mean + variance, noise, gamma.view(-1)

    def backward(self, x, y_t, t, noise_fn):
        """
        :param x: [N x C x H x W]
        :param y_t: [N x C x H x W]
        :param t: [N]
        :param noise_fn: Function that predicts the noise of the current
            timestep.
        :returns: [N x C X H x W]

        """

        alpha = self.get_value(self.alphas, t)
        gamma = self.get_value(self.gammas, t)
        gamma_prev = self.get_value(self.gammas_prev, t)
        noise_pred = noise_fn(x, y_t, gamma.view(-1))

        y_0_hat = (1 / torch.sqrt(gamma)) * (
            y_t -
            torch.sqrt(1 - gamma) * noise_pred
        )
        y_0_hat = torch.clamp(y_0_hat, -1, 1)

        mean = (
            (torch.sqrt(gamma_prev) * (1 - alpha) / (1 - gamma)) * y_0_hat +
            (torch.sqrt(alpha) * (1 - gamma_prev) / (1 - gamma)) * y_t
        )
        variance = (1 - gamma_prev) * (1 - alpha) / (1 - gamma)
        variance = torch.log(torch.clamp(variance, min=1e-20))
        sqrt_variance = torch.exp(0.5 * variance)

        noise = torch.randn_like(y_t) * (t > 1).view(-1, 1, 1, 1)

        return mean + sqrt_variance * noise

    def get_value(self, values, t):
        """
        Reshapes the value to be multiplied with a batch of images.

        :param values: [N x C x H x W]
        :param t: [N]
        :returns: [N x 1 x 1 x 1]

        """

        return values[t].view(-1, 1, 1, 1)
