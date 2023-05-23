"""Implementation of Palette: Image-to-Image Diffusion Models (Saharia et al.,
2022)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import write_video, write_png
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

    def forward(self, x, output_video=False):
        batch_size = x.shape[0]

        y_t = torch.randn_like(x)
        video_array = torch.unsqueeze(y_t, dim=1)
        for i in tqdm(reversed(range(self.diffusion_inf.timesteps))):
            t = torch.full((batch_size,), i, device=x.device)
            y_t = self.diffusion_inf.backward(x, y_t, t, self.unet)

            if output_video:
                video_array = torch.cat(
                    [video_array, torch.unsqueeze(y_t, dim=1)],
                    dim=1,
                )

        if output_video:
            rgb_video_array = torch.cat(
                [video_array, video_array, video_array],
                dim=2,
            )
            return y_t, rgb_video_array

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

        return F.mse_loss(pred, target)

    def get_beta_schedule(
        self,
        steps: int = 2000,
        start: float = 1e-6,
        end: float = 1e-2,
        warmup_frac: float = 0.5,
    ):
        betas = end * torch.ones(steps, dtype=torch.float)
        warmup_steps = int(steps * warmup_frac)
        betas[:warmup_steps] = torch.linspace(
            start,
            end,
            warmup_steps,
            dtype=torch.float,
        )

        return betas

    def configure_optimizers(self):
        return torch.optim.Adam(self.unet.parameters(), lr=1e-4)

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
        epoch_dir = os.path.join(self.logger.log_dir, str(self.current_epoch))
        val_diffusion_dir = os.path.join(epoch_dir, "val_diffusion")
        val_output_dir = os.path.join(epoch_dir, "val_output")

        if not os.path.exists(epoch_dir):
            os.mkdir(epoch_dir)

        if not os.path.exists(val_diffusion_dir):
            os.mkdir(val_diffusion_dir)

        if not os.path.exists(val_output_dir):
            os.mkdir(val_output_dir)

    def validation_step(self, batch, batch_idx):
        x, y_0 = batch
        batch_size = x.shape[0]

        y_pred, video_array = self.forward(x, output_video=True)

        # Output video and model outputs
        for ind, video in enumerate(video_array):
            video = to_int(denormalize(video))
            video = video.permute(0, 2, 3, 1).cpu()

            index = batch_size * batch_idx + ind
            write_video(
                os.path.join(
                    self.logger.log_dir,
                    str(self.current_epoch),
                    "val_diffusion",
                    f"diffusion_{index}.mp4",
                ),
                video,
                fps=self.diffusion_inf.timesteps / 10,
            )

        for ind, y_tx in enumerate(y_pred):
            index = batch_size * batch_idx + ind
            write_png(
                to_int(denormalize(y_tx)).cpu(),
                os.path.join(
                    self.logger.log_dir,
                    str(self.current_epoch),
                    "val_output",
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

    def forward(self, y_0, t):
        """
        :param y_0: [N x C x H x W]
        :param y: [N]
        :returns: [N x C x H x W], [N x C x H x W], [N]

        """

        noise = torch.randn_like(y_0) * (t > 0).view(-1, 1, 1, 1)
        gamma = self.get_value(self.gammas, t)

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
        noise_pred = noise_fn(x, y_t, gamma.view(-1))

        mean = (1 / torch.sqrt(alpha)) * (
            y_t -
            ((1 - alpha) / torch.sqrt(1 - gamma)) * noise_pred
        )
        sqrt_variance = torch.sqrt(1 - alpha)

        noise = torch.randn_like(y_t) * (t > 0).view(-1, 1, 1, 1)

        return torch.clamp(mean + sqrt_variance * noise, -1, 1)

    def get_value(self, values, t):
        """
        Reshapes the value to be multiplied with a batch of images.

        :param values: [N x C x H x W]
        :param t: [N]
        :returns: [N x 1 x 1 x 1]

        """

        return values[t].view(-1, 1, 1, 1)
