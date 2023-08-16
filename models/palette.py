"""Implementation of Palette: Image-to-Image Diffusion Models (Saharia et al.,
2022)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import write_png
import pytorch_lightning as pl
from tqdm import tqdm
import os
import math
from typing import Literal
from .guided_diffusion.unet import UNet
from .utils import denormalize, to_int, ssim, psnr, rmse


class Palette(pl.LightningModule):
    """
    Palette image-to-image diffusion model.

    :param in_channels: Input channels.
    :param out_channels: Output channels.
    :param channel_mults: Channel multipliers for each level of the U-net.
    :param attention_res: Downsample rates at which attention blocks should be
        added after the residual blocks.
    :param num_heads: Number of heads used by all attention layers.
    :param dropout: Dropout percentage.
    :param schedule_type: Noise schedule type. Either cosine or linear.
    :param learn_var: Learn the variance aswell.

    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channel_mults: tuple[int] = (1, 1, 2, 2, 4, 4),
        attention_res: tuple[int] = (16, 8),
        dropout: float = 0.1,
        schedule_type: Literal["linear", "cosine"] = "linear",
        learn_var: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learn_var = learn_var

        self.unet = UNet(
            in_channel=in_channels * 2,
            out_channel=out_channels * 2 if learn_var else out_channels,
            res_blocks=2,
            inner_channel=128,
            channel_mults=channel_mults,
            attn_res=attention_res,
            num_heads=4,
            dropout=dropout,
            conv_resample=True,
            image_size=256,
        )

        # Noise schedules
        self.diffusion = DiffusionModel(
            schedule_type,
            2000,
            1e-6,
            0.01,
            learn_var=learn_var,
            device=self.device,
        )
        self.diffusion_inf = DiffusionModel(
            "cosine",
            100,
            learn_var=learn_var,
            device=self.device,
        )

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

        # Predict the added noise (and optionally variance) and compute loss
        model_output = self.unet(x, y_t, gamma)

        noise_pred = model_output
        if self.learn_var:
            noise_pred, var_pred = model_output.split(x.shape[1], dim=1)

        loss = F.mse_loss(noise_pred, noise)
        vlb_loss = self.diffusion.vlb_term(model_output, y_0, y_t, t).mean()

        self.log("mse_loss", loss, prog_bar=True)
        self.log("vlb_loss", vlb_loss, prog_bar=True)

        if self.learn_var:
            loss += 0.001 * vlb_loss

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

        # Write outputs of the model
        for ind, y_tx in enumerate(y_pred):
            write_png(
                to_int(denormalize(y_tx)).cpu(),
                os.path.join(
                    self.logger.log_dir,
                    str(self.current_epoch + 1),
                    f"output_{batch_size * batch_idx + ind}.png",
                ),
                compression_level=0,
            )

        den_y_pred = denormalize(y_pred)
        den_y_0 = denormalize(y_0)

        self.log("val_ssim", ssim(den_y_pred, den_y_0), prog_bar=True)
        self.log("val_psnr", psnr(den_y_pred, den_y_0), prog_bar=True)
        self.log("val_rmse", rmse(den_y_pred, den_y_0), prog_bar=True)


class DiffusionModel(nn.Module):
    def __init__(
        self,
        schedule_type: Literal["linear", "cosine"],
        timesteps: int,
        start: float = 1e-6,
        end: float = 0.01,
        learn_var: bool = False,
        device="cpu",
    ):
        super().__init__()

        self.timesteps = timesteps
        self.learn_var = learn_var

        match schedule_type:
            case "linear":
                betas = linear_beta_schedule(timesteps, start, end)

            case "cosine":
                betas = cosine_beta_schedule(timesteps)

            case _:
                raise ValueError(f"{schedule_type} is not supported.")

        betas = betas.to(device)

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
        :param t: [N]
        :returns: y_noised [N x C x H x W], noise [N x C x H x W], gamma [N]

        """

        noise = torch.randn_like(y_0) * (t > 0).view(-1, 1, 1, 1)
        gamma_prev = self.get_value(self.gammas_prev, t)
        gamma_cur = self.get_value(self.gammas, t)
        gamma = (gamma_cur-gamma_prev) * \
            torch.rand_like(gamma_cur) + gamma_prev

        mean = torch.sqrt(gamma) * y_0
        variance = torch.sqrt(1 - gamma) * noise

        return mean + variance, noise, gamma.view(-1)

    def backward(self, x, y_t, t, model):
        """
        :param x: [N x C x H x W]
        :param y_t: [N x C x H x W]
        :param t: [N]
        :param model: U-net model that predicts the noise and optionally the
            variance.
        :returns: [N x C X H x W]

        """

        gamma = self.gammas[t]
        model_output = model(x, y_t, gamma)

        mean, log_variance = self.p_mean_variance(model_output, y_t, t)
        sqrt_variance = torch.exp(0.5 * log_variance)

        noise = torch.randn_like(y_t) * (t > 1).view(-1, 1, 1, 1)

        return mean + sqrt_variance * noise

    def q_mean_variance(self, y_0, y_t, t):
        """Compute q(y_{t-1} | y_t, y_0) parameters."""

        alpha = self.get_value(self.alphas, t)
        gamma = self.get_value(self.gammas, t)
        gamma_prev = self.get_value(self.gammas_prev, t)

        mean = (
            (torch.sqrt(gamma_prev) * (1 - alpha) / (1 - gamma)) * y_0 +
            (torch.sqrt(alpha) * (1 - gamma_prev) / (1 - gamma)) * y_t
        )
        var_lower_bound = (1 - alpha) * (1 - gamma_prev) / (1 - gamma)
        var_lower_bound = torch.clamp(var_lower_bound, min=1e-20)
        log_variance = torch.log(var_lower_bound)

        return mean, log_variance

    def p_mean_variance(self, model_output, y_t, t):
        """Compute p(y_{t-1} | y_t) parameters."""

        alpha = self.get_value(self.alphas, t)
        gamma = self.get_value(self.gammas, t)
        gamma_prev = self.get_value(self.gammas_prev, t)

        # If the variance is not learned, we want the lower bound of the
        # variance, so fix var_interp to 0
        var_interp = 0
        noise_pred = model_output
        if self.learn_var:
            noise_pred, var_interp = model_output.split(y_t.shape[1], dim=1)
            # The range of the U-net is [-1, 1]
            var_interp = (var_interp + 1) / 2

        var_lower_bound = (1 - alpha) * (1 - gamma_prev) / (1 - gamma)
        var_lower_bound = torch.clamp(var_lower_bound, min=1e-20)
        var_upper_bound = 1 - alpha

        log_variance = (
            var_interp * torch.log(var_upper_bound) +
            (1-var_interp) * torch.log(var_lower_bound)
        )

        y_0_hat = (1 / torch.sqrt(gamma)) * (
            y_t -
            torch.sqrt(1 - gamma) * noise_pred
        )
        y_0_hat = torch.clamp(y_0_hat, -1, 1)

        mean = (
            (torch.sqrt(gamma_prev) * (1 - alpha) / (1 - gamma)) * y_0_hat +
            (torch.sqrt(alpha) * (1 - gamma_prev) / (1 - gamma)) * y_t
        )
        return mean, log_variance

    def vlb_term(self, model_output, y_0, y_t, t):
        """Compute a term for the variational lower-bound."""

        # Learn the variance using the variational bound, but do not let it
        # affect the mean prediction
        if self.learn_var:
            noise_pred, var_interp = model_output.split(y_t.shape[1], dim=1)
            model_output = torch.cat([noise_pred.detach(), var_interp], dim=1)

        true_mean, true_log_variance = self.q_mean_variance(y_0, y_t, t)
        pred_mean, pred_log_variance = self.p_mean_variance(
            model_output, y_t, t)

        kl = normal_kl(true_mean, true_log_variance,
                       pred_mean, pred_log_variance)
        # Take mean for each item in batch
        kl = kl.mean(dim=[1, 2, 3]) / math.log(2.0)

        nll = -discretized_gaussian_log_likelihood(
            y_0,
            means=pred_mean,
            log_scales=0.5 * pred_log_variance,
        )
        nll = nll.mean(dim=[1, 2, 3]) / math.log(2.0)

        return torch.where(t == 0, nll, kl)

    def get_value(self, values, t):
        """
        Reshapes the value to be multiplied with a batch of images.

        :param values: [N x C x H x W]
        :param t: [N]
        :returns: [N x 1 x 1 x 1]

        """

        return values[t].view(-1, 1, 1, 1)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule as proposed in (Nichol and Dhariwal, 2021)."""

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    gammas = torch.cos((torch.pi / 2) * ((x / timesteps) + s) / (1 + s))
    gammas = gammas / gammas[0]
    betas = 1 - (gammas[1:] / gammas[:-1])

    return torch.clamp(betas, 0.0001, 0.9999)


def linear_beta_schedule(
    timesteps: int,
    start: float = 1e-6,
    end: float = 0.01,
) -> torch.Tensor:
    return torch.linspace(start, end, timesteps)


def normal_kl(mean1, log_var1, mean2, log_var2):
    """Compute the KL divergence between two Gaussians."""

    # Set variances to be tensors
    if not isinstance(log_var1, torch.Tensor):
        log_var1 = torch.tensor(log_var1).to(mean1.device)

    if not isinstance(log_var2, torch.Tensor):
        log_var2 = torch.tensor(log_var2).to(mean2.device)

    return 0.5 * (
        -1.0 +
        (log_var2 - log_var1) +
        torch.exp(log_var1 - log_var2) +
        ((mean1 - mean2) ** 2) * torch.exp(-log_var2)
    )


def approx_standard_normal_cdf(x):
    """A fast approximation of the cumulative distribution function of the
    standard normal."""

    return 0.5 * (
        1.0 +
        torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: The target images. It is assumed that this was uint8 values,
        rescaled to the range [-1, 1].
    :param means: The Gaussian mean Tensor.
    :param log_scales: The Gaussian log stddev Tensor.
    :returns: A tensor like x of log probabilities (in nats).

    """

    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999,
            log_one_minus_cdf_min,
            torch.log(cdf_delta.clamp(min=1e-12)),
        ),
    )

    return log_probs
