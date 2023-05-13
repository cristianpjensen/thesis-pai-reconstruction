"""Implementation of Palette: Image-to-Image Diffusion Models (Saharia et al.,
2022)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.io import write_video
from torchmetrics.functional import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim,
)
import pytorch_lightning as pl
from .guided_diffusion.unet import UNet as GuidedDiffusionUNet
import math


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
    :param use_guided_diffusion: Whether to use the U-net implementation from
        the guided diffusion repository or to use this implementation. They
        should be exactly the same.

    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        inner_channels: int = 64,
        channel_mults: tuple[int] = (1, 2, 4, 8),
        num_res_blocks: int = 3,
        attention_res: tuple[int] = (4, 8),
        num_heads: int = 1,
        dropout: float = 0.,
        use_guided_diffusion: bool = False,
        output_diffusion_videos: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.output_video = output_diffusion_videos

        if use_guided_diffusion:
            self.unet = GuidedDiffusionUNet(
                in_channel=in_channels * 2,
                out_channel=out_channels,
                inner_channel=inner_channels,
                res_blocks=num_res_blocks,
                attn_res=attention_res,
                num_heads=num_heads,
                dropout=dropout,
                conv_resample=True,
                image_size=256,
            )
        else:
            self.unet = UNet(
                in_channels=in_channels,
                out_channels=out_channels,
                inner_channels=inner_channels,
                channel_mults=channel_mults,
                num_res_blocks=num_res_blocks,
                attention_res=attention_res,
                num_heads=num_heads,
                dropout=dropout,
            )

        # Training scheduler
        self.diffusion = DiffusionModel(1e-6, 0.01, 2000, device=self.device)
        self.diffusion_val = DiffusionModel(1e-4, 0.09, 1000, device=self.device)

        self.transform_back = transforms.Compose([
            transforms.ConvertImageDtype(torch.uint8),
        ])

    def forward(self, x):
        return x

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

    def validation_step(self, batch, batch_idx):
        x, y_0 = batch
        batch_size = x.shape[0]

        y_t = torch.randn_like(y_0)
        y_t_noise = torch.cat([y_t, torch.zeros_like(y_t)], dim=3)
        video_array = torch.unsqueeze(y_t_noise, dim=1)
        for i in reversed(range(self.diffusion_val.timesteps)):
            t = torch.full((batch_size,), i, device=x.device)
            y_t, noise_pred = self.diffusion_val.backward(x, y_t, t, self.unet)

            if self.output_video:
                y_t_noise = torch.cat([y_t, noise_pred], dim=3)
                video_array = torch.cat(
                    [video_array, torch.unsqueeze(y_t_noise, dim=1)],
                    dim=1,
                )

        if self.output_video:
            for ind, video in enumerate(video_array):
                video = video.permute(0, 2, 3, 1).cpu()
                video = self.transform_back(video)

                index = batch_size * batch_idx + ind
                write_video(f"./validation_{index}.mp4", video, fps=50)

        self.log("val_ssim", ssim(y_t, y_0, data_range=1.0), prog_bar=True)
        self.log("val_psnr", psnr(y_t, y_0, data_range=1.0), prog_bar=True)


class DiffusionModel(nn.Module):
    def __init__(self, start, end, timesteps: int = 2000, device="cpu"):
        super().__init__()

        self.timesteps = timesteps
        betas = torch.linspace(start, end, timesteps).to(device)

        self.register_buffer("alphas", 1 - betas)
        self.register_buffer("gammas", torch.cumprod(self.alphas, axis=0))
        self.register_buffer(
            "gammas_prev",
            torch.cat([torch.ones(1), self.gammas[:-1]], axis=0),
        )

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

        mean, variance, noise_pred = self._p_mean_variance(x, y_t, t, noise_fn)
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)

        return mean + torch.sqrt(variance) * noise, noise_pred

    def _p_mean_variance(self, x, y_t, t, noise_fn):
        gamma = self.get_value(self.gammas, t)
        alpha = self.get_value(self.alphas, t)
        gamma_prev = self.get_value(self.gammas_prev, t)

        noise_pred = noise_fn(x, y_t, gamma.view(-1))
        y0_hat = self._predict_y0(x, y_t, gamma, noise_pred)
        y0_hat = torch.clamp(y0_hat, -1, 1)
        mean = (
            (torch.sqrt(gamma_prev) * (1 - alpha) / (1 - gamma)) * y0_hat +
            (torch.sqrt(alpha) * (1 - gamma_prev) / (1 - gamma)) * y_t
        )
        variance = (1 - gamma_prev) * (1 - alpha) / (1 - gamma)

        return mean, variance, noise_pred

    def _predict_y0(self, x, y_t, gamma, noise_pred):
        return (
            torch.sqrt(1 / gamma) * y_t -
            torch.sqrt(1 / gamma - 1) * noise_pred
        )

    def get_value(self, values, t):
        """
        Reshapes the value to be multiplied with a batch of images.

        :param values: [N x C x H x W]
        :param t: [N]
        :returns: [N x 1 x 1 x 1]

        """

        return values[t].view(-1, 1, 1, 1)


class UNet(nn.Module):
    """
    U-net model with residual blocks, self-attention, and sinusoidal
    time-embeddings.

    :param in_channels: Input channels.
    :param out_channels: Output channels.
    :param inner_channels: Channel multiple, make sure it is a multiple of 2.
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
        num_res_blocks: int = 3,
        attention_res: tuple[int] = (4, 8),
        num_heads: int = 1,
        dropout: float = 0.,
    ):
        super().__init__()

        emb_dim = inner_channels * 4
        self.cond_emb = nn.Sequential(
            PositionalEncoding(inner_channels),
            nn.Linear(inner_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.init_conv = nn.Conv2d(
            in_channels * 2,
            inner_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Down
        downs = []
        in_out_mults = list(zip((1,) + channel_mults, channel_mults))
        for level, (in_mult, out_mult) in enumerate(in_out_mults):
            downs.append(ResAttentionDownBlock(
                in_mult * inner_channels,
                out_mult * inner_channels,
                emb_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                use_attention=out_mult in attention_res,
                num_heads=num_heads,
            ))

        self.downs = nn.ModuleList(downs)

        # Mid
        channels = channel_mults[-1] * inner_channels
        self.mid = nn.ModuleList([
            ResAttentionBlock(
                channels,
                channels,
                emb_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                num_heads=num_heads,
            ),
            ResAttentionBlock(
                channels,
                channels,
                emb_dim,
                num_res_blocks=1,
                dropout=dropout,
                num_heads=num_heads,
            ),
            ResAttentionBlock(
                channels,
                channels,
                emb_dim,
                num_res_blocks=1,
                dropout=dropout,
                use_attention=False,
            ),
        ])

        # Up
        ups = []
        for level, (out_mult, in_mult) in list(enumerate(in_out_mults))[::-1]:
            ups.append(ResAttentionUpBlock(
                2 * in_mult * inner_channels,
                out_mult * inner_channels,
                emb_dim,
                num_res_blocks=num_res_blocks + 1,
                dropout=dropout,
                use_attention=out_mult in attention_res,
                num_heads=num_heads,
            ))

        self.ups = nn.ModuleList(ups)

        channels = channel_mults[0] * inner_channels
        self.out = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x, y_noise, gamma):
        """
        :param x: [N x in_channels x H x W]
        :param y_noise: [N x in_channels x H x W]
        :param gamma: [N]
        :returns: [N x out_channels x H x W]

        """

        emb = self.cond_emb(gamma)
        h = self.init_conv(torch.cat([x, y_noise], dim=1))

        feats = []
        for layer in self.downs:
            h = layer(h, emb)
            feats.append(h)

        for layer in self.mid:
            h = layer(h, emb)

        for layer in self.ups:
            h = layer(torch.cat([h, feats.pop()], dim=1), emb)

        return self.out(h)


class PositionalEncoding(nn.Module):
    """
    Creates sinusoidal timestep embeddings for a vector of gammas.

    :param dim: Dimensionality of the output.
    :param max_period: Controls minimum frequency of the embeddings.

    """

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, gamma):
        """
        :param gamma: [N]
        :returns: [N x dim]

        """

        half = self.dim // 2
        steps = torch.arange(half, dtype=torch.float) / half
        freqs = torch.exp(-math.log(self.max_period) * steps).to(gamma.device)
        encoding = gamma.unsqueeze(1).float() * freqs.unsqueeze(0)
        embedding = torch.cat(
            [torch.cos(encoding), torch.sin(encoding)],
            dim=-1,
        )

        return embedding


class ResAttentionDownBlock(nn.Module):
    """
    Residual-Attention block with an downsample at the end.

    :param in_channels: Input channels.
    :param out_channels: Output channels.
    :param emb_dim: Embedding dimensionality.
    :param num_res_bocks: Amount of residual blocks.
    :param dropout: Dropout probability.
    :param use_attention: Whether to use self-attention after the residual
        blocks or not.
    :param num_heads: Amount of heads used in self-attention layer.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        num_res_blocks: int,
        dropout: float = 0.,
        use_attention: bool = True,
        num_heads: int = 1,
    ):
        super().__init__()

        self.res_attention = ResAttentionBlock(
            in_channels,
            out_channels,
            emb_dim,
            num_res_blocks,
            dropout,
            use_attention,
            num_heads,
        )
        self.down = Downsample(out_channels, out_channels)

    def forward(self, x, emb):
        """
        :param x: [N x in_channels x H x W]
        :param emb: [N x emb_dim]
        :returns: [N x out_channels x H / 2 x W / 2]

        """

        h = self.res_attention(x, emb)
        return self.down(h)


class Downsample(nn.Module):
    """
    Single convolution that downsamples the input (/2).

    :param in_channels: Input channels.
    :param out_channels: Output channels.

    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.down_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x H/2 x W/2]

        """

        return self.down_conv(x)


class ResAttentionUpBlock(nn.Module):
    """
    Residual-Attention block with an upsample at the end.

    :param in_channels: Input channels.
    :param out_channels: Output channels.
    :param emb_dim: Embedding dimensionality.
    :param num_res_bocks: Amount of residual blocks.
    :param dropout: Dropout probability.
    :param use_attention: Whether to use self-attention after the residual
        blocks or not.
    :param num_heads: Amount of heads used in self-attention layer.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        num_res_blocks: int,
        dropout: float = 0.,
        use_attention: bool = True,
        num_heads: int = 1,
    ):
        super().__init__()

        self.res_attention = ResAttentionBlock(
            in_channels,
            out_channels,
            emb_dim,
            num_res_blocks,
            dropout,
            use_attention,
            num_heads,
        )
        self.up = Upsample(out_channels, out_channels)

    def forward(self, x, emb):
        """
        :param x: [N x in_channels x H x W]
        :param emb: [N x emb_dim]
        :returns: [N x out_channels x H / 2 x W / 2]

        """

        h = self.res_attention(x, emb)
        return self.up(h)


class Upsample(nn.Module):
    """
    Upsample (x2) and convolution.

    :param in_channels: Input channels.
    :param out_channels: Output channels.

    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x 2*H x 2*W]

        """

        return self.up_conv(x)


class ResAttentionBlock(nn.Module):
    """
    Residual-Attention block.

    :param in_channels: Input channels.
    :param out_channels: Output channels.
    :param emb_dim: Time noise channels.
    :param num_res_blocks: Amount of residual blocks.
    :param dropout: Dropout percentage.
    :param use_attention: Whether to use self-attention after the residual
        blocks or not.
    :param num_heads: Amount of heads.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        num_res_blocks: int,
        dropout: float = 0,
        use_attention: bool = True,
        num_heads: int = 1,
    ):
        super().__init__()

        res = []
        for _ in range(num_res_blocks):
            res.append(
                ResBlock(
                    in_channels,
                    out_channels,
                    emb_dim,
                    dropout=dropout,
                )
            )

            in_channels = out_channels

        self.res = nn.ModuleList(res)
        self.attention = SelfAttention(
            out_channels,
            num_heads,
        ) if use_attention else nn.Identity()

    def forward(self, x, emb):
        """
        :param x: [N x in_channels x H x W]
        :param emb: [N x emb_dim]
        :returns: [N x out_channels x H x W]

        """

        for res in self.res:
            x = res(x, emb)

        return self.attention(x)


class SelfAttention(nn.Module):
    """
    Self-attention module with group normalization.

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
            nn.GroupNorm(32, channels),
            nn.Conv1d(
                channels,
                channels * 3,
                kernel_size=1,
            ),
        )

        self.out = nn.Conv2d(channels, channels, kernel_size=1)

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
        query_heads = query.reshape(batch_heads, self.channels_per_head, h * w)
        key_heads = key.reshape(batch_heads, self.channels_per_head, h * w)
        value_heads = value.reshape(batch_heads, self.channels_per_head, h * w)

        # They are already transposed, so transpose query back
        scores = torch.bmm(query_heads.transpose(1, 2), key_heads)
        scores /= math.sqrt(self.channels_per_head)
        attention = F.softmax(scores, dim=-1)

        weighted = torch.bmm(attention, value_heads.transpose(1, 2))
        weighted = weighted.reshape(n, -1, h, w)

        return self.out(weighted)


class ResBlock(nn.Module):
    """
    Residual block with a skip-connection.

    :param in_channels: Input channels.
    :param out_channels: Output channels.
    :param emb_dim: Time noise channels.
    :param dropout: Dropout percentage.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        dropout: float = 0,
    ):
        super().__init__()

        self.block1 = Block(in_channels, out_channels)
        self.block2 = Block(out_channels, out_channels, dropout=dropout)

        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

        # Sets the skip connection to the correct amount of channels
        self.skip_connection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
        ) if in_channels != out_channels else nn.Identity()

    def forward(self, x, emb):
        """
        :param x: [N x in_channels x H x W]
        :param emb: [N x emb_dim]
        :return: [N x out_channels x H x W]

        """

        h = self.block1(x)
        t = self.time_emb(emb).type(h.dtype)
        t = t.view(t.shape[0], t.shape[1], 1, 1)
        h = self.block2(h + t)

        return h + self.skip_connection(x)


class Block(nn.Module):
    """
    Convolution with normalization, activation function, and dropout.

    :param in_channels: Input channels.
    :param out_channels: Output channels.
    :param num_groups: Number of groups in the GroupNorm layer.
    :param dropout: Dropout percentage.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x H x W]

        """

        return self.block(x)


if __name__ == "__main__":
    # Make sure that the shapes are correct.
    x = torch.randn((3, 6, 256, 256))
    emb = torch.ones((3,))

    unet = UNet(6, 3, num_res_blocks=2, attention_res=(8,))
    y = unet(x, emb)
    print(x.shape, y.shape)