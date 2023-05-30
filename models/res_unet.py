import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim,
)
import pytorch_lightning as pl
from .utils import denormalize


class ResUnetGAN(pl.LightningModule):
    """GAN with residual U-net.

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
        channel_mults: tuple[int] = (1, 2, 4, 8,),
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

        self.generator = ResUnet(
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
        wandb_logger = self.loggers[1]

        input, target = batch
        pred = self.forward(input)

        for y in pred:
            wandb_logger.log_image(
                key="predictions",
                images=[denormalize(y)],
            )

        g_ssim = ssim(denormalize(pred), denormalize(target), data_range=1.0)
        g_psnr = psnr(denormalize(pred), denormalize(target), data_range=1.0)

        self.log("val_ssim", g_ssim, prog_bar=True)
        self.log("val_psnr", g_psnr, prog_bar=True)


class ResUnet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channel_mults: tuple[int] = (1, 2, 4, 8, 8, 8),
        dropout: float = 0.5,
    ):
        super().__init__()

        downs = []
        for mult in channel_mults:
            channels = mult * 64
            downs.append(Downsample(in_channels, channels))
            in_channels = channels

        self.downs = nn.ModuleList(downs)

        ups = []
        for index, mult in reversed(list(enumerate(channel_mults[:-1]))):
            channels = mult * 64

            ups.append(
                Upsample(
                    in_channels,
                    channels,
                    dropout=dropout if (
                        index < 3 and
                        mult == max(channel_mults)
                    ) else 0
                )
            )

            in_channels = channels * 2

        ups.append(Upsample(in_channels, 64))

        self.ups = nn.ModuleList(ups)

        self.out = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        self.out.apply(init_weights)

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x H x W]

        """

        h = x.type(torch.float32)

        skips = []
        for layer in self.downs:
            h = layer(h)
            skips.append(h)

        # Remove last skip connection, snce that should not be used.
        skips.pop()

        for index, layer in enumerate(self.ups):
            # The first upsample does not get a skip connection.
            if index != 0:
                h = torch.cat([h, skips.pop()], dim=1)

            h = layer(h)

        return self.out(h)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        self.net = nn.Sequential(
            Downsample(in_channels * 2, 64),
            Downsample(64, 128),
            Downsample(128, 256),
            nn.ZeroPad2d(1),
            ResidualBlock(256, 512, stride=1),
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 1, kernel_size=1, padding=0),
        )

        self.net.apply(init_weights)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        return self.net(x)


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.up = nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride=1),
            nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

        self.up.apply(init_weights)

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x (H * 2) x (W * 2)]

        """

        return self.up(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.res = ResidualBlock(in_channels, out_channels, stride=2)

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x (H / 2) x (W / 2)]

        """

        return self.res(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
    ):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(in_channels) if in_channels > 32 else nn.Identity(),
            nn.ReLU() if in_channels > 32 else nn.Identity(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.conv_block.apply(init_weights)
        self.conv_skip.apply(init_weights)

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


def init_weights(m: nn.Module):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0., 0.02)
