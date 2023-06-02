import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from .utils import denormalize, init_weights, ssim, psnr


class GAN(pl.LightningModule):
    """Generative adversarial network.

    :param generator: Generator architecture.
    :param discriminator: Discriminator architecture.
    :param l1_lambda: Weight given to the L1 loss over the discriminator loss.

    :note: Make sure to set `self.example_input_array` and call
        `self.save_hyperparameters()`. Also make sure to pass the generator,
        discriminator, and L1 lambda to the `super().__init__(...)` call.

    """

    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        l1_lambda: float = 50,
    ):
        super().__init__()
        self.automatic_optimization = False

        self.generator = generator
        self.discriminator = discriminator
        self.l1_lambda = l1_lambda

        self.generator.apply(init_weights)
        self.discriminator.apply(init_weights)

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

        # We want to fool the discriminator into predicting all ones
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

        return opt_g, opt_d

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        # Train discriminator.
        self.toggle_optimizer(opt_d)

        input_, target = batch
        pred = self.generator(input_)

        target_label = self.discriminator(input_, target)
        pred_label = self.discriminator(input_, pred)
        d_loss = self.discriminator_loss(pred_label, target_label)

        self.log("d_loss", d_loss, prog_bar=True)

        self.discriminator.zero_grad(set_to_none=True)
        self.manual_backward(d_loss)
        opt_d.step()

        self.untoggle_optimizer(opt_d)

        # Train generator.
        self.toggle_optimizer(opt_g)

        pred = self.generator(input_)
        pred_label = self.discriminator(input_, pred)
        g_loss = self.generator_loss(pred, pred_label, target)

        g_ssim = ssim(denormalize(pred), denormalize(target))
        g_psnr = psnr(denormalize(pred), denormalize(target))

        self.log("g_loss", g_loss, prog_bar=True)
        self.log("train_ssim", g_ssim, prog_bar=True)
        self.log("train_psnr", g_psnr, prog_bar=True)

        self.generator.zero_grad(set_to_none=True)
        self.manual_backward(g_loss)
        opt_g.step()

        self.untoggle_optimizer(opt_g)

    def validation_step(self, batch, batch_idx):
        wandb_logger = self.loggers[1]

        input_, target = batch
        pred = self.forward(input_)

        for y in pred:
            wandb_logger.log_image(
                key="predictions",
                images=[denormalize(y)],
            )

        g_ssim = ssim(denormalize(pred), denormalize(target))
        g_psnr = psnr(denormalize(pred), denormalize(target))

        self.log("val_ssim", g_ssim, prog_bar=True)
        self.log("val_psnr", g_psnr, prog_bar=True)


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
