import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim,
)
import pytorch_lightning as pl
import math
from .utils import denormalize


class ModernUnetGAN(pl.LightningModule):
    """GAN.

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
        num_res_blocks: int = 3,
        channel_mults: tuple[int] = (1, 2, 4, 8,),
        att_mults: tuple[int] = (8,),
        num_heads: int = 4,
        dropout: float = 0.2,
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
            num_res_blocks=num_res_blocks,
            channel_mults=channel_mults,
            att_mults=att_mults,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.discriminator = Discriminator(
            in_channels,
            num_res_blocks=num_res_blocks,
            num_heads=num_heads,
        )

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
        num_res_blocks: int = 3,
        channel_mults: tuple[int] = (1, 2, 4, 8,),
        att_mults: tuple[int] = (8,),
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        downs = []
        for index, mult in enumerate(channel_mults):
            channels = mult * 64

            downs.append(
                Downsample(
                    in_channels,
                    channels,
                    num_res_blocks=num_res_blocks,
                    dropout=0,
                    use_attention=mult in att_mults,
                    num_heads=num_heads,
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
                    num_res_blocks=num_res_blocks,
                    dropout=dropout if (
                        index < 3 and
                        mult == max(channel_mults)
                    ) else 0,
                    use_attention=mult in att_mults,
                    num_heads=num_heads,
                )
            )

            # Multiply by 2 for the skip-connections.
            in_channels = channels * 2

        ups.append(
            Upsample(
                in_channels,
                64,
                num_res_blocks=num_res_blocks,
                dropout=0,
                use_attention=False,
            )
        )

        self.ups = nn.ModuleList(ups)
        self.out = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(
                64,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.Tanh(),
        )

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
    def __init__(
        self,
        in_channels: int = 3,
        num_res_blocks: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()

        # Input: 256 (pixels), 6 (channels)
        self.net = nn.Sequential(
            Downsample(
                in_channels * 2,
                64,
                num_res_blocks=num_res_blocks,
                use_attention=False,
            ),
            Downsample(
                64,
                128,
                num_res_blocks=num_res_blocks,
                use_attention=False,
            ),
            Downsample(
                128,
                256,
                num_res_blocks=num_res_blocks,
                use_attention=True,
                num_heads=num_heads,
            ),
            ResAttentionBlock(
                256,
                512,
                num_res_blocks=num_res_blocks,
                use_attention=True,
                num_heads=num_heads,
            ),
            ResBlock(512, 1),
        )

        self.net.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0., 0.02)

    def forward(self, x, y):
        """
        :param x: [N x in_channels x H x W]
        :param y: [N x in_channels x H x W]
        :returns: [N x 1 x OUT x OUT]

        """

        xy_concat = torch.cat([x, y], dim=1)
        return self.net(xy_concat)


class Upsample(nn.Module):
    """
    Residual-Attention-Upsample block.

    :param in_channels: Input channels.
    :param out_channels: Output channels.
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
        num_res_blocks: int = 2,
        dropout: float = 0,
        use_attention: bool = True,
        num_heads: int = 1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            ResAttentionBlock(
                in_channels,
                out_channels,
                num_res_blocks,
                dropout,
                use_attention,
                num_heads,
            ),
            nn.Upsample(scale_factor=2),
        )

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x (H / 2) x (W / 2)]

        """

        return self.net(x)


class Downsample(nn.Module):
    """
    Residual-Attention-Downsample block.

    :param in_channels: Input channels.
    :param out_channels: Output channels.
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
        num_res_blocks: int = 2,
        dropout: float = 0,
        use_attention: bool = True,
        num_heads: int = 1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            ResAttentionBlock(
                in_channels,
                out_channels,
                num_res_blocks,
                dropout,
                use_attention,
                num_heads,
            ),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x (H / 2) x (W / 2)]

        """

        return self.net(x)


class ResAttentionBlock(nn.Module):
    """
    Residual-Attention block.

    :param in_channels: Input channels.
    :param out_channels: Output channels.
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
        num_res_blocks: int = 2,
        dropout: float = 0,
        use_attention: bool = True,
        num_heads: int = 1,
    ):
        super().__init__()

        net = []
        for _ in range(num_res_blocks):
            net.append(
                ResBlock(
                    in_channels,
                    out_channels,
                    dropout=dropout,
                )
            )

            in_channels = out_channels

        net.append(
            AttentionBlock(
                out_channels,
                num_heads,
            ) if use_attention else nn.Identity()
        )

        self.net = nn.Sequential(*net)

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x H x W]

        """

        return self.net(x)


class AttentionBlock(nn.Module):
    """
    Attention block that allows spatial positions to attend to each other.

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
            nn.BatchNorm1d(channels),
            nn.Conv1d(
                channels,
                channels * 3,
                kernel_size=1,
            ),
        )

        self.out = nn.Conv2d(channels, channels, kernel_size=1)

        self.qkv_conv.apply(self.init_weights)
        self.out.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, 0., 0.02)

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
        scale = 1 / math.sqrt(self.channels_per_head)
        query_heads = query.reshape(batch_heads, self.channels_per_head, h * w)
        key_heads = key.reshape(batch_heads, self.channels_per_head, h * w)
        value_heads = value.reshape(batch_heads, self.channels_per_head, h * w)

        scores = torch.einsum(
            "bct,bcs->bts",
            query_heads * scale,
            key_heads * scale,
        )
        attention = F.softmax(scores.float(), dim=-1).type(scores.dtype)

        weighted = torch.einsum("bts,bcs->bct", attention, value_heads)
        weighted = weighted.reshape(n, -1, h, w)

        return self.out(weighted)


class ResBlock(nn.Module):
    """
    Residual block with a skip-connection.

    :param in_channels: Input channels.
    :param out_channels: Output channels.
    :param dropout: Dropout percentage.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0,
    ):
        super().__init__()

        self.blocks = nn.Sequential(
            Block(in_channels, out_channels),
            Block(out_channels, out_channels, dropout=dropout),
        )

        # Sets the skip connection to the correct amount of channels
        self.skip_connection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        ) if in_channels != out_channels else nn.Identity()

        self.skip_connection.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0., 0.02)

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :return: [N x out_channels x H x W]

        """

        return self.blocks(x) + self.skip_connection(x)


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
            nn.BatchNorm2d(in_channels) if in_channels >= 32 else nn.Identity(),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
        )

        self.block.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0., 0.02)

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x H x W]

        """

        return self.block(x)
