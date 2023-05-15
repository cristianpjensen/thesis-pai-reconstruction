import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim,
)
import pytorch_lightning as pl


class ResNetGAN(pl.LightningModule):
    def __init__(self, l1_lambda: float):
        super().__init__()
        self.save_hyperparameters()
        self.example_input_array = torch.Tensor(32, 3, 256, 256)
        self.automatic_optimization = False

        self.generator = UNet()
        self.discriminator = Discriminator()

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


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super().__init__()

        # Input: 256 (pixels), 3 (channels)
        self.down1 = Downsample(in_channels, 64, use_attention=False)  # 128, 64
        self.down2 = Downsample(64, 128, use_attention=False)   # 64, 128
        self.down3 = Downsample(128, 256, use_attention=False)  # 32, 256
        self.down4 = Downsample(256, 512)  # 16, 512
        self.down5 = Downsample(512, 512)  # 8, 512
        self.down6 = Downsample(512, 512)  # 4, 512
        self.down7 = Downsample(512, 512)  # 2, 512
        self.down8 = Downsample(512, 512)  # 1, 512

        # Skip-connections are added here, so the amount of channels of the
        # output is doubled.
        self.up8 = Upsample(512, 512, dropout=0.5)   # 2, 1024
        self.up7 = Upsample(1024, 512, dropout=0.5)  # 4, 1024
        self.up6 = Upsample(1024, 512, dropout=0.5)  # 8, 1024
        self.up5 = Upsample(1024, 512)  # 16, 1024
        self.up4 = Upsample(1024, 256, use_attention=False)  # 32, 512
        self.up3 = Upsample(512, 128, use_attention=False)   # 64, 256
        self.up2 = Upsample(256, 64, use_attention=False)    # 128, 128
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


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()

        self.net = nn.Sequential(
            Downsample(in_channels * 2, 64, use_attention=False),  # 128, 64
            Downsample(64, 128, use_attention=False),   # 64, 128
            Downsample(128, 256),  # 32, 256
            nn.ZeroPad2d(1),  # 33, 512
            nn.Conv2d(
                256,
                1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),  # 31, 1
        )

        self.net.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0., 0.02)

    def forward(self, x, y):
        xy_concat = torch.cat((x, y), dim=1)
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
            nn.ConvTranspose2d(
                out_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.SiLU(),
        )

        self.net.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight, 0., 0.02)

    def forward(self, x):
        """
        :param x: [N x in_channels x H x W]
        :returns: [N x out_channels x (H * 2) x (W * 2)]

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
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.SiLU(),
        )

        self.net.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0., 0.02)

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
        num_res_blocks: int,
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
            SelfAttention(
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
            nn.ReLU(),
            Block(out_channels, out_channels, dropout=dropout),
        )

        # Sets the skip connection to the correct amount of channels
        self.skip_connection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
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
            nn.GroupNorm(
                32,
                in_channels
            ) if in_channels > 32 else nn.Identity(),
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


if __name__ == "__main__":
    unet = UNet(3, 3)

    x = torch.randn((2, 3, 256, 256))
    y = unet(x)

    print(y.shape)

    dis = Discriminator(3)
    z = dis(x, y)
    print(z.shape)
