import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Union
from .gan import GAN, Discriminator


class SwinUnetGAN(GAN):
    """Implementation of a GAN with the Swin Transformer U-net.

    :param in_channels: Input channels that can vary if the images are
        grayscale or color.
    :param out_channels: Input channels that can vary if the images are
        grayscale or color.
    :param l1_lambda: How much the L1 loss should be weighted in the loss
        function.

    :input: [N x in_channels x H x W]
    :output: [N x out_channels x H x W]

    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        l1_lambda: float = 50,
        dropout: float = 0.05,
    ):
        generator = SwinUnet(
            in_channels,
            out_channels,
            image_size=256,
            attention_dropout=dropout,
            stochastic_dropout=dropout,
            dropout=dropout,
        )
        discriminator = Discriminator(in_channels)

        super().__init__(generator, discriminator, l1_lambda=l1_lambda)

        self.example_input_array = torch.Tensor(2, in_channels, 256, 256)
        self.save_hyperparameters()


class SwinUnet(nn.Module):
    """Swin U-net (Cao et al. 2021).

    :param in_channels: Input channels that can vary if the images are
        grayscale or color.
    :param out_channels: Input channels that can vary if the images are
        grayscale or color.

    :input: [N x in_channels x H x W]
    :output: [N x out_channels x H x W]

    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        image_size: int = 256,
        window_size: int = 8,
        num_heads: (int, int, int, int) = (2, 4, 8, 16),
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_dropout: float = 0.0,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(in_channels, patch_size=4, embed_dim=64)

        self.encode1 = EncoderBlock(
            64,
            64,
            window_size=window_size,
            num_heads=num_heads[0],
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_dropout=stochastic_dropout,
        )
        self.encode2 = EncoderBlock(
            128,
            32,
            window_size=window_size,
            num_heads=num_heads[1],
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_dropout=stochastic_dropout,
        )
        self.encode3 = EncoderBlock(
            256,
            16,
            window_size=window_size,
            num_heads=num_heads[2],
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_dropout=stochastic_dropout,
        )
        self.bottleneck = SwinTransformerBlock(
            512,
            8,
            window_size=window_size,
            num_heads=num_heads[3],
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_dropout=stochastic_dropout,
        )

        self.decode3 = DecoderBlock(
            512,
            8,
            window_size=window_size,
            num_heads=num_heads[2],
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_dropout=stochastic_dropout,
        )
        self.concat_back_dim3 = nn.Linear(512, 256)
        self.decode2 = DecoderBlock(
            256,
            16,
            window_size=window_size,
            num_heads=num_heads[1],
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_dropout=stochastic_dropout,
        )
        self.concat_back_dim2 = nn.Linear(256, 128)
        self.decode1 = DecoderBlock(
            128,
            32,
            window_size=window_size,
            num_heads=num_heads[0],
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_dropout=stochastic_dropout,
        )
        self.concat_back_dim1 = nn.Linear(128, 64)

        self.patch_expand_4x = PatchExpand(64, scale=4)
        self.out = nn.Sequential(
            nn.Conv2d(64, out_channels, kernel_size=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.patch_embed(x)

        e2 = self.encode1(e1)
        e3 = self.encode2(e2)
        e4 = self.encode3(e3)

        d4 = self.bottleneck(e4)

        d3 = self.decode3(d4)
        d3 = torch.cat([d3, e3], dim=-1)
        d3 = self.concat_back_dim3(d3)

        d2 = self.decode2(d3)
        d2 = torch.cat([d2, e2], dim=-1)
        d2 = self.concat_back_dim2(d2)

        d1 = self.decode1(d2)
        d1 = torch.cat([d1, e1], dim=-1)
        d1 = self.concat_back_dim1(d1)

        y = self.patch_expand_4x(d1)
        y = y.permute(0, 3, 1, 2).contiguous()
        return self.out(y)


class PatchEmbed(nn.Module):
    """Patch partition -> Linear embedding.

    :param in_channels: Input channels.
    :param patch_size: Patch size.
    :param embed_dim: Embedding dimensionality.

    :input: [N x C x H x W]
    :output: [N x (H / patch_size) x (W / patch_size) x embed_dim]

    """

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 4,
        embed_dim: int = 64,
    ):
        super().__init__()

        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        n, c, h, w = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x.reshape(n, h // self.patch_size, w // self.patch_size, -1)


class EncoderBlock(nn.Module):
    """Encoder block that downsamples the input by 2.

    :param channels: Input channels.
    :param image_size: Image size.
    :param window_size: Window size.
    :param num_heads: Number of attention heads.
    :param mlp_ratio: Ratio of MLP hidden dimensionality.
    :param dropout: Dropout ratio in MLP layers.
    :param attention_dropout: Dropout ratio after attention mechanism.
    :param stochastic_dropout: Dropout ratio after (S)W-MSA layers.

    :input: [N x image_size x image_size x channels]
    :output: [N x (image_size / 2) x (image_size / 2) x (channels * 2)]

    """

    def __init__(
        self,
        channels: int,
        image_size: int,
        window_size: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_dropout: float = 0.0,
    ):
        super().__init__()

        self.encode = nn.Sequential(
            SwinTransformerBlock(
                channels,
                image_size,
                window_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                stochastic_dropout=stochastic_dropout,
            ),
            PatchMerge(channels),
        )

    def forward(self, x):
        return self.encode(x)


class DecoderBlock(nn.Module):
    """Decoder block that upsamples the input by 2.

    :param channels: Input channels.
    :param image_size: Image size.
    :param window_size: Window size.
    :param num_heads: Number of attention heads.
    :param mlp_ratio: Ratio of MLP hidden dimensionality.
    :param dropout: Dropout ratio in MLP layers.
    :param attention_dropout: Dropout ratio after attention mechanism.
    :param stochastic_dropout: Dropout ratio after (S)W-MSA layers.

    :input: [N x image_size x image_size x channels]
    :output: [N x (image_size * 2) x (image_size * 2) x (channels / 2)]

    """

    def __init__(
        self,
        channels: int,
        image_size: int,
        window_size: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_dropout: float = 0.0,
    ):
        super().__init__()

        self.decode = nn.Sequential(
            SwinTransformerBlock(
                channels,
                image_size,
                window_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout,
                stochastic_dropout=stochastic_dropout,
            ),
            PatchExpand(channels)
        )

    def forward(self, x):
        return self.decode(x)


class PatchMerge(nn.Module):
    """Patch merging layer.

    :param channels: Input channels.

    :input: [N x H x W x channels]
    :output: [N x (H / 2) x (W / 2) x (2 * channels)]

    """

    def __init__(self, channels: int):
        super().__init__()

        self.channels = channels

        self.merge = nn.Sequential(
            nn.LayerNorm(4 * channels),
            nn.Linear(4 * channels, 2 * channels, bias=False),
        )

    def forward(self, x):
        n, h, w, c = x.shape

        # Make all possible x that skips every other value
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        # Concatenate
        x = torch.cat([x0, x1, x2, x3], dim=-1)

        x = x.view(n, -1, 4 * c)
        x_merged = self.merge(x)

        return x_merged.reshape(n, h // 2, w // 2, -1)


class PatchExpand(nn.Module):
    """Patch expansion layer.

    :param channels: Input channels.
    :param scale: How much to scale the input up by (2 or 4).

    :input: [N x H x W x channels]
    :output: [N x (H * 2) x (W * 2) x channels]

    """

    def __init__(self, channels: int, scale: Union[2, 4] = 2):
        super().__init__()

        self.channels = channels
        self.scale = scale

        if scale == 2:
            self.expand = nn.Linear(channels, 2 * channels, bias=False)
            self.norm = nn.LayerNorm(channels // 2)

        if scale == 4:
            self.expand = nn.Linear(channels, 16 * channels, bias=False)
            self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        n, h, w, c = x.shape
        x = x.view(n, -1, c)
        x = self.expand(x)

        x = x.view(n, h, w, 2 * c) if (
            self.scale == 2
        ) else x.view(n, h, w, 16 * c)
        x = rearrange(
            x,
            "n h w (p1 p2 c) -> n (h p1) (w p2) c",
            p1=self.scale,
            p2=self.scale,
        )

        x = x.view(n, -1, c // 2) if self.scale == 2 else x.view(n, -1, c)
        x = self.norm(x)
        x = x.view(n, h * 2, w * 2, c // 2) if (
            self.scale == 2
        ) else x.view(n, h * 4, w * 4, c)

        return x


class SwinTransformerBlock(nn.Module):
    """Swin transformer block (Cao et al. 2021).

    :param channels: Number of input channels.
    :param image_size: Height and width of input image.
    :param window_size: Window size.
    :param num_heads: Number of attention heads.
    :param mlp_ratio: Multiplicator of hidden dimension in MLP layers.
    :param dropout: Dropout ratio in MLP layer.
    :param attention_dropout: Dropout ratio after attention mechanism.
    :param stochastic_dropout: Stochastic dropout ratio.

    :input: [N x H x W x C]
    :output: [N x H x W x C]

    """

    def __init__(
        self,
        channels: int,
        image_size: int,
        window_size: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_dropout: float = 0.0,
    ):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.LayerNorm(channels),
            WindowMSA(
                channels,
                window_size,
                image_size,
                num_heads,
                attention_dropout=attention_dropout,
                output_dropout=stochastic_dropout,
            ),
        )

        self.block2 = nn.Sequential(
            nn.LayerNorm(channels),
            MLP(
                channels,
                int(channels * mlp_ratio),
                channels,
                dropout=dropout,
            ),
        )

        self.block3 = nn.Sequential(
            nn.LayerNorm(channels),
            ShiftedWindowMSA(
                channels,
                window_size,
                image_size,
                num_heads,
                attention_dropout=attention_dropout,
                output_dropout=stochastic_dropout,
            ),
        )

        self.block4 = nn.Sequential(
            nn.LayerNorm(channels),
            MLP(channels, int(channels * mlp_ratio), channels),
        )

    def forward(self, x):
        x1_hat = self.block1(x) + x
        x1 = self.block2(x1_hat) + x1_hat
        x2_hat = self.block3(x1) + x1
        x2 = self.block4(x2_hat) + x2_hat

        return x2


class WindowMSA(nn.Module):
    """Window-based multi-head self attention (W-MSA).

    :param channels: Number of input channels.
    :param window_size: Window size.
    :param image_size: Image size.
    :param num_heads: Number of attention heads.
    :param attention_dropout: Dropout ratio after attention mechanism.
    :param output_dropout: Dropout ratio of output.

    :input: [N x image_size x image_size x channels]
    :output: [N x image_size x image_size x channels]

    """

    def __init__(
        self,
        channels: int,
        window_size: int,
        image_size: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        output_dropout: float = 0.0,
    ):
        super().__init__()

        self.window_size = window_size
        self.image_size = image_size
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5

        self.qkv = nn.Linear(channels, channels * 3)
        self.attention_drop = nn.Dropout(attention_dropout)

        self.out = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Dropout(output_dropout),
        )

        self.pos_relative_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size - 1) * (2 * self.window_size - 1),
                num_heads,
            )
        )

        h_coords = torch.arange(self.window_size)
        w_coords = torch.arange(self.window_size)
        coords = torch.stack(
            torch.meshgrid([h_coords, w_coords], indexing="ij"),
        )
        coords_flat = torch.flatten(coords, 1)
        coords_relative = coords_flat.unsqueeze(2) - coords_flat.unsqueeze(1)
        coords_relative = coords_relative.permute(1, 2, 0).contiguous()
        coords_relative[:, :, 0] += 3 * self.window_size - 2
        coords_relative[:, :, 1] += self.window_size - 1

        self.register_buffer(
            "pos_rel_index",
            coords_relative.sum(-1).view(-1),
        )

    def forward(self, x):
        b, h, w, c = x.shape
        d = int(self.window_size * self.window_size)

        windows = window_partition(x, self.window_size)
        windows = windows.view(-1, d, c)
        n = windows.shape[0]

        # Get query, key, and value
        qkv = self.qkv(windows)
        qkv = qkv.reshape(n, d, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv
        query = query * self.scale

        # Get attention values with relative position biases
        attention = (query @ key.transpose(-2, -1))
        pos_relative_bias = self.pos_relative_bias_table[self.pos_rel_index]
        pos_relative_bias = pos_relative_bias.view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )
        pos_relative_bias = pos_relative_bias.permute(2, 0, 1).contiguous()
        pos_relative_bias = pos_relative_bias.unsqueeze(0)
        attention += pos_relative_bias
        attention = F.softmax(attention, dim=-1)
        attention = self.attention_drop(attention)

        # Return value weighted by its attention score
        weighted = (attention @ value).transpose(1, 2)
        weighted = weighted.reshape(n, d, c)
        weighted = self.out(weighted)

        # Reconstruct image from windows
        output = weighted.reshape(n, self.window_size, self.window_size, c)
        output = window_reverse(weighted, self.window_size, self.image_size)

        return output


class ShiftedWindowMSA(nn.Module):
    """Shifted window-based multi-head self attention (SW-MSA).

    :param channels: Number of input channels.
    :param window_size: Window size.
    :param image_size: Image size.
    :param num_heads: Number of attention heads.
    :param attention_dropout: Dropout ratio after attention mechanism.
    :param output_dropout: Dropout ratio of output.

    :input: [N x image_size x image_size x channels]
    :output: [N x image_size x image_size x channels]

    """

    def __init__(
        self,
        channels: int,
        window_size: int,
        image_size: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        output_dropout: float = 0.0,
    ):
        super().__init__()

        self.window_size = window_size
        self.image_size = image_size
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5

        self.qkv = nn.Linear(channels, channels * 3)
        self.attention_drop = nn.Dropout(attention_dropout)

        self.out = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Dropout(output_dropout),
        )

        self.pos_relative_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size - 1) * (2 * self.window_size - 1),
                num_heads,
            )
        )

        h_coords = torch.arange(self.window_size)
        w_coords = torch.arange(self.window_size)
        coords = torch.stack(
            torch.meshgrid([h_coords, w_coords], indexing="ij"),
        )
        coords_flat = torch.flatten(coords, 1)
        coords_relative = coords_flat.unsqueeze(2) - coords_flat.unsqueeze(1)
        coords_relative = coords_relative.permute(1, 2, 0).contiguous()
        coords_relative[:, :, 0] += 3 * self.window_size - 2
        coords_relative[:, :, 1] += self.window_size - 1

        self.register_buffer(
            "pos_rel_index",
            coords_relative.sum(-1).view(-1),
        )

        shift_size = window_size // 2
        img_mask = torch.zeros((1, image_size, image_size, 1))

        img_mask[:, :-window_size, :-window_size, :] = 0
        img_mask[:, :-window_size, -window_size:-shift_size, :] = 1
        img_mask[:, :-window_size, -shift_size:, :] = 2
        img_mask[:, -window_size:-shift_size, :-window_size, :] = 3
        img_mask[:, -window_size:-shift_size, -window_size:-shift_size, :] = 4
        img_mask[:, -window_size:-shift_size, -shift_size:, :] = 5
        img_mask[:, -shift_size:, :-window_size, :] = 6
        img_mask[:, -shift_size:, -window_size:-shift_size, :] = 7
        img_mask[:, -shift_size:, -shift_size:, :] = 8

        mask_windows = window_partition(img_mask, window_size)
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attention_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attention_mask = attention_mask.masked_fill(
            attention_mask != 0,
            float(-100.0),
        )
        attention_mask = attention_mask.masked_fill(
            attention_mask == 0,
            float(0.0),
        )

        self.register_buffer("attention_mask", attention_mask)

    def forward(self, x):
        b, h, w, c = x.shape
        d = int(self.window_size * self.window_size)

        shift_size = self.window_size // 2
        x_shifted = torch.roll(x, (-shift_size, -shift_size), dims=(1, 2))

        windows = window_partition(x_shifted, self.window_size)
        windows = windows.view(-1, self.window_size * self.window_size, c)
        n = windows.shape[0]

        # Get query, key, and value
        qkv = self.qkv(windows)
        qkv = qkv.reshape(n, d, 3, self.num_heads, c // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv
        query = query * self.scale

        # Get attention values with relative position biases
        attention = (query @ key.transpose(-2, -1))
        pos_relative_bias = self.pos_relative_bias_table[self.pos_rel_index]
        pos_relative_bias = pos_relative_bias.view(
            self.window_size * self.window_size,
            self.window_size * self.window_size,
            -1,
        )
        pos_relative_bias = pos_relative_bias.permute(2, 0, 1).contiguous()
        pos_relative_bias = pos_relative_bias.unsqueeze(0)
        attention += pos_relative_bias

        # Add shifted mask
        num_windows = self.attention_mask.shape[0]
        attention = attention.view(n // num_windows, num_windows, self.num_heads, d, d)
        attention += self.attention_mask.unsqueeze(1).unsqueeze(0)
        attention = attention.view(-1, self.num_heads, d, d)

        attention = F.softmax(attention, dim=-1)
        attention = self.attention_drop(attention)

        # Return value weighted by its attention score
        weighted = (attention @ value).transpose(1, 2)
        weighted = weighted.reshape(n, d, c)
        weighted = self.out(weighted)

        # Reconstruct image from windows
        output = weighted.reshape(n, self.window_size, self.window_size, c)
        output = window_reverse(weighted, self.window_size, self.image_size)

        # Shift back
        output = torch.roll(output, (shift_size, shift_size), dims=(1, 2))

        return output


class MLP(nn.Module):
    """Multi-layer perceptron.

    :param in_dims: Input dimensions.
    :param hidden_dims: Dimensions of hidden layer.
    :param out_dims: Output dimensions.
    :param dropout: Dropout percentage.

    :input: [N x H x W x in_dims]
    :output: [N x H x W x out_dims]

    :input: [N x D x in_dims]
    :output: [N x D x out_dims]

    """

    def __init__(
        self,
        in_dims: int,
        hidden_dims: int,
        out_dims: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dims, hidden_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, in_dims),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        n, h, w, c = x.shape
        x = x.view(n, h * w, c)
        x = self.net(x)
        return x.reshape(n, h, w, -1)


def window_partition(x: torch.Tensor, window_size: int):
    """Partitions an image into windows of size [window_size x window_size].

    :param x: [N x H x W x C]
    :param window_size: Window size.
    :returns: [(N x num_windows) x window_size x window_size x C]

    """

    n, h, w, c = x.shape
    x = x.view(n, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, c)

    return windows


def window_reverse(windows: torch.Tensor, window_size: int, image_size: int):
    """Reconstructs the image from its windows of size `window_size`.

    :param x: [(N x num_windows) x window_size x window_size x C]
    :param window_size: Window size.
    :returns: [N x H x W x C]

    """

    channels = windows.shape[-1]
    windows = windows.view(
        -1,
        image_size // window_size,
        image_size // window_size,
        window_size,
        window_size,
        channels,
    )
    x = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(-1, image_size, image_size, channels)

    return x
