from models.pix2pix import (
    GeneratorEncoderDecoder,
    GeneratorUNet,
    PixelDiscriminator,
    Patch16Discriminator,
    Patch70Discriminator,
    Patch286Discriminator,
)
from models.attention_unet import Pix2PixAttentionUNet


# Generators and disciminators that can be used for the conditional GAN.
GENERATORS = {
    "Encoder-Decoder": GeneratorEncoderDecoder,
    "U-net": GeneratorUNet,
    "pix2pix with attention skip-connections": Pix2PixAttentionUNet
}

DISCRIMINATORS = {
    "PixelGAN": PixelDiscriminator,
    "PatchGAN 16x16": Patch16Discriminator,
    "PatchGAN 70x70": Patch70Discriminator,
    "PatchGAN 286x286": Patch286Discriminator,
}
