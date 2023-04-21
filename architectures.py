from models.baseline import (
    GeneratorEncoderDecoder,
    GeneratorUNet,
    Patch70Discriminator,
)


# Generators and disciminators that can be used for the conditional GAN.
GENERATORS = {
    "BASELINE: U-net": GeneratorUNet,
    "Encoder-Decoder": GeneratorEncoderDecoder,
}

DISCRIMINATORS = {
    "BASELINE: PatchGAN 70x70": Patch70Discriminator,
}
