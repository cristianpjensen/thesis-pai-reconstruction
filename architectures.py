from models.baseline import (
    GeneratorEncoderDecoder,
    GeneratorUNet,
    Patch70Discriminator,
)


# Generators and disciminators that can be used for the conditional GAN.
GENERATORS = {
    "BASELINE: U-net": {"model": GeneratorUNet, "tag": "unet"},
    "Encoder-Decoder": {"model": GeneratorEncoderDecoder, "tag": "encdec"},
}

DISCRIMINATORS = {
    "BASELINE: PatchGAN 70x70": {
        "model": Patch70Discriminator,
        "tag": "patch70"
    },
}
