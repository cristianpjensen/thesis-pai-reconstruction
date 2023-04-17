import torch
from simple_term_menu import TerminalMenu
from pix2pix import (
    GeneratorEncoderDecoder,
    GeneratorUNet,
    PixelDiscriminator,
    Patch16Discriminator,
    Patch70Discriminator,
    Patch286Discriminator,
)
from train import train
import utils


# Generators and disciminators that can be used for the conditional GAN.
GENERATORS = {
    "Encoder-Decoder": GeneratorEncoderDecoder,
    "U-net": GeneratorUNet,
}

DISCRIMINATORS = {
    "PixelGAN": PixelDiscriminator,
    "PatchGAN 16x16": Patch16Discriminator,
    "PatchGAN 70x70": Patch70Discriminator,
    "PatchGAN 286x286": Patch286Discriminator,
}

# Hyperparameters
INPUT_SIZE = 256
NUM_EPOCHS = 200
L1_LAMBDA = 100


if torch.cuda.is_available():
    print("Using CUDA device...\n")
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    print("Using MPS device...\n")
    device = torch.device("mps")
else:
    print("Using CPU device...\n")
    device = torch.device("cpu")


def main():
    generator_enum = list(GENERATORS.keys())
    menu = TerminalMenu(generator_enum, title="Generator")
    generator_model = generator_enum[menu.show()]
    print(generator_model)

    discriminator_enum = list(DISCRIMINATORS.keys())
    menu = TerminalMenu(discriminator_enum, title="\nDiscriminator")
    discriminator_model = discriminator_enum[menu.show()]
    print(discriminator_model, "\n")

    train_loader, val_loader = utils.load_dataset(
        "./data/maps/",
        ["train", "val"],
        INPUT_SIZE,
    )

    train(train_loader, val_loader, generator_model, discriminator_model)


if __name__ == "__main__":
    main()
