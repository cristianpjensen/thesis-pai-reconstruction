from simple_term_menu import TerminalMenu
from glob import glob
from rich.console import Console
from rich.prompt import Prompt
import re
from train import train
from architectures import GENERATORS, DISCRIMINATORS
import utils


console = Console()


def print_property(property: str, value: str):
    console.print(f"{property}: [bold blue]{value}")


def prompt_int(prompt: str, default: int):
    while True:
        value = Prompt.ask(
            f"{prompt} (default: {default})",
            default=default,
        )

        if value == default:
            return default

        if re.match("[-+]?\\d+$", value) is not None:
            return int(value)

        console.print(f"[red]{value} is not an integer!")


def main():
    # Let user make choices about architecture.

    # Choose generator.
    generator_enum = list(GENERATORS.keys())
    menu = TerminalMenu(generator_enum, title="Generator")
    generator_model = generator_enum[menu.show()]
    print_property("Generator", generator_model)

    # Choose discriminator.
    discriminator_enum = list(DISCRIMINATORS.keys())
    menu = TerminalMenu(discriminator_enum, title="Discriminator")
    discriminator_model = discriminator_enum[menu.show()]
    print_property("Discriminator", discriminator_model)

    console.print(
        "NOTE: The directory must be in `data/` and",
        "contain `train` and `val` directories.",
        style="grey50"
    )

    # Choose data location.
    directories = [x.split("/")[-1] for x in glob("./data/*")]
    menu = TerminalMenu(directories, title="Data directory")
    directory = directories[menu.show()]
    print_property("Data directory", directory)
    print()

    l1_lambda = prompt_int("L1 lambda", 100)
    num_epochs = prompt_int("Epochs", 200)
    print()

    # Load data.
    train_loader, val_loader = utils.load_dataset(
        f"./data/{directory}/",
        ["train", "val"],
        256,
    )

    # Train chosen architecture on specified data.
    train(
        train_loader,
        val_loader,
        generator_model,
        discriminator_model,
        l1_lambda=l1_lambda,
        num_epochs=num_epochs,
    )


if __name__ == "__main__":
    main()
