from simple_term_menu import TerminalMenu
from natsort import natsorted
from rich.console import Console
from rich.prompt import Prompt
from datetime import datetime
import re
import os
from train import train
from architectures import GENERATORS, DISCRIMINATORS


console = Console()


def print_property(property: str, value: str):
    console.print(f"{property}: [sea_green1]{value}")


def prompt_int(prompt: str, default: int):
    while True:
        value = Prompt.ask(
            f"{prompt}",
            default=str(default),
        )

        if value == str(default):
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
        "\nNOTE: The data must be in `data/`.",
        style="grey50"
    )

    # Choose data location.
    directories = natsorted([
        dir for dir in os.listdir("./data")
        if os.path.isdir(os.path.join("./data", dir))
    ])
    menu = TerminalMenu(directories, title="Data directory")
    data_dir = directories[menu.show()]
    print_property("Data directory", data_dir)

    # Choose input directory.
    directories = natsorted([
        dir for dir in os.listdir(os.path.join("./data", data_dir))
        if os.path.isdir(os.path.join("./data", data_dir, dir))
    ])
    menu = TerminalMenu(directories, title="Input directory")
    input_dir = directories[menu.show()]
    print_property("Input directory", input_dir)

    # Choose label directory.
    directories = natsorted([
        dir for dir in os.listdir(os.path.join("./data", data_dir))
        if os.path.isdir(os.path.join("./data", data_dir, dir))
    ])
    menu = TerminalMenu(directories, title="Label directory")
    label_dir = directories[menu.show()]
    print_property("Label directory", label_dir)

    # Choose name for data that will be used in the name of the output file.
    data_name = Prompt.ask(
        "Data name used for output filename",
        default=data_dir
    )
    print()

    input_dir = os.path.join("./data", data_dir, input_dir)
    label_dir = os.path.join("./data", data_dir, label_dir)

    # Prompt hyperparameters.
    l1_lambda = prompt_int("L1 lambda", 50)
    num_epochs = prompt_int("Epochs", 20)
    print()

    id = datetime.now().strftime("%Y%m%d%H%M")
    filename = "%s_gen:%s_dis:%s_data:%s_l1:%d_epochs:%d" % (
        id,
        GENERATORS[generator_model]["tag"],
        DISCRIMINATORS[discriminator_model]["tag"],
        data_name,
        l1_lambda,
        num_epochs,
    )

    console.print(f'Output filename: [bold orange3]{filename}')
    print()

    # Train chosen architecture on specified data.
    train(
        input_dir,
        label_dir,
        generator_model,
        discriminator_model,
        l1_lambda=l1_lambda,
        num_epochs=num_epochs,
        filename=filename,
    )


if __name__ == "__main__":
    main()
