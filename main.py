import pytorch_lightning as pl
from argparse import ArgumentParser
import pathlib
from models.pix2pix import Pix2Pix
from models.palette import Palette
from dataset import ImageDataModule


def main(hparams):
    pl.seed_everything(42, workers=True)

    model = None
    match hparams.model:
        case "pix2pix":
            model = Pix2Pix(l1_lambda=hparams.l1_lambda)

        case "palette":
            model = Palette(
                in_channels=3,
                out_channels=3,
                inner_channels=64,
                channel_mults=(1, 2, 4, 8),
                num_res_blocks=3,
                attention_res=(4, 8),
                num_heads=1,
                dropout=0.,
            )

    if model is None:
        raise ValueError(f"Incorrect model name ({hparams.model})")

    data_module = ImageDataModule(
        hparams.input_dir,
        hparams.target_dir,
        batch_size=hparams.batch_size,
        val_size=0.3,
    )

    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=hparams.epochs,
        log_every_n_steps=10,
        logger=pl.loggers.CSVLogger("logs", name=hparams.name),
        precision=hparams.precision,
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("name")
    parser.add_argument(
        "-i",
        "--input-dir",
        type=pathlib.Path,
        help="Input images directory path",
    )
    parser.add_argument(
        "-t",
        "--target-dir",
        type=pathlib.Path,
        help="Target images directory path",
    )
    parser.add_argument("-l1", "--l1-lambda", default=50, type=int)
    parser.add_argument("-e", "--epochs", default=200, type=int)
    parser.add_argument("-bs", "--batch-size", default=2, type=int)
    parser.add_argument(
        "-p",
        "--precision",
        default="32",
        help="Floating-point precision"
    )
    parser.add_argument(
        "-m",
        "--model",
        default="pix2pix",
        choices=["pix2pix", "palette"],
    )
    args = parser.parse_args()

    main(args)
