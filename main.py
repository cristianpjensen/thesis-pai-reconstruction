import pytorch_lightning as pl
from argparse import ArgumentParser
import pathlib
from models.pix2pix import Pix2Pix
from dataset import ImageDataModule


def main(hparams):
    pl.seed_everything(42, workers=True)

    model = Pix2Pix(
        name=hparams.name,
        l1_lambda=hparams.l1_lambda,
    )

    data_module = ImageDataModule(
        hparams.input_dir,
        hparams.target_dir,
        batch_size=hparams.batch_size,
    )

    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=hparams.epochs,
        log_every_n_steps=1,
        logger=pl.loggers.CSVLogger("logs"),
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("-i", "--input-dir", type=pathlib.Path)
    parser.add_argument("-t", "--target-dir", type=pathlib.Path)
    parser.add_argument("-l1", "--l1-lambda", default=50, type=int)
    parser.add_argument("-e", "--epochs", default=200, type=int)
    parser.add_argument("-bs", "--batch-size", default=2, type=int)
    args = parser.parse_args()

    main(args)
