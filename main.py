import pytorch_lightning as pl
import argparse
from argparse import ArgumentParser
import pathlib
from models.pix2pix import Pix2Pix
from models.palette import Palette
from models.transgan import TransGAN
from models.resnet import ResNetGAN
from models.attention_unet import AttentionUNet
from dataset import ImageDataModule
from callbacks.ema import EMACallback


def main(hparams):
    pl.seed_everything(42, workers=True)

    model = None
    match hparams.model:
        case "pix2pix":
            model = Pix2Pix(
                in_channels=1 if hparams.grayvalues else 3,
                out_channels=1 if hparams.grayvalues else 3,
                l1_lambda=hparams.l1_lambda,
            )

        case "palette":
            model = Palette(
                in_channels=1 if hparams.grayvalues else 3,
                out_channels=1 if hparams.grayvalues else 3,
                inner_channels=64,
                channel_mults=(1, 2, 4, 8),
                num_res_blocks=2,
                attention_res=(8,),
                num_heads=4,
                dropout=0.2,
                output_diffusion_videos=True,
                use_guided_diffusion=True,
            )

        case "transgan":
            model = TransGAN(
                in_channels=1 if hparams.grayvalues else 3,
                out_channels=1 if hparams.grayvalues else 3,
                l1_lambda=hparams.l1_lambda,
            )

        case "resnet":
            model = ResNetGAN(
                in_channels=1 if hparams.grayvalues else 3,
                out_channels=1 if hparams.grayvalues else 3,
                l1_lambda=hparams.l1_lambda,
            )

        case "attention-unet":
            model = AttentionUNet(
                in_channels=1 if hparams.grayvalues else 3,
                out_channels=1 if hparams.grayvalues else 3,
                l1_lambda=hparams.l1_lambda,
            )

    if model is None:
        raise ValueError(f"Incorrect model name ({hparams.model})")

    data_module = ImageDataModule(
        hparams.input_dir,
        hparams.target_dir,
        batch_size=hparams.batch_size,
        val_size=hparams.val_size,
        normalize=hparams.model == "palette",
        grayvalues=hparams.grayvalues,
    )

    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=hparams.epochs,
        log_every_n_steps=10,
        check_val_every_n_epoch=hparams.val_epochs,
        logger=pl.loggers.CSVLogger("logs", name=hparams.name),
        precision=hparams.precision,
        callbacks=[EMACallback(0.9999)],
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
    parser.add_argument("-vs", "--val-size", default=0.3, type=float)
    parser.add_argument(
        "-ve",
        "--val-epochs",
        default=10,
        help="Validation run every n epochs.",
        type=int
    )
    parser.add_argument(
        "-p",
        "--precision",
        default="32",
        help="Floating-point precision"
    )
    parser.add_argument(
        "--grayvalues",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to turn the images into grayvalue images."
    )
    parser.add_argument(
        "-m",
        "--model",
        default="pix2pix",
        choices=["pix2pix", "palette", "transgan", "resnet", "attention-unet"],
    )
    args = parser.parse_args()

    main(args)
