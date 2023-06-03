import torch
import pytorch_lightning as pl
import argparse
from argparse import ArgumentParser
import pathlib
from models.pix2pix import Pix2Pix
from models.palette import Palette
from models.attention_unet import AttentionUnetGAN
from models.res_unet import ResUnetGAN
from models.trans_unet import TransUnetGAN
from dataset import ImageDataModule
from callbacks.ema import EMACallback


torch.set_float32_matmul_precision("medium")


def main(hparams):
    channel_mults = [int(x) for x in hparams.channel_mults.split(",")]
    att_mults = [int(x) for x in hparams.attention_mults.split(",")]

    model = None
    match hparams.model:
        case "pix2pix":
            model = Pix2Pix(
                in_channels=3,
                out_channels=3,
                channel_mults=channel_mults,
                dropout=hparams.dropout,
                l1_lambda=hparams.l1_lambda,
            )

        case "attention_unet":
            model = AttentionUnetGAN(
                in_channels=3,
                out_channels=3,
                channel_mults=channel_mults,
                dropout=hparams.dropout,
                l1_lambda=hparams.l1_lambda,
            )

        case "palette":
            model = Palette(
                in_channels=3,
                out_channels=3,
                inner_channels=64,
                channel_mults=channel_mults,
                attention_res=att_mults,
                dropout=hparams.dropout,
                num_res_blocks=2,
                num_heads=4,
            )

        case "res18_unet":
            model = ResUnetGAN(
                in_channels=3,
                out_channels=3,
                res_type="18",
                channel_mults=channel_mults,
                dropout=hparams.dropout,
                l1_lambda=hparams.l1_lambda,
            )

        case "res50_unet":
            model = ResUnetGAN(
                in_channels=3,
                out_channels=3,
                res_type="50",
                channel_mults=channel_mults,
                dropout=hparams.dropout,
                l1_lambda=hparams.l1_lambda,
            )

        case "resv2_unet":
            model = ResUnetGAN(
                in_channels=3,
                out_channels=3,
                res_type="v2",
                channel_mults=channel_mults,
                dropout=hparams.dropout,
                l1_lambda=hparams.l1_lambda,
            )

        case "resnext_unet":
            model = ResUnetGAN(
                in_channels=3,
                out_channels=3,
                res_type="next",
                channel_mults=channel_mults,
                dropout=hparams.dropout,
                l1_lambda=hparams.l1_lambda,
            )

        case "trans_unet":
            model = TransUnetGAN(
                in_channels=3,
                out_channels=3,
                image_size=256,
                channel_mults=channel_mults,
                patch_size=2,
                num_heads=8,
                l1_lambda=hparams.l1_lambda,
            )

        case _:
            raise ValueError(f"Incorrect model name ({hparams.model})")

    data_module = ImageDataModule(
        hparams.input_dir,
        hparams.target_dir,
        batch_size=hparams.batch_size,
        val_size=hparams.val_size,
        normalize=True,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=3,
        monitor="val_ssim",
        mode="max",
        filename="checkpoint-{epoch:02d}-{val_ssim:.2f}-{val_psnr:.2f}",
    )

    wandb_logger = pl.loggers.WandbLogger(
        project="pat-reconstruction",
        name=hparams.name,
    )
    csv_logger = pl.loggers.CSVLogger("logs", name=hparams.name)

    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        max_steps=hparams.steps,
        log_every_n_steps=10,
        check_val_every_n_epoch=hparams.val_epochs,
        logger=[csv_logger, wandb_logger],
        precision=hparams.precision,
        callbacks=[
            EMACallback(0.9999),
            checkpoint_callback,
        ] if hparams.ema else [checkpoint_callback],
        benchmark=True,
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
    parser.add_argument("-s", "--steps", default=-1, type=int)
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
        "--ema",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Whether to use EMA weight updating."
    )
    parser.add_argument(
        "--channel-mults",
        default="1,2,4,8,8,8,8,8",
        help="""
            Defines the U-net architecture's depth and width. Should be
            comma-separated powers of 2.
        """,
    )
    parser.add_argument(
        "--attention-mults",
        default="8",
        help="""
            At what channel multiples attention should be used, if the model
            supports it. Should be comma-separated powers of 2.
        """,
    )
    parser.add_argument(
        "--dropout",
        default=0.0,
        type=float,
    )
    parser.add_argument(
        "-m",
        "--model",
        default="pix2pix",
        choices=[
            "pix2pix",
            "attention_unet",
            "res18_unet",
            "res50_unet",
            "resv2_unet",
            "resnext_unet",
            "trans_unet",
            "palette",
        ],
    )
    args = parser.parse_args()

    main(args)
