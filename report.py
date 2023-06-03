import torch
from torchmetrics.functional import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim,
    mean_squared_error as mse,
)
from torchvision.io import write_png
from argparse import ArgumentParser
import pathlib
import os
from models.pix2pix import Pix2Pix
from models.palette import Palette
from models.attention_unet import AttentionUnetGAN
from models.res_unet import ResUnetGAN
from models.vit_unet import ViTUnetGAN
from reporting.depth_ssim import depth_ssim
from dataset import ImageDataModule
from models.utils import denormalize, to_int


def main(hparams):
    model = None
    match hparams.model:
        case "pix2pix":
            model = Pix2Pix.load_from_checkpoint(hparams.checkpoint)
            model.freeze()

        case "palette":
            model = Palette.load_from_checkpoint(hparams.checkpoint)
            model.freeze()

        case "attention_unet":
            model = AttentionUnetGAN.load_from_checkpoint(hparams.checkpoint)
            model.freeze()

        case "res18_unet" | "res50_unet" | "resv2_unet" | "resnext_unet":
            model = ResUnetGAN.load_from_checkpoint(hparams.checkpoint)
            model.freeze()

        case "vit_unet":
            model = ViTUnetGAN.load_from_checkpoint(hparams.checkpoint)
            model.freeze()

        case _:
            raise ValueError(f"Incorrect model name ({hparams.model})")

    data_module = ImageDataModule(
        hparams.input_dir,
        hparams.target_dir,
        batch_size=hparams.batch_size,
        normalize=True,
    )
    data_module.setup("predict")
    dataloader = data_module.predict_dataloader()

    preds = [model(batch[0].to(model.device)) for batch in dataloader]
    preds = torch.cat(preds, axis=0)
    preds = denormalize(preds).cpu()

    targets = [batch[1] for batch in dataloader]
    targets = torch.cat(targets, axis=0)
    targets = denormalize(targets).cpu()

    ssims, ssim_images = ssim(
        preds,
        targets,
        data_range=1.0,
        return_full_image=True,
        reduction="none",
    )

    # Output average SSIM over depth
    ssim_over_depth = depth_ssim(preds, targets)
    ssim_over_depth_string = "depth,ssim\n"
    for depth, val in enumerate(ssim_over_depth, 1):
        ssim_over_depth_string += f"{depth},{val}\n"

    report_dir = os.path.join("reports", hparams.name)

    if not os.path.isdir(report_dir):
        os.mkdir(report_dir)

    with open(os.path.join(report_dir, "depth_ssim.csv"), "w") as f:
        f.write(ssim_over_depth_string)

    # Output prediction images
    outputs_dir = os.path.join(report_dir, "outputs")
    if not os.path.isdir(outputs_dir):
        os.mkdir(outputs_dir)

    for index, pred in enumerate(preds):
        write_png(
            to_int(pred),
            os.path.join(outputs_dir, f"{str(index).zfill(5)}.png"),
            compression_level=0,
        )

    # Output SSIM maps
    ssim_images_dir = os.path.join(report_dir, "ssim_images")
    if not os.path.isdir(ssim_images_dir):
        os.mkdir(ssim_images_dir)

    for index, ssim_image in enumerate(ssim_images):
        write_png(
            to_int(ssim_image),
            os.path.join(
                report_dir,
                "ssim_images",
                f"{str(index).zfill(5)}.png",
            )
        )

    # Output mean statistics over entire test dataset
    ssim_stat = ssim(preds, targets, data_range=1.0)
    psnr_stat = psnr(preds, targets, data_range=1.0)
    rmse_stat = mse(preds, targets, squared=False)

    with open(os.path.join(report_dir, "stats.txt"), "w") as f:
        f.write(f"SSIM: {ssim_stat}\npSNR: {psnr_stat}\nRMSE: {rmse_stat}\n")

    ssim_per_image_string = ""
    for index, image_ssim in enumerate(ssims):
        ssim_per_image_string += f"{str(index).zfill(5)}: {image_ssim}\n"

    with open(os.path.join(report_dir, "ssim_per_image.txt"), "w") as f:
        f.write(ssim_per_image_string)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("name")
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=pathlib.Path,
        help="Path to checkpoint",
    )
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
    parser.add_argument("-bs", "--batch-size", default=2, type=int)
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
            "vit_unet",
            "palette",
        ],
    )
    args = parser.parse_args()

    main(args)
