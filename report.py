import torch
from torchmetrics.functional import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim,
)
from torchvision.io import write_png
import torchvision.transforms as transforms
import pytorch_lightning as pl
from models.pix2pix import Pix2Pix
from models.palette import Palette
from models.transgan import TransGAN
from models.resnet import ResNetGAN
from models.attention_unet import AttentionUNet
from reporting.depth_ssim import depth_ssim
from dataset import ImageDataModule
from argparse import ArgumentParser
import pathlib
import os


to_int = transforms.Compose([
    transforms.ConvertImageDtype(torch.uint8),
])


def main(hparams):
    pl.seed_everything(42, workers=True)

    model = None
    match hparams.model:
        case "pix2pix":
            model = Pix2Pix.load_from_checkpoint(hparams.checkpoint)
            model.freeze()

        case "palette":
            model = Palette.load_from_checkpoint(hparams.checkpoint)
            model.freeze()

        case "transgan":
            model = TransGAN.load_from_checkpoint(hparams.checkpoint)
            model.freeze()

        case "resnet":
            model = ResNetGAN.load_from_checkpoint(hparams.checkpoint)
            model.freeze()

        case "attention-unet":
            model = AttentionUNet.load_from_checkpoint(hparams.checkpoint)
            model.freeze()

    if model is None:
        raise ValueError(f"Incorrect model name ({hparams.model})")

    data_module = ImageDataModule(
        hparams.input_dir,
        hparams.target_dir,
        batch_size=hparams.batch_size,
    )
    data_module.setup("predict")

    # Get inputs and outputs into big tensors
    inputs = [batch[0] for batch in data_module.predict_dataloader()]
    inputs = torch.cat(inputs, axis=0).to(model.device)

    targets = [batch[1] for batch in data_module.predict_dataloader()]
    targets = torch.cat(targets, axis=0).cpu()

    preds = model(inputs)
    preds = torch.clamp(preds, 0, 1).cpu()

    print("SSIM:", ssim(preds, targets, data_range=1.0).tolist())
    print("pSNR:", psnr(preds, targets, data_range=1.0).tolist())

    # Output SSIM over depth
    ssim_over_depth = depth_ssim(preds, targets)
    ssim_over_depth_string = "depth,ssim\n"
    for depth, val in enumerate(ssim_over_depth, 1):
        ssim_over_depth_string += f"{depth},{val}\n"

    if not os.path.isdir(hparams.pred_dir):
        os.mkdir(hparams.pred_dir)

    with open(os.path.join(hparams.pred_dir, "depth_ssim.csv"), "w") as f:
        f.write(ssim_over_depth_string)

    for index, pred in enumerate(preds):
        write_png(
            to_int(pred),
            os.path.join(hparams.pred_dir, f"{index}.png"),
            compression_level=0,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("checkpoint")
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
    parser.add_argument(
        "-p",
        "--pred-dir",
        type=pathlib.Path,
        help="Directory path that the predictions will be saved to.",
    )
    parser.add_argument("-bs", "--batch-size", default=2, type=int)
    parser.add_argument(
        "-m",
        "--model",
        default="pix2pix",
        choices=["pix2pix", "palette", "transgan", "resnet", "attention-unet"],
    )
    args = parser.parse_args()

    main(args)
