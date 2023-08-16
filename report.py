import torch
import torch.nn as nn
from torchmetrics.functional import (
    peak_signal_noise_ratio as psnr,
    structural_similarity_index_measure as ssim,
    mean_squared_error as mse,
)
from matplotlib import colormaps
from torchvision.io import write_png
from argparse import ArgumentParser
import pathlib
import os
from fvcore.nn import FlopCountAnalysis
from models.pix2pix import Pix2Pix
from models.palette import Palette
from models.attention_unet import AttentionUnetGAN
from models.res_unet import ResUnetGAN
from models.trans_unet import TransUnetGAN
from models.utils import denormalize, to_int, get_parameter_count
from dataset import ImageDataModule


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

        case "trans_unet":
            model = TransUnetGAN.load_from_checkpoint(hparams.checkpoint)
            model.freeze()

        case "identity":
            def model(x): return x

        case _:
            raise ValueError(f"Incorrect model name ({hparams.model})")

    if isinstance(model, nn.Module):
        device = model.device
    else:
        device = "cpu"

    data_module = ImageDataModule(
        hparams.data,
        batch_size=hparams.batch_size,
    )
    data_module.setup("predict")
    dataloader = data_module.predict_dataloader()

    preds = [denormalize(model(batch[0].to(device))) for batch in dataloader]
    preds = torch.cat(preds, axis=0)
    preds = preds.cpu()
    # preds = denormalize(preds).cpu()

    targets = [denormalize(batch[1]) for batch in dataloader]
    targets = torch.cat(targets, axis=0)
    targets = targets.cpu()

    # Compute SSIM, PSNR, and MSE per image
    psnrs = []
    ssims = []
    mses = []
    ssim_images = []
    for pred, target in zip(preds.split(64), targets.split(64)):
        current_ssim, current_ssim_images = ssim(
            pred,
            target,
            data_range=1.0,
            return_full_image=True,
            reduction="none",
        )
        ssims.append(current_ssim)
        ssim_images.append(current_ssim_images)

        current_psnr = torch.tensor([
            psnr(p, t, data_range=1.0) for p, t in zip(pred, target)
        ])
        psnrs.append(current_psnr)

        current_mse = torch.tensor([
            mse(p, t) for p, t in zip(pred, target)
        ])
        mses.append(current_mse)

    ssims = torch.cat(ssims)
    ssim_images = torch.cat(ssim_images)
    psnrs = torch.cat(psnrs)
    mses = torch.cat(mses)

    # Output average SSIM over depth and standard deviation
    ssim_over_depth = depth_ssim(preds, targets)
    ssim_over_depth_string = "depth,mean,std\n"
    for depth, (mean, std) in enumerate(ssim_over_depth, 1):
        ssim_over_depth_string += f"{depth},{mean},{std}\n"

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
        output_hot_image(
            pred,
            os.path.join(outputs_dir, f"{str(index).zfill(5)}.png"),
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
    ssim_stat = ssims.mean()
    psnr_stat = psnrs.mean()
    rmse_stat = mse(preds, targets, squared=False)
    parameter_count = get_parameter_count(model)

    # Count FLOPs
    flops = 0
    if isinstance(model, nn.Module):
        input_ = torch.randn(1, 3, 256, 256).to(device)
        flops = FlopCountAnalysis(model, input_)
        flops = flops.total()

    with open(os.path.join(report_dir, "stats.txt"), "w") as f:
        f.write(f"SSIM: {ssim_stat}\n")
        f.write(f"PSNR: {psnr_stat}\n")
        f.write(f"RMSE: {rmse_stat}\n")
        f.write(f"FLOPs: {flops}\n")
        f.write(f"Parameter count: {parameter_count}\n")

    # Output SSIM per image
    ssim_per_image_string = "image,ssim\n"
    for index, image_ssim in enumerate(ssims):
        ssim_per_image_string += f"{str(index).zfill(5)},{image_ssim}\n"

    with open(os.path.join(report_dir, "ssim_per_image.csv"), "w") as f:
        f.write(ssim_per_image_string)

    # Output PSNR per image
    psnr_per_image_string = "image,psnr\n"
    for index, image_psnr in enumerate(psnrs):
        psnr_per_image_string += f"{str(index).zfill(5)},{image_psnr}\n"

    with open(os.path.join(report_dir, "psnr_per_image.csv"), "w") as f:
        f.write(psnr_per_image_string)

    # Output RMSE per image
    mse_per_image_string = "image,mse\n"
    for index, image_mse in enumerate(mses):
        mse_per_image_string += f"{str(index).zfill(5)},{image_mse}\n"

    with open(os.path.join(report_dir, "mse_per_image.csv"), "w") as f:
        f.write(mse_per_image_string)


def depth_ssim(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_depths: int = 16
) -> list[(float, float)]:
    """Compute mean and standard deviation of SSIM over depth of images. The
    depth goes in the y-axis of the image.

    :param preds: [N x C x H x W]
    :param targets: [N x C x H x W]
    :returns: [num_depths]

    """

    x_depths = preds.chunk(num_depths, dim=2)
    y_depths = targets.chunk(num_depths, dim=2)

    ssims = []
    for depth in range(num_depths):
        depth_ssim = ssim(
            x_depths[depth],
            y_depths[depth],
            data_range=1.0,
            reduction="none",
        )
        mean = depth_ssim.mean()
        std = depth_ssim.std()
        ssims.append((mean, std))

    return torch.tensor(ssims)


def output_hot_image(img: torch.Tensor, filename: str):
    """Outputs a hot-encoded image using the matplotlib hot colormap.

    :arg img: [1 x H x W]
    :arg filename: File location to save output.

    """

    colormap = colormaps["hot"]
    img = colormap(img)
    img = img[0, :, :, :3]
    img = torch.Tensor(img)
    img = torch.permute(img, (2, 0, 1))
    write_png(to_int(img), filename)


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
        "-d",
        "--data",
        type=pathlib.Path,
        help="YAML file of all data points",
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
            "trans_unet",
            "palette",
            "identity",
        ],
    )
    args = parser.parse_args()

    main(args)
