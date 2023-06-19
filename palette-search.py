import torch
from argparse import ArgumentParser
import pathlib
from tqdm import tqdm
from models.palette import Palette, DiffusionModel
from dataset import ImageDataModule
from models.utils import ssim, denormalize


torch.set_float32_matmul_precision("medium")


def main(hparams):
    model = Palette.load_from_checkpoint(hparams.checkpoint)
    model.freeze()

    data_module = ImageDataModule(
        hparams.input_dir,
        hparams.target_dir,
        batch_size=hparams.batch_size,
        val_size=hparams.val_size,
        normalize=True,
    )
    data_module.setup("fit")

    val_data = data_module.val_dataloader()

    starts = torch.logspace(-7, -2, 20)
    ends = torch.logspace(-5, -1, 20)
    grid = torch.cartesian_prod(starts, ends)

    output = "start,end,ssim\n"

    for start, end in tqdm(grid):
        # No diffusion process if start value is greater than the end value
        if start >= end:
            continue

        diffusion = DiffusionModel(
            "linear",
            timesteps=100,
            start=start,
            end=end,
            learn_var=model.learn_var,
            device=model.device,
        )

        ssims = []
        for batch in val_data:
            x, y = batch
            x = x.to(model.device)
            y = y.to(model.device)

            y_t = torch.randn_like(x, device=model.device)
            for i in tqdm(
                reversed(range(diffusion.timesteps)),
                total=diffusion.timesteps,
            ):
                t = torch.full((hparams.batch_size,), i, device=x.device)
                y_t = diffusion.backward(x, y_t, t, model.unet)

            y_0 = denormalize(y_t)
            y = denormalize(y)

            ssims.append(ssim(y_0, y))

        ssim_current = torch.tensor(ssims).mean()

        output += f"{start},{end},{ssim_current}\n"

    with open(hparams.output_file, "w") as f:
        f.write(output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("output_file")
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
        "-c",
        "--checkpoint",
        type=pathlib.Path,
        help="Checkpoint of the trained Palette model.",
    )
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--val-size", default=0.2, type=float)
    args = parser.parse_args()

    main(args)
