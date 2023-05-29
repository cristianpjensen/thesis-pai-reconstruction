import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import pytorch_lightning as pl
from natsort import natsorted
import os


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        input_dir: str,
        target_dir: str,
        batch_size: int = 1,
        val_size: float | int = 0.2,
        normalize: bool = False,
        grayscale: bool = False,
    ):
        super().__init__()
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.batch_size = batch_size
        self.val_size = val_size
        self.normalize = normalize
        self.grayscale = grayscale

        trans = [
            transforms.Resize((256, 256), antialias=True),
            transforms.ConvertImageDtype(torch.float32),
        ]

        if normalize:
            trans.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            )

        if grayscale:
            trans.append(transforms.Grayscale(1))

        self.transform = transforms.Compose(trans)

    def _get_pairs(self, input_dir, target_dir):
        input_imgs = get_image_filenames(input_dir)
        target_imgs = get_image_filenames(target_dir)

        if len(input_imgs) != len(target_imgs):
            raise Exception("There should be the same amount of input" +
                            "images as target images")

        if len(input_imgs) == 0:
            raise Exception("No images in specified directories.")

        return list(zip(input_imgs, target_imgs))

    def setup(self, stage: str):
        if stage == "fit":
            batches = self._get_pairs(self.input_dir, self.target_dir)
            split_index = round(
                len(batches) * (1 - self.val_size)
            ) if self.val_size < 1 else round(len(batches) - self.val_size)
            self.train_pairs = batches[:split_index]
            self.val_pairs = batches[split_index:]

        if stage == "validate":
            self.val_pairs = self._get_pairs(self.input_dir, self.target_dir)

        if stage == "test":
            self.test_pairs = self._get_pairs(self.input_dir, self.target_dir)

        if stage == "predict":
            self.pred_pairs = self._get_pairs(self.input_dir, self.target_dir)

    def train_dataloader(self):
        return DataLoader(
            ImageDataset(self.train_pairs, transform=self.transform),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            ImageDataset(self.val_pairs, transform=self.transform),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            ImageDataset(self.test_pairs, transform=self.transform),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            ImageDataset(self.pred_pairs, transform=self.transform),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )


class ImageDataset(Dataset):
    """Dataset class used for image data that has input and targets in separate
    directories in the same order. Supports batches."""

    def __init__(
        self,
        paths: list[(str, str)] | list[(str,)],
        transform=None,
    ):
        super().__init__()

        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        paths = self.paths[idx]

        output = []
        for path in paths:
            tensor = read_image(path, mode=ImageReadMode.RGB)
            tensor = self.transform(tensor)
            output.append(tensor.float())

        return output


def get_image_filenames(dir: str):
    """Find all image files in the directory and return as naturally sorted
    list (according to the numbers in the filename)."""

    return natsorted([
        os.path.join(dir, f) for f in os.listdir(dir)
        if os.path.isfile(os.path.join(dir, f)) and
        os.path.splitext(f)[1].lower() in [".jpeg", ".jpg", ".png"]
    ])
