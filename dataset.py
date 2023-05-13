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
        val_size: float = 0.2,
    ):
        super().__init__()
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.batch_size = batch_size
        self.val_size = val_size

        self.transform = transforms.Compose([
            transforms.Resize((256, 256), antialias=True),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def _get_pairs(self):
        input_imgs = get_image_filenames(self.input_dir)
        target_imgs = get_image_filenames(self.target_dir)

        if len(input_imgs) != len(target_imgs):
            raise Exception("There should be the same amount of input" +
                            "images as target images")

        if len(input_imgs) == 0:
            raise Exception("No images in specified directories.")

        return list(zip(input_imgs, target_imgs))

    def setup(self, stage: str):
        if stage == "fit":
            batches = self._get_pairs()
            split_index = round(len(batches) * (1 - self.val_size))
            self.train_pairs = batches[:split_index]
            self.val_pairs = batches[split_index:]

        if stage == "validate":
            self.val_pairs = self._get_pairs()

        if stage == "test":
            self.test_pairs = self._get_pairs()

        if stage == "predict":
            self.pred_pairs = self._get_pairs()

    def train_dataloader(self):
        return DataLoader(
            ImageDataset(self.train_pairs, transform=self.transform),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            ImageDataset(self.val_pairs, transform=self.transform),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            ImageDataset(self.test_pairs, transform=self.transform),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )


class ImageDataset(Dataset):
    """Dataset class used for image data that has input and targets in separate
    directories in the same order. Supports batches."""

    def __init__(
        self,
        pairs: list[(str, str)],
        transform=None,
    ):
        super().__init__()

        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_path, target_path = self.pairs[idx]
        input_tensor = read_image(input_path, mode=ImageReadMode.RGB)
        target_tensor = read_image(target_path, mode=ImageReadMode.RGB)

        if self.transform:
            input_tensor = self.transform(input_tensor)
            target_tensor = self.transform(target_tensor)

        return input_tensor.float(), target_tensor.float()


def get_image_filenames(dir: str):
    """Find all image files in the directory and return as naturally sorted
    list (according to the numbers in the filename)."""

    return natsorted([
        os.path.join(dir, f) for f in os.listdir(dir)
        if os.path.isfile(os.path.join(dir, f)) and
        os.path.splitext(f)[1].lower() in [".jpeg", ".jpg", ".png"]
    ])
