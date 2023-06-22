import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import pytorch_lightning as pl
import yaml
import os


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_list_file: str,
        batch_size: int = 1,
        val_size: float | int = 0.2,
        normalize: bool = True,
    ):
        super().__init__()

        with open(data_list_file, "r") as f:
            data_list = yaml.safe_load(f)

        data_dir = os.path.dirname(data_list_file)
        self.data_tuples: list[(str, str, int)] = list(map(
            lambda x: (
                os.path.join(data_dir, x["input"]),
                os.path.join(data_dir, x["ground_truth"]),
                x["noise"],
            ),
            data_list,
        ))

        self.batch_size = batch_size
        self.val_size = val_size
        self.normalize = normalize

        trans = [
            transforms.Resize((256, 256), antialias=True),
            transforms.ConvertImageDtype(torch.float32),
        ]

        if normalize:
            trans.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            )

        self.transform = transforms.Compose(trans)

    def setup(self, stage: str):
        if stage == "fit":
            data_size = len(self.data_tuples)
            if self.val_size < 1:
                split_index = round(data_size * (1 - self.val_size))
            else:
                split_index = round(data_size - self.val_size)

            self.train_split = self.data_tuples[:split_index]
            self.val_split = self.data_tuples[split_index:]

        if stage == "validate":
            self.val_split = self.data_tuples

        if stage == "test":
            self.test_split = self.data_tuples

        if stage == "predict":
            self.pred_split = self.data_tuples

    def train_dataloader(self):
        return DataLoader(
            ImageDataset(self.train_split, transform=self.transform),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            ImageDataset(self.val_split, transform=self.transform),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            ImageDataset(self.test_split, transform=self.transform),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            ImageDataset(self.pred_split, transform=self.transform),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )


class ImageDataset(Dataset):
    """Dataset class used for image data that has input and targets in separate
    directories in the same order. Supports batches."""

    def __init__(
        self,
        data_tuples: list[(str, str, int)],
        transform=None,
    ):
        super().__init__()
        self.data_tuples = data_tuples
        self.transform = transform

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):
        (input_, gt, noise) = self.data_tuples[idx]

        input_tensor = read_image(input_, mode=ImageReadMode.RGB)
        input_tensor = self.transform(input_tensor)
        gt_tensor = read_image(gt, mode=ImageReadMode.RGB)
        gt_tensor = self.transform(gt_tensor)

        return input_tensor, gt_tensor, noise
