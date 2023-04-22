import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from natsort import natsorted
from typing import TypeVar
import os


T = TypeVar('T')


def get_batches(lst: list[T], batch_size: int) -> list[tuple[T]]:
    """Partition a list into batches of `batch_size` as tuples.

    Source: https://stackoverflow.com/a/23286299"""

    iterators = [iter(lst)] * batch_size
    return list(zip(*iterators))


def get_image_filenames(dir: str):
    """Find all image files in the directory and return as naturally sorted
    list (according to the numbers in the filename)."""

    return natsorted([
        os.path.join(dir, f) for f in os.listdir(dir)
        if os.path.isfile(os.path.join(dir, f)) and
        os.path.splitext(f)[1].lower() in [".jpeg", ".jpg", ".png"]
    ])


class SplitImageDataset():
    def __init__(
        self,
        input_dir: str,
        label_dir: str,
        batch_size: int = 1,
        val_size: float = 0.25,
        shuffle: bool = False,
        transform=None,
        label_transform=None,
    ):
        input_dir = input_dir
        label_dir = label_dir
        input_imgs = get_image_filenames(input_dir)
        label_imgs = get_image_filenames(label_dir)

        # There should be the same amount of input images as label images.
        if len(input_imgs) != len(label_imgs):
            raise Exception("There should be the same amount of input images" +
                            "as label images")

        # There should be images.
        if len(input_imgs) == 0:
            raise Exception("No images in specified directories.")

        image_pairs = list(zip(input_imgs, label_imgs))

        if shuffle:
            # Shuffle with pytorch, so we only need to set the seed for
            # pytorch before our training loop.
            image_pairs = [
                image_pairs[i] for i in torch.randperm(len(image_pairs))
            ]

        batches = get_batches(image_pairs, batch_size)
        split_index = round(len(batches) * (1 - val_size))
        train_batches = batches[:split_index]
        val_batches = batches[split_index:]

        self.train = ImageDataset(train_batches, transform, label_transform)
        self.validation = ImageDataset(val_batches, transform, label_transform)


class ImageDataset(Dataset):
    """Dataset class used for image data that has input and labels in separate
    directories in the same order. Supports batches."""

    def __init__(
        self,
        batches: list[tuple[(str, str)]],
        transform=None,
        label_transform=None,
    ):
        super().__init__()

        self.batches = batches
        self.transform = transform
        if label_transform:
            self.label_transform = label_transform
        else:
            self.label_transform = transform

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self._batch_to_tensor(self.batches[idx])

    def _batch_to_tensor(
        self,
        batch: list[(str, str)]
    ) -> (torch.Tensor, torch.Tensor):
        """Reads the input and label images in the batch and returns as batched
        tensor.

        Output shapes: (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
        """

        input_tensors = []
        label_tensors = []

        # Collect all tensors of all images in the batch.
        for input_path, label_path in batch:
            input_tensor = read_image(input_path, mode=ImageReadMode.RGB)
            label_tensor = read_image(label_path, mode=ImageReadMode.RGB)

            if self.transform:
                input_tensor = self.transform(input_tensor)

            if self.label_transform:
                label_tensor = self.label_transform(label_tensor)

            input_tensors.append(input_tensor)
            label_tensors.append(label_tensor)

        # Stack tensors.
        return torch.stack(input_tensors), torch.stack(label_tensors)
