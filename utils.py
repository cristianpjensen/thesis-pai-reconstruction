import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
from natsort import natsorted
import os


class InputLabelImageDataset(Dataset):
    def __init__(
        self,
        input_dir: str,
        label_dir: str,
        batch_size: int = 1,
        transform=None,
        label_transform=None,
    ):
        super().__init__()

        self.input_dir = input_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        input_imgs = self._get_image_filenames(input_dir)
        label_imgs = self._get_image_filenames(label_dir)

        if len(input_imgs) != len(label_imgs):
            raise Exception("There should be the same amount of input images" +
                            "as label images")

        if len(input_imgs) == 0:
            raise Exception("No images in specified directories.")

        image_pairs = list(zip(input_imgs, label_imgs))
        self.batches = self._get_batches(image_pairs)

        self.transform = transform
        if label_transform:
            self.label_transform = label_transform
        else:
            self.label_transform = transform

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self._batch_to_tensor(self.batches[idx])

    def _get_image_filenames(self, dir: str):
        # Use `natsorted` to sort according to the numbers in the filenames.
        return natsorted([
            f for f in os.listdir(dir)
            if os.path.isfile(os.path.join(dir, f)) and
            os.path.splitext(f)[1].lower() in [".jpeg", ".jpg", ".png"]
        ])

    def _get_batches(self, imgs: list[(str, str)]) -> list[tuple[(str, str)]]:
        """Partition an image list into batches of `self.batch_size` as tuples.

        Source: https://stackoverflow.com/a/23286299"""

        iterators = [iter(imgs)] * self.batch_size
        return list(zip(*iterators))

    def _batch_to_tensor(
        self,
        batch: list[(str, str)]
    ) -> (torch.Tensor, torch.Tensor):
        input_tensors = []
        label_tensors = []

        # Collect all tensors of all images in the batch.
        for input_image, label_image in batch:
            # Input image.
            input_path = os.path.join(self.input_dir, input_image)
            input_tensor = read_image(input_path, mode=ImageReadMode.RGB)

            if self.transform:
                input_tensor = self.transform(input_tensor)

            input_tensors.append(input_tensor)

            # Label image.
            label_path = os.path.join(self.label_dir, label_image)
            label_tensor = read_image(label_path, mode=ImageReadMode.RGB)

            if self.label_transform:
                label_tensor = self.label_transform(label_tensor)

            label_tensors.append(label_tensor)

        # Stack tensors.
        return torch.stack(input_tensors), torch.stack(label_tensors)
