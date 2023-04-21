from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_dataset(
    path: str,
    subfolders: list[str],
    input_size: int,
) -> list[DataLoader]:
    """Returns data loaders in the same order as `subfolders`."""

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size * 2)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    data_loaders = []

    for subfolder in subfolders:
        dataset = datasets.ImageFolder(path, transform)
        class_index = dataset.class_to_idx[subfolder]

        n = 0
        for i in range(len(dataset)):
            if dataset.imgs[n][1] != class_index:
                del dataset.imgs[n]
                n -= 1

            n += 1

        data_loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
        )
        data_loaders.append(data_loader)

    return data_loaders
