import torch
from torchmetrics.functional import (
    structural_similarity_index_measure as ssim,
)


def depth_ssim(x, y, num_depths: int = 16):
    """
    :param x: [N x C x H x W]
    :param y: [N x C x H x W]
    :returns: [num_depths]

    """

    x_depths = x.chunk(num_depths, dim=2)
    y_depths = y.chunk(num_depths, dim=2)

    ssims = []
    for depth in range(num_depths):
        depth_ssim = ssim(x_depths[depth], y_depths[depth], data_range=1.0)
        ssims.append(depth_ssim)

    return torch.tensor(ssims)


if __name__ == "__main__":
    # Testing
    x = torch.rand(2, 3, 256, 256)
    y = torch.rand(2, 3, 256, 256)

    print(depth_ssim(x, y, num_depths=16))
