import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchmetrics.functional import (
    structural_similary_index_measure,
    peak_signal_noise_ratio,
)


denormalize = transforms.Lambda(lambda x: torch.clamp(x * 0.5 + 0.5, 0, 1))
to_int = transforms.ConvertImageDtype(torch.uint8)


def init_weights(module: nn.Module):
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d)):
        module.weight.data.normal_(0.0, 0.02)

    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.zero_()


def ssim(pred: torch.Tensor, target: torch.Tensor):
    return structural_similary_index_measure(pred, target, data_range=1.0)


def psnr(pred: torch.Tensor, target: torch.Tensor):
    return peak_signal_noise_ratio(pred, target, data_range=1.0)
