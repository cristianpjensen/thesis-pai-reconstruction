import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchmetrics.functional import (
    structural_similarity_index_measure,
    peak_signal_noise_ratio,
    mean_squared_error,
)


denormalize = transforms.Lambda(lambda x: torch.clamp(x * 0.5 + 0.5, 0, 1))
to_int = transforms.ConvertImageDtype(torch.uint8)


def init_weights(module: nn.Module):
    if isinstance(
        module,
        (nn.Conv1d, nn.Conv2d, nn.ConvTranspose2d, nn.Linear),
    ):
        nn.init.normal_(module.weight, 0.0, 0.02)

    if isinstance(
        module,
        (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm),
    ):
        nn.init.constant_(module.weight, 1.0)

        nn.init.constant_(module.bias, 0.0)


def get_parameter_count(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


def ssim(pred: torch.Tensor, target: torch.Tensor):
    return structural_similarity_index_measure(pred, target, data_range=1.0)


def psnr(pred: torch.Tensor, target: torch.Tensor):
    return peak_signal_noise_ratio(pred, target, data_range=1.0)


def rmse(pred: torch.Tensor, target: torch.Tensor):
    return mean_squared_error(pred, target, squared=False)
