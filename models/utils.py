import torch
import torchvision.transforms as transforms


denormalize = transforms.Lambda(lambda x: (x + 1) / 2)
to_int = transforms.ConvertImageDtype(torch.uint8)
