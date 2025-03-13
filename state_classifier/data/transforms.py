"""Data transforms for state classification model."""
import torch
from torchvision.transforms import v2
from ..utils.constants import IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms(size=(224, 224)):
    """
    Get transforms for training data.

    Args:
        size (tuple): Target image size (height, width)

    Returns:
        callable: Composed transforms
    """
    return v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size=size, scale=(0.8, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=10.0),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
        v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05)),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(size=(224, 224)):
    """
    Get transforms for validation data.

    Args:
        size (tuple): Target image size (height, width)

    Returns:
        callable: Composed transforms
    """
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(size=size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_test_transforms(size=(224, 224)):
    """
    Get transforms for test data.

    Args:
        size (tuple): Target image size (height, width)

    Returns:
        callable: Composed transforms
    """
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(size=size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def unnormalize_image(tensor):
    """
    Unnormalize an image tensor from ImageNet normalization.

    Args:
        tensor (torch.Tensor): Normalized image tensor

    Returns:
        torch.Tensor: Unnormalized image tensor
    """
    mean = torch.tensor(IMAGENET_MEAN, device=tensor.device)
    std = torch.tensor(IMAGENET_STD, device=tensor.device)

    return tensor * std[:, None, None] + mean[:, None, None]