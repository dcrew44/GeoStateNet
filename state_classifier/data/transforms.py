"""Data transforms for state classification model."""
import torch
from torchvision.transforms import v2
from ..utils.constants import IMAGENET_MEAN, IMAGENET_STD
from PIL import Image

def get_train_transforms(size=(224, 224)):
    """
    Get transforms for training data, similar to FastAI's aug_transforms with their parameters.
    """
    lighting_tfm = v2.RandomApply([
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
    ], p=0.75)

    affine_tfm = v2.RandomApply([
        v2.RandomAffine(
            degrees=10.0,
            translate=(0.1, 0.1),
            scale=(1.0, 1.1),
            interpolation=Image.BILINEAR
        )
    ], p=0.75)

    warp_tfm = v2.RandomApply([
        v2.RandomPerspective(distortion_scale=0.2, p=1.0, interpolation=Image.BILINEAR, fill=0)
    ], p=0.75)

    return v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size=size, scale=(1.0, 1.0), ratio=(0.9, 1.1), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        affine_tfm,
        lighting_tfm,
        warp_tfm,
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