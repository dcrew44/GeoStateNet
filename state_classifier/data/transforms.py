"""Data transforms for state classification model."""
import torch
from torchvision.transforms import v2
from ..utils.constants import IMAGENET_MEAN, IMAGENET_STD
from PIL import Image


def get_train_transforms(size=(224, 224)):
    """
    Get transforms for training data, exactly matching FastAI's aug_transforms.
    """
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(size=256),  # FastAI typically resizes slightly larger first
        v2.RandomHorizontalFlip(p=0.5),  # FastAI's Flip transform (p=0.5)

        # FastAI's Affine transforms (all with p_affine=0.75)
        v2.RandomApply([
            v2.RandomAffine(
                degrees=10.0,  # max_rotate=10.0
                scale=(1.0, 1.1),  # min_zoom=1.0, max_zoom=1.1
                interpolation=Image.BILINEAR,
                fill=0
            )
        ], p=0.75),

        # FastAI's Warp transform (p_affine=0.75)
        v2.RandomApply([
            v2.RandomPerspective(
                distortion_scale=0.2,  # max_warp=0.2
                p=1.0,  # Always apply when RandomApply is triggered
                interpolation=Image.BILINEAR,
                fill=0
            )
        ], p=0.75),

        # FastAI's separate Brightness and Contrast (p_lighting=0.75)
        v2.RandomApply([
            v2.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0)
        ], p=0.75),

        v2.RandomApply([
            v2.ColorJitter(brightness=0, contrast=0.2, saturation=0, hue=0)
        ], p=0.75),

        # Center crop to final size (FastAI behavior)
        v2.CenterCrop(size=size),

        v2.ToDtype(torch.float32, scale=True),

        v2.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),

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