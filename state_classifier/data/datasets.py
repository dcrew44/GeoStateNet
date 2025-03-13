"""Dataset classes for state classification."""
import os
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


class TestSet(ImageFolder):
    """
    Custom dataset for testing that returns image path along with image and label.

    This allows for tracking sample IDs during evaluation and testing.
    """

    def __getitem__(self, idx):
        """
        Get item with path information.

        Args:
            idx (int): Index of the sample

        Returns:
            tuple: (image, label, path)
        """
        image, label = super().__getitem__(idx)
        path = self.samples[idx][0]
        return image, label, path


def get_subset_dataset(dataset, train_size=0.5, seed=42):
    """
    Creates a subset of the dataset while maintaining class proportions.

    Args:
        dataset: The original ImageFolder dataset
        train_size: Proportion of the data to use (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        A Subset of the original dataset
    """
    # Get all targets/labels from the dataset
    targets = np.array([target for _, target in dataset.samples])

    # Get indices for each class to maintain proportions
    indices = np.arange(len(dataset))

    # Use stratified split to maintain class distribution
    subset_indices, _ = train_test_split(
        indices,
        train_size=train_size,  # Now correctly using train_size
        stratify=targets,
        random_state=seed
    )

    # Create a Subset dataset using the selected indices
    subset_dataset = Subset(dataset, subset_indices)

    return subset_dataset


def create_train_val_datasets(dataset_root, train_transforms, val_transforms, train_val_split=0.8, seed=42,
                              full_dataset_size=1.0):
    """
    Create train and validation datasets with proper transforms.
    """
    # Load full dataset without transforms first
    full_dataset = ImageFolder(root=dataset_root)

    # Get a subset if needed
    if full_dataset_size < 1.0:
        full_dataset = get_subset_dataset(full_dataset, train_size=full_dataset_size, seed=seed)

    # Get targets for stratified split
    if isinstance(full_dataset, Subset):
        targets = np.array([full_dataset.dataset.targets[idx] for idx in full_dataset.indices])
        indices = np.arange(len(full_dataset))
    else:
        targets = np.array([target for _, target in full_dataset.samples])
        indices = np.arange(len(full_dataset))

    # Use stratified split for train/val
    train_indices, val_indices = train_test_split(
        indices,
        train_size=train_val_split,
        stratify=targets,
        random_state=seed
    )

    # Create subsets with appropriate transforms
    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)

    # Create wrapped datasets with different transforms
    train_dataset = TransformSubset(train_set, transform=train_transforms)
    val_dataset = TransformSubset(val_set, transform=val_transforms)

    return train_dataset, val_dataset


# Additional helper class needed
class TransformSubset(torch.utils.data.Dataset):
    """Dataset wrapper that applies a transform to a Subset."""

    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

def create_test_dataset(test_root, test_transforms):
    """
    Create test dataset with proper transforms.

    Args:
        test_root (str): Root directory for test data
        test_transforms (callable): Transforms to apply to test data

    Returns:
        TestSet: Test dataset
    """
    return TestSet(root=test_root, transform=test_transforms)