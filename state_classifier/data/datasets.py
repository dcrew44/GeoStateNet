"""Dataset classes for state classification."""
import os
import torch
from torchvision.datasets import ImageFolder


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


def create_train_val_datasets(dataset_root, train_transforms, val_transforms, train_val_split=0.8, seed=42):
    """
    Create train and validation datasets with proper transforms.

    Args:
        dataset_root (str): Root directory of the dataset
        train_transforms (callable): Transforms for training set
        val_transforms (callable): Transforms for validation set
        train_val_split (float): Fraction of data to use for training
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    # Load full dataset without transforms first
    full_dataset = ImageFolder(root=dataset_root, transform=None)

    # Calculate split sizes
    train_size = int(train_val_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Create random split
    train_set, val_set = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Apply transforms to each subset
    train_set.dataset.transform = train_transforms
    val_set.dataset.transform = val_transforms

    return train_set, val_set


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