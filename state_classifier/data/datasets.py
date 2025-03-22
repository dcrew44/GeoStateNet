"""Dataset classes for state classification."""
import os
import torch
import numpy as np
import random
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset, random_split
from sklearn.model_selection import train_test_split
from ..data.transforms import get_train_transforms, get_val_transforms

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
    Create train and validation datasets with proper transforms,
    ensuring that images from the same location stay together.
    """
    # Load full dataset without transforms first
    full_dataset = ImageFolder(root=dataset_root)

    # Group sample indices by location ID
    location_to_indices = {}
    for idx, (path, _) in enumerate(full_dataset.samples):
        filename = os.path.basename(path)
        # "2007_5oyTy...._0.jpg" becomes "2007_5oyTy...."
        location_id = "_".join(filename.split("_")[:-1])

        if location_id not in location_to_indices:
            location_to_indices[location_id] = []

        location_to_indices[location_id].append(idx)

    # Get list of all location IDs
    location_ids = list(location_to_indices.keys())

    # Set random seed for reproducibility
    random.seed(seed)
    random.shuffle(location_ids)

    # Split locations into train and val
    num_train_locations = int(len(location_ids) * train_val_split)
    train_location_ids = location_ids[:num_train_locations]
    val_location_ids = location_ids[num_train_locations:]

    # Collect indices for train and val sets
    train_indices = []
    for loc_id in train_location_ids:
        train_indices.extend(location_to_indices[loc_id])

    val_indices = []
    for loc_id in val_location_ids:
        val_indices.extend(location_to_indices[loc_id])

    # Create Subset datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # Set transforms
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms

    print(f"Train set: {len(train_indices)} images from {len(train_location_ids)} locations")
    print(f"Val set: {len(val_indices)} images from {len(val_location_ids)} locations")

    return train_dataset, val_dataset


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