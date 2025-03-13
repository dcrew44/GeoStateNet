"""Batch samplers for custom dataset sampling strategies."""
import os
import itertools
from torch.utils.data.sampler import BatchSampler


class CombinedSampleBatchSampler(BatchSampler):
    """
    Custom batch sampler that groups related samples (e.g., rotations of the same image).

    This sampler is designed for the 50States10K dataset where each location has
    multiple perspective images that should be processed together.
    """

    def __init__(self, dataset, groups_per_batch=64):
        """
        Initialize the sampler.

        Args:
            dataset: Dataset with 'samples' attribute containing (path, label) tuples
            groups_per_batch (int): Number of sample groups per batch
        """
        self.dataset = dataset
        self.groups_per_batch = groups_per_batch

        # Group indices by sample ID (assumes sample id is all parts except the last underscore)
        self.sample_groups = {}
        for idx, (path, _) in enumerate(dataset.samples):
            filename = os.path.basename(path)
            # "2007_5oyTy...._0.jpg" becomes "2007_5oyTy...."
            sample_id = "_".join(filename.split("_")[:-1])

            if sample_id not in self.sample_groups:
                self.sample_groups[sample_id] = []

            self.sample_groups[sample_id].append(idx)

        # Only keep groups that have exactly 4 images (all rotations present)
        self.groups = [
            group for group in self.sample_groups.values()
            if len(group) == 4
        ]

        # Combine multiple sample groups into one batch
        self.batches = [
            list(itertools.chain.from_iterable(
                self.groups[i:i + groups_per_batch]))
            for i in range(0, len(self.groups), groups_per_batch)
        ]

    def __iter__(self):
        """Iterate over batches."""
        for batch in self.batches:
            yield batch

    def __len__(self):
        """Number of batches."""
        return len(self.batches)