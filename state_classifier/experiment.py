"""Experiment orchestration for state classification."""
import torch
import wandb
from torch.utils.data import DataLoader

from state_classifier.config.config_utils import load_config, override_config_with_wandb
from state_classifier.data.datasets import create_train_val_datasets, create_test_dataset
from state_classifier.data.samplers import CombinedSampleBatchSampler
from state_classifier.data.transforms import get_train_transforms, get_val_transforms, get_test_transforms
from state_classifier.models import build_state_classifier
from state_classifier.trainers.trainer import Trainer
from state_classifier.trainers import Tester


class Experiment:
    """
    Experiment class that orchestrates the complete workflow:
    - Data loading
    - Model building
    - Training
    - Testing
    - Logging
    """

    def __init__(self, config):
        """
        Initialize the experiment.

        Args:
            config (SimpleNamespace): Experiment configuration
        """
        self.config = config

        # Build components
        self.model = self._build_model()
        self.train_loader, self.val_loader = self._build_dataloaders()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            config=config
        )

        # Create test loader and tester
        self.test_loader = self._build_test_loader()
        self.tester = Tester(
            model=self.model,
            test_loader=self.test_loader,
            config=self.config
        )

    def run(self):
        """
        Run the complete experiment workflow:
        1. Initialize W&B
        2. Train model
        3. Test model
        4. Finish logging
        """
        # Initialize W&B
        wandb.init(
            project=self.config.wandb.project,
            config=self._get_config_dict()
        )
        wandb.watch(self.model)

        # Train model
        self.trainer.train()

        # Test model
        final_step = self.trainer.current_epoch + 1
        self.tester.test(test_step=final_step)

        # Finish logging
        wandb.finish()

    def train(self):
        """Run only the training part of the experiment."""
        self.trainer.train()

    def test(self):
        """Run only the testing part of the experiment."""
        self.tester.test()

    def _get_config_dict(self):
        """
        Convert config to dictionary for logging.

        Returns:
            dict: Configuration as a dictionary
        """

        def recursive_vars(obj):
            if isinstance(obj, dict):
                return {k: recursive_vars(v) for k, v in obj.items()}
            elif hasattr(obj, '__dict__'):
                return {k: recursive_vars(v) for k, v in vars(obj).items()}
            else:
                return obj

        return recursive_vars(self.config)

    def _build_model(self):
        """
        Build model based on configuration.

        Returns:
            nn.Module: The model
        """
        model_cfg = self.config.model_cfg
        return build_state_classifier(num_classes=model_cfg.num_classes)

    def _build_dataloaders(self):
        """
        Build train and validation dataloaders.

        Returns:
            tuple: (train_loader, val_loader)
        """
        # Get transforms
        train_transforms = get_train_transforms(size=(224, 224))
        val_transforms = get_val_transforms(size=(224, 224))

        # Get the full dataset size from config, default to 1.0 if not present
        full_dataset_size = getattr(self.config, 'full_dataset_size', 1.0)

        # Create datasets
        train_set, val_set = create_train_val_datasets(
            dataset_root=self.config.dataset_root,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            train_val_split=self.config.train_val_split,
            seed=self.config.seed
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            prefetch_factor=self.config.prefetch_factor,
            persistent_workers=True
        )

        val_loader = DataLoader(
            val_set,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=True
        )

        return train_loader, val_loader

    def _build_test_loader(self):
        """
        Build test dataloader with CombinedSampleBatchSampler.

        Returns:
            DataLoader: Test dataloader
        """
        # Get transforms
        test_transforms = get_test_transforms(size=(224, 224))

        # Create test dataset
        test_set = create_test_dataset(
            test_root=self.config.test_dataset_root,
            test_transforms=test_transforms
        )

        # Create batch sampler that handles the 4 perspectives per location
        batch_sampler = CombinedSampleBatchSampler(
            test_set,
            groups_per_batch=self.config.test_groups_per_batch
        )

        # Create dataloader with batch sampler
        test_loader = DataLoader(
            test_set,
            batch_sampler=batch_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        return test_loader


def run_experiment(config_path):
    """
    Run experiment from config file.

    Args:
        config_path (str): Path to config file
    """
    # Load config
    config = load_config(config_path)

    # Override with W&B if needed
    try:
        import wandb
        if wandb.run:
            config = override_config_with_wandb(config)
    except ImportError:
        pass

    # Create and run experiment
    experiment = Experiment(config)
    experiment.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run state classification experiment")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    run_experiment(args.config)