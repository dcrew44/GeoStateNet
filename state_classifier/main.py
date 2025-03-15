"""
Main script for 50States10K state classification model.

This script serves as the entry point for training and evaluating
a model that predicts which US state a streetview image was taken in.
"""
import argparse
import os
import torch
import wandb

from state_classifier.config.config_utils import load_config, override_config_with_wandb
from experiment import Experiment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a model for the 50States10K dataset."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "full"],
        default="full",
        help="Run mode: train, test, or full (both)"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )

    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Running as part of a W&B sweep"
    )

    parser.add_argument(
        "--full_dataset_size",
        type=float,
        default=1.0,
        help="Training set size. Choose smaller values (0.2) for running experiments on smaller datasets."
    )

    return parser.parse_args()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    config.full_dataset_size = args.full_dataset_size

    # Set sweep flag if needed
    if args.sweep:
        config.is_sweep = True

    # Initialize W&B if requested
    if args.wandb:
        if not args.sweep:  # Don't initialize if running as part of a sweep
            wandb.init(
                project=config.wandb.project,
                name=config.wandb.run_name if hasattr(config.wandb, "run_name") else None,
            )

        # Override config with W&B values
        config = override_config_with_wandb(config)

    # Create experiment
    experiment = Experiment(config)

    # Load checkpoint if resuming
    if args.resume:
        # If only the filename is provided, prepend the checkpoints directory
        if not os.path.isabs(args.resume) and not args.resume.startswith('./'):
            checkpoint_path = os.path.join(config.checkpoints_dir, args.resume)
        else:
            checkpoint_path = args.resume

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            experiment.model.load_state_dict(checkpoint["model_state"])

            # Optionally restore optimizer and scheduler state too
            if "optimizer_state" in checkpoint and hasattr(experiment, "trainer"):
                experiment.trainer.optimizer.load_state_dict(checkpoint["optimizer_state"])

            if "scheduler_state" in checkpoint and hasattr(experiment, "trainer"):
                experiment.trainer.scheduler.load_state_dict(checkpoint["scheduler_state"])

            print(f"Resumed from checkpoint: {checkpoint_path}")

            # Set current epoch if available
            if "epoch" in checkpoint and hasattr(experiment, "trainer"):
                experiment.trainer.current_epoch = checkpoint["epoch"] + 1
                print(f"Starting from epoch {experiment.trainer.current_epoch}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")

    # Run in selected mode
    if args.mode == "train" or args.mode == "full":
        experiment.train()

    if args.mode == "test" or args.mode == "full":
        experiment.test()

    # Finish W&B run
    if args.wandb and not args.sweep:
        wandb.finish()


if __name__ == "__main__":
    main()