"""Callbacks for training loops."""
import torch
import os
import wandb


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Stops training when a monitored metric has stopped improving.
    """

    def __init__(self, patience=2, delta=0.0):
        """
        Initialize the EarlyStopping object.

        Args:
            patience (int): How many epochs to wait after last improvement
            delta (float): Minimum change in monitored value to qualify as improvement
        """
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        """
        Update internal state based on validation loss.

        Args:
            val_loss (float): The validation loss to evaluate

        Returns:
            bool: True if training should stop
        """
        score = -val_loss  # Higher score is better (negative loss)

        if self.best_score is None:
            # First epoch
            self.best_score = score
        elif score < self.best_score + self.delta:
            # Score didn't improve enough
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Score improved
            self.best_score = score
            self.counter = 0

        return self.early_stop


class ModelCheckpoint:
    """Save model checkpoints during training."""

    def __init__(self, filepath, save_best_only=True, log_to_wandb=True):
        """
        Initialize ModelCheckpoint.

        Args:
            filepath (str): Path where to save the model
            save_best_only (bool): Only save when the monitored metric improves
            log_to_wandb (bool): Whether to log checkpoints to W&B
        """
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.log_to_wandb = log_to_wandb
        self.best_score = float('inf')

    def __call__(self, model, epoch, val_loss, optimizer=None, scheduler=None, scaler=None, config=None):
        """
        Save checkpoint if conditions are met.

        Args:
            model (torch.nn.Module): Model to save
            epoch (int): Current epoch
            val_loss (float): Validation loss
            optimizer (torch.optim.Optimizer, optional): Optimizer state
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler state
            scaler (torch.cuda.amp.GradScaler, optional): Scaler for mixed precision
            config (SimpleNamespace, optional): Configuration object

        Returns:
            bool: Whether a new checkpoint was saved
        """
        is_best = val_loss < self.best_score

        if (not self.save_best_only) or is_best:
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_loss": val_loss,
            }

            if optimizer is not None:
                checkpoint["optimizer_state"] = optimizer.state_dict()

            if scheduler is not None:
                checkpoint["scheduler_state"] = scheduler.state_dict()

            if scaler is not None:
                checkpoint["scaler_state"] = scaler.state_dict()

            if config is not None:
                checkpoint["config"] = config

            torch.save(checkpoint, self.filepath)

            if is_best:
                self.best_score = val_loss
                # Save a copy as best model
                best_path = os.path.join(os.path.dirname(self.filepath), "best_model.pth")
                torch.save(checkpoint, best_path)

                if self.log_to_wandb:
                    artifact = wandb.Artifact(
                        name=f"model-checkpoint-epoch-{epoch}",
                        type="model",
                        metadata={
                            "epoch": epoch,
                            "val_loss": val_loss,
                            "is_best": True
                        }
                    )
                    artifact.add_file(best_path)
                    wandb.log_artifact(artifact, aliases=["latest", "best"])

            return True

        return False