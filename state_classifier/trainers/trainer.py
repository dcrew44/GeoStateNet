"""Trainer class for state classification model."""
import os
import shutil
import sys
import random
import signal
import torch
import torch.nn as nn
import wandb
from torch.amp import GradScaler
from tqdm import tqdm

from ..utils.callbacks import EarlyStopping
from ..utils.constants import get_state_index_to_abbrev
from ..utils.logging import WandbLogger
from ..models.classifier import unfreeze_model_layers, get_parameter_groups, unfreeze_bn


class Trainer:
    """
    Trainer for the state classification model.

    Handles training loop, validation, and checkpointing.
    """

    def __init__(self, model, train_loader, val_loader, config):
        """
        Initialize the trainer.

        Args:
            model (nn.Module): Model to train
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            config (SimpleNamespace): Training configuration
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup components
        self.optimizer = self._build_optimizer()
        self.loss_fn = self._build_loss()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler(device=self.device)
        # Setup monitoring
        self.early_stopping = EarlyStopping(
            patience=config.hyperparameters.patience,
            delta=config.hyperparameters.early_stopping_delta
        )

        # Setup logging
        self.logger = WandbLogger(config)

        # Get class names safely by traversing the dataset chain
        self.class_names = self._get_class_names(val_loader.dataset)
        self.state_abbrev = get_state_index_to_abbrev()

        # Training state
        self.current_epoch = 0
        self.best_loss = float("inf")

        # Register signal handler for graceful interruption
        signal.signal(signal.SIGINT, self.interrupt_handler)

    def _get_class_names(self, dataset):
        """
        Safely extract class names from a dataset, handling wrapper cases.

        Args:
            dataset: A dataset object that might be wrapped

        Returns:
            list: Class names if found, or empty list
        """
        # Try to get classes directly
        if hasattr(dataset, 'classes'):
            return dataset.classes

        # Try to get from dataset.dataset (for Subset)
        if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'classes'):
            return dataset.dataset.classes

        # Try to go deeper (for nested wrappers)
        if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'dataset') and hasattr(dataset.dataset.dataset,
                                                                                           'classes'):
            return dataset.dataset.dataset.classes

        # If all else fails, return empty list and log warning
        print("WARNING: Could not find class names in dataset. Using numeric indices instead.")
        return []

    def _build_optimizer(self, lr=0.001, betas=(0.9, 0.99), eps=1e-5):
        """Build optimizer with separate param groups for all BatchNorm layers."""
        # Create parameter groups with and without weight decay
        param_groups = get_parameter_groups(self.model)

        return torch.optim.Adam(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps
            )

    def _build_loss(self):
        """
        Build loss function.

        Returns:
            nn.Module: Loss function
        """
        return nn.CrossEntropyLoss()

    def _build_scheduler(self, lr=0.001, betas=(0.85, 0.95), epochs=10):
        """
        Build learning rate scheduler.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler
        """
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=lr,
            total_steps=len(self.train_loader) * epochs,
            pct_start=0.25,
            div_factor=25,
            final_div_factor=100000,
            base_momentum=betas[0],
            max_momentum=betas[1],
            cycle_momentum=True
        )

    def save_checkpoint(self, epoch, is_best=False):
        """
        Save checkpoint with model and optimizer state.

        Args:
            epoch (int): Current epoch
            is_best (bool): Whether this is the best model so far
        """
        # Ensure checkpoint directory exists
        os.makedirs(self.config.checkpoints_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "val_loss": self.best_loss,
            "parameters": self.config.hyperparameters.__dict__,
        }

        # Save to file
        filename = os.path.join(self.config.checkpoints_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, filename)

        if is_best:
            best_filename = os.path.join(self.config.checkpoints_dir, "best_model.pth")
            torch.save(checkpoint, best_filename)

            # Log to W&B if not in sweep mode
            if not getattr(self.config, "is_sweep", False):
                artifact = wandb.Artifact(
                    name=f"model-checkpoint-epoch-{epoch}",
                    type="model",
                    metadata={
                        "epoch": epoch,
                        "val_loss": self.best_loss,
                        "is_best": True
                    }
                )
                artifact.add_file(best_filename)
                wandb.log_artifact(artifact, aliases=["latest", "best"])

    def interrupt_handler(self, signum, frame):
        """
        Handle interrupt signal (Ctrl+C) by saving checkpoint.

        Args:
            signum: Signal number
            frame: Current stack frame
        """
        print("\nInterrupt received, saving checkpoint...")
        interrupt_path = os.path.join(self.config.checkpoints_dir, f"interrupt_epoch_{self.current_epoch}.pth")

        # Ensure directory exists
        os.makedirs(self.config.checkpoints_dir, exist_ok=True)

        # Save checkpoint
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
        }
        torch.save(checkpoint, interrupt_path)

        print(f"Saved interrupt checkpoint to {interrupt_path}")
        sys.exit(0)

    def train_one_epoch(self, epoch):
        """
        Train for one epoch.

        Args:
            epoch (int): Current epoch

        Returns:
            float: Training loss for this epoch
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Determine if we should log images this epoch
        should_log_images = (epoch % 3 == 0) and not getattr(self.config, "is_sweep", False)
        selected_batch = None

        for batch_idx, (images, labels) in enumerate(tqdm(self.train_loader, desc=f"Training [{epoch + 1}]")):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            with torch.autocast(device_type="cuda"):
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Update learning rate
            self.scheduler.step()

            # Update statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            # Randomly select a batch for visualization
            if should_log_images and not selected_batch and random.random() < 0.1:
                selected_batch = (images, labels)

        # Log sample images if we have a selected batch
        if should_log_images and selected_batch:
            with torch.no_grad():
                outputs = self.model(selected_batch[0])
                _, predicted = outputs.max(1)

            self.logger.log_sample_images(
                images=selected_batch[0],
                labels=selected_batch[1],
                predictions=predicted,
                class_names=self.class_names,
                step=epoch
            )

        # Calculate epoch metrics
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        # Log metrics
        metrics = {
            "train/loss": epoch_loss,
            "train/accuracy": epoch_acc,
            "train/lr": self.scheduler.get_last_lr()[0]
        }
        self.logger.log_metrics(metrics, step=epoch)

        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%")

        return epoch_loss

    def validate_one_epoch(self, epoch):
        """
        Validate for one epoch.

        Args:
            epoch (int): Current epoch

        Returns:
            float: Validation loss for this epoch
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        # Determine if we should log confusion matrix
        should_log_cm = (epoch % 3 == 0) and not getattr(self.config, "is_sweep", False)

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc=f"Validating [{epoch + 1}]"):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Forward pass with mixed precision
                with torch.autocast(device_type="cuda"):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, labels)

                # Update statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)

                # Collect predictions and labels for confusion matrix
                # all_preds.extend(predicted.cpu().numpy())
                # all_labels.extend(labels.cpu().numpy())

        # Calculate epoch metrics
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        # Log metrics
        metrics = {
            "val/loss": epoch_loss,
            "val/accuracy": epoch_acc
        }

        self.logger.log_metrics(metrics, step=epoch, commit=True)

        # Log confusion matrix and predictions table if needed
        # if should_log_cm:
        #     self.logger.log_confusion_matrix(
        #         y_true=all_labels,
        #         y_pred=all_preds,
        #         class_mapping=self.state_abbrev,
        #         step=epoch,
        #         title="Validation Confusion Matrix"
        #     )
        #
        #     self.logger.log_predictions_table(
        #         preds=all_preds,
        #         labels=all_labels,
        #         class_mapping=self.state_abbrev,
        #         step=epoch,
        #         table_name="Validation Predictions"
        #     )

        print(f"Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.2f}%")

        return epoch_loss

    def train_phase(self, phase=1, start_step=0, phase_epochs=10):
        self.model.to(self.device)

        print(f"=== Phase {phase} ===")
        self.current_epoch = start_step
        best_loss = float("inf")

        for epoch in range(start_step, phase_epochs):
            self.current_epoch = epoch
            self.train_one_epoch(epoch)
            val_loss = self.validate_one_epoch(epoch)

            # Save checkpoint if this is the best model so far
            if val_loss < best_loss:
                best_loss = val_loss
                self.best_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

                # Save a phase-specific checkpoint
                phase_path = os.path.join(self.config.checkpoints_dir, f"best_phase{phase}.pth")
                torch.save(self.model.state_dict(), phase_path)

            # Check for early stopping
            if self.early_stopping(val_loss):
                print(f"Early stopping triggered in phase {phase}")
                break

        # Load best model from phase
        phase_path = os.path.join(self.config.checkpoints_dir, f"best_phase{phase}.pth")
        if os.path.exists(phase_path):
            self.model.load_state_dict(torch.load(phase_path))
            print(f"Loaded best Phase {phase} weights.")



    def train(self):
        """
        Full training process with two phases:
        1. Train only the classifier head
        2. Fine-tune selected layers
        """
        # Move model to device
        self.model.to(self.device)

        start_step = getattr(self.config, "start_step", 0)

        self.current_epoch = 0

        if self.config.train_phases.start_phase == 1:
            phase1_epochs = getattr(self.config.hyperparameters, "phase1_epochs", 10)
            phase1_lr = getattr(self.config.hyperparameters, "phase1_lr", 1e-3)

            self.optimizer = self._build_optimizer(lr=phase1_lr)
            self.scheduler = self._build_scheduler(lr=phase1_lr, epochs=phase1_epochs)
            self.scaler = GradScaler(device=self.device)
            self.early_stopping = EarlyStopping(
                patience=self.config.hyperparameters.patience,
                delta=self.config.hyperparameters.early_stopping_delta
            )

            if wandb:
                wandb.watch(self.model, log="all", log_freq=1000)
            self.train_phase(phase=1, start_step=start_step, phase_epochs=phase1_epochs)

        start_step = self.current_epoch

        if self.config.train_phases.start_phase == 2 or self.config.train_phases.phase_2:
            phase2_epochs = getattr(self.config.hyperparameters, "phase2_epochs", 5)
            phase2_lr = getattr(self.config.hyperparameters, "phase2_lr", 1e-3)

            unfreeze_model_layers(
                self.model,
                freeze_conv1=True,
                freeze_bn1=True,
                freeze_layer1=True,
                freeze_layer2=True,
                freeze_layer3=True,
                freeze_layer4=False
            )
            self.optimizer = self._build_optimizer(phase2_lr)
            self.scheduler = self._build_scheduler(lr=phase2_lr, epochs=phase2_epochs)
            self.scaler = GradScaler(device=self.device)
            self.early_stopping = EarlyStopping(
                patience=self.config.hyperparameters.patience,
                delta=self.config.hyperparameters.early_stopping_delta
            )

            if wandb:
                wandb.watch(self.model, log="all", log_freq=1000)
            self.train_phase(phase=2, start_step=start_step, phase_epochs=phase2_epochs)
        start_step = self.current_epoch

        if self.config.train_phases.start_phase == 3 or self.config.train_phases.phase_3:
            phase3_epochs = getattr(self.config.hyperparameters, "phase3_epochs", 5)
            phase3_lr = getattr(self.config.hyperparameters, "phase3_lr", 1e-3)

            unfreeze_model_layers(
                self.model,
                freeze_conv1=True,
                freeze_bn1=True,
                freeze_layer1=True,
                freeze_layer2=False,
                freeze_layer3=False,
                freeze_layer4=False
            )
            self.optimizer = self._build_optimizer(phase3_lr)
            self.scheduler = self._build_scheduler(lr=phase3_lr, epochs=phase3_epochs)
            self.scaler = GradScaler(device=self.device)
            self.early_stopping = EarlyStopping(
                patience=self.config.hyperparameters.patience,
                delta=self.config.hyperparameters.early_stopping_delta
            )

            if wandb:
                wandb.watch(self.model, log="all", log_freq=1000)
            self.train_phase(phase=3, start_step=start_step, phase_epochs=phase3_epochs)
        self.log_final_best_artifact()

    def log_final_best_artifact(self):
        """Log the final best model as a W&B artifact."""
        best_checkpoint_path = os.path.join(self.config.checkpoints_dir, "best_model.pth")

        # Create a descriptive name
        new_name = os.path.join(self.config.checkpoints_dir, f"best_model_run_{wandb.run.id}.pth")

        # Make a copy with the run ID in the name
        if os.path.exists(best_checkpoint_path):
            shutil.copy(best_checkpoint_path, new_name)

            # Create artifact
            artifact = wandb.Artifact(
                name=f"resnet101_run_{wandb.run.id}_valLoss_{self.best_loss:.4f}",
                type="model",
                metadata={"val_loss": self.best_loss, "run_id": wandb.run.id}
            )

            # Add file and log
            artifact.add_file(new_name)
            wandb.log_artifact(artifact)
        else:
            print(f"Warning: Could not find best model checkpoint at {best_checkpoint_path}")