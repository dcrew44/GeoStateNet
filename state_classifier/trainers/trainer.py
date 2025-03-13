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
from ..models.classifier import unfreeze_model_layers


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
        self.class_names = self.val_loader.dataset.dataset.classes
        self.state_abbrev = get_state_index_to_abbrev()

        # Training state
        self.current_epoch = 0
        self.best_loss = float("inf")

        # Register signal handler for graceful interruption
        signal.signal(signal.SIGINT, self.interrupt_handler)

    def _build_optimizer(self):
        """
        Build optimizer based on configuration.

        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        if self.config.hyperparameters.optimizer == "Adam":
            return torch.optim.AdamW(
                trainable_params,
                lr=self.config.hyperparameters.lr,
                weight_decay=self.config.hyperparameters.weight_decay,
                betas=self.config.hyperparameters.betas,
            )
        elif self.config.hyperparameters.optimizer == "SGD":
            return torch.optim.SGD(
                trainable_params,
                lr=self.config.hyperparameters.lr,
                momentum=self.config.hyperparameters.momentum,
                weight_decay=self.config.hyperparameters.weight_decay
            )
        else:
            return torch.optim.AdamW(
                trainable_params,
                lr=self.config.hyperparameters.lr,
                weight_decay=self.config.hyperparameters.weight_decay
            )

    def _build_loss(self):
        """
        Build loss function.

        Returns:
            nn.Module: Loss function
        """
        return nn.CrossEntropyLoss()

    def _build_scheduler(self):
        """
        Build learning rate scheduler.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: Learning rate scheduler
        """
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=self.config.hyperparameters.lr,
            total_steps=len(self.train_loader) * self.config.hyperparameters.num_epochs,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1000,
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
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate epoch metrics
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        # Log metrics
        metrics = {
            "val/loss": epoch_loss,
            "val/accuracy": epoch_acc
        }
        self.logger.log_metrics(metrics, step=epoch)

        # Log confusion matrix and predictions table if needed
        if should_log_cm:
            self.logger.log_confusion_matrix(
                y_true=all_labels,
                y_pred=all_preds,
                class_mapping=self.state_abbrev,
                step=epoch,
                title="Validation Confusion Matrix"
            )

            self.logger.log_predictions_table(
                preds=all_preds,
                labels=all_labels,
                class_mapping=self.state_abbrev,
                step=epoch,
                table_name="Validation Predictions"
            )

        print(f"Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.2f}%")

        return epoch_loss

    def train(self):
        """
        Full training process with two phases:
        1. Train only the classifier head
        2. Fine-tune selected layers
        """
        # Move model to device
        self.model.to(self.device)

        # === Phase 1: Head-only Training ===
        print("=== Phase 1: Head-only Training ===")
        phase1_epochs = getattr(self.config.hyperparameters, "num_epochs", 10)
        self.current_epoch = 0
        best_loss_phase1 = float("inf")

        for epoch in range(phase1_epochs):
            self.current_epoch = epoch
            self.train_one_epoch(epoch)
            val_loss = self.validate_one_epoch(epoch)

            # Save checkpoint if this is the best model so far
            if val_loss < best_loss_phase1:
                best_loss_phase1 = val_loss
                self.best_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)

                # Save a phase-specific checkpoint
                phase1_path = os.path.join(self.config.checkpoints_dir, "best_phase1.pth")
                torch.save(self.model.state_dict(), phase1_path)

            # Check for early stopping
            if self.early_stopping(val_loss):
                print("Early stopping triggered in phase 1")
                break

        # Load best model from phase 1
        phase1_path = os.path.join(self.config.checkpoints_dir, "best_phase1.pth")
        if os.path.exists(phase1_path):
            self.model.load_state_dict(torch.load(phase1_path))
            print("Loaded best Phase 1 weights.")

        # === Phase 2: Fine-tuning Selected Layers ===
        print("=== Phase 2: Fine-tuning Selected Layers ===")

        # Unfreeze selected layers
        unfreeze_model_layers(
            self.model,
            freeze_conv1=True,
            freeze_bn1=True,
            freeze_layer1=True,
            freeze_layer2=True,
            freeze_layer3=False,
            freeze_layer4=False
        )

        # Update optimizer with new learning rate for fine-tuning
        finetune_lr = getattr(self.config.hyperparameters, "finetune_lr", 1e-4)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=finetune_lr,
            weight_decay=self.config.hyperparameters.finetune_weight_decay,
            betas=(0.9, 0.999)
        )

        # Update scheduler for phase 2
        phase2_epochs = getattr(self.config.hyperparameters, "phase2_epochs", 5)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            total_steps=len(self.train_loader) * phase2_epochs,
            max_lr=finetune_lr,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1000,
        )

        # Reset early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.hyperparameters.patience,
            delta=self.config.hyperparameters.early_stopping_delta
        )

        # Train for phase 2
        best_loss_phase2 = float("inf")
        for epoch in range(phase2_epochs):
            phase2_epoch = epoch + phase1_epochs
            self.current_epoch = phase2_epoch

            self.train_one_epoch(phase2_epoch)
            val_loss = self.validate_one_epoch(phase2_epoch)

            # Save checkpoint if this is the best model so far
            if val_loss < best_loss_phase2:
                best_loss_phase2 = val_loss
                self.best_loss = val_loss
                self.save_checkpoint(phase2_epoch, is_best=True)

                # Save a phase-specific checkpoint
                phase2_path = os.path.join(self.config.checkpoints_dir, "best_phase2.pth")
                torch.save(self.model.state_dict(), phase2_path)

            # Check for early stopping
            if self.early_stopping(val_loss):
                print("Early stopping triggered in phase 2")
                break

        # Load best model from phase 2 (or keep phase 1 if phase 2 didn't improve)
        phase2_path = os.path.join(self.config.checkpoints_dir, "best_phase2.pth")
        if os.path.exists(phase2_path) and best_loss_phase2 < best_loss_phase1:
            self.model.load_state_dict(torch.load(phase2_path))
            print("Loaded best Phase 2 weights.")

        # Log final best model as artifact
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