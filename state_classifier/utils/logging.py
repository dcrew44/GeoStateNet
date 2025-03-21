"""Logging utilities for training and evaluation."""
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import seaborn as sns
from sklearn.metrics import confusion_matrix


class WandbLogger:
    """Centralized logging interface for Weights & Biases."""

    def __init__(self, config=None):
        """
        Initialize the logger.

        Args:
            config (SimpleNamespace, optional): Configuration to log
        """
        self.config = config

    def log_metrics(self, metrics, step=None, commit=False):
        """
        Log a dictionary of metrics to W&B.

        Args:
            metrics (dict): Dictionary of metrics to log
            step (int, optional): Step or epoch number
            commit (bool): Whether to commit immediately
        """
        wandb.log(metrics, step=step, commit=commit)

    def log_confusion_matrix(self, y_true, y_pred, class_mapping=None, step=None, title="Confusion Matrix"):
        """
        Log a confusion matrix visualization to W&B.

        Args:
            y_true (array-like): Ground truth labels
            y_pred (array-like): Predicted labels
            class_mapping (dict, optional): Mapping from indices to class names
            step (int, optional): Step or epoch number
            title (str): Title for the confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)

        # Get class labels
        if class_mapping:
            labels = [class_mapping.get(idx, str(idx)) for idx in sorted(class_mapping.keys())]
        else:
            labels = [str(i) for i in range(cm.shape[0])]

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(title)

        wandb.log({title: wandb.Image(plt.gcf())}, step=step, commit=False)
        plt.close()

    def log_accuracy_per_state(self, y_true, y_pred, class_mapping=None, step=None, title="Accuracy per state"):
        """
        Log a graph of accuracy per state.

        Args:
            y_true (array-like): Ground truth labels
            y_pred (array-like): Predicted labels
            class_mapping (dict, optional): Mapping from indices to class names
            step (int, optional): Step or epoch number
            title (str): Title for the confusion matrix
        """


    def log_predictions_table(self, preds, labels, class_mapping=None, step=None, table_name="Predictions"):
        """
        Log a table of predictions to W&B.

        Args:
            preds (array-like): Predicted class indices
            labels (array-like): Ground truth class indices
            class_mapping (dict, optional): Mapping from indices to class names
            step (int, optional): Step or epoch number
            table_name (str): Name for the table
        """
        pred_table = wandb.Table(columns=["Sample Index", "Prediction", "Ground Truth"])

        for i, (pred, true) in enumerate(zip(preds, labels)):
            if class_mapping:
                pred_name = class_mapping.get(pred, str(pred))
                true_name = class_mapping.get(true, str(true))
            else:
                pred_name = str(pred)
                true_name = str(true)

            pred_table.add_data(i, pred_name, true_name)

        wandb.log({table_name: pred_table}, step=step, commit=False)

    def log_sample_images(self, images, labels, predictions=None, class_names=None, step=None):
        """
        Log sample images with labels and predictions to W&B.

        Args:
            images (tensor): Batch of images (N, C, H, W)
            labels (tensor): Ground truth labels
            predictions (tensor, optional): Predicted labels
            class_names (list, optional): Class names
            step (int, optional): Step or epoch number
        """
        # Move tensors to CPU
        images_cpu = images.cpu().float()
        labels_cpu = labels.cpu()

        if predictions is not None:
            predictions_cpu = predictions.cpu()

        sample_images = []

        # Unnormalize images
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

        for i in range(min(8, len(images_cpu))):
            img_unnorm = images_cpu[i] * std + mean
            img_unnorm = torch.clamp(img_unnorm, 0, 1).numpy().transpose(1, 2, 0)

            caption = f"True: {class_names[labels_cpu[i]] if class_names else labels_cpu[i]}"
            if predictions is not None:
                caption += f"\nPred: {class_names[predictions_cpu[i]] if class_names else predictions_cpu[i]}"

            sample_images.append(wandb.Image(img_unnorm, caption=caption))

        wandb.log({"sample_images": sample_images}, step=step)