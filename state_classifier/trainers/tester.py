"""Tester class for evaluating state classification model."""
import torch
import wandb
from tqdm import tqdm

from ..utils.constants import get_state_index_to_abbrev
from ..utils.logging import WandbLogger


class Tester:
    """
    Tester for evaluating model performance on a test dataset.

    Handles inference, metric calculation, and result logging.
    """

    def __init__(self, model, test_loader, config):
        """
        Initialize the tester.

        Args:
            model (nn.Module): Model to evaluate
            test_loader (DataLoader): Test data loader
            config (SimpleNamespace): Test configuration
        """
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move model to device
        self.model.to(self.device)

        # Setup logging
        self.logger = WandbLogger(config)

        # Get class names and state abbreviations
        if hasattr(test_loader.dataset, 'classes'):
            self.class_names = test_loader.dataset.classes
        else:
            self.class_names = None

        self.state_abbrev = get_state_index_to_abbrev()

    def test(self, test_step=0):
        """
        Run inference on the test set and log results.

        Args:
            test_step (int): Step number for logging

        Returns:
            float: Test accuracy
        """
        self.model.eval()

        all_predicted = []
        all_labels = []
        results_table = wandb.Table(columns=["id", "predicted", "truth"])
        group_id_counter = 0

        with torch.inference_mode():
            for images, labels, paths in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                # Forward pass
                outputs = self.model(images)

                # Process results by sample groups (each group has 4 images)
                if outputs.shape[0] % 4 != 0:
                    raise ValueError(
                        "Batch size is not a multiple of 4. Check your batch sampler or dataset."
                    )

                # Group outputs for the 4 perspectives of each location
                num_groups = outputs.shape[0] // 4
                outputs_grouped = outputs.view(num_groups, 4, -1)

                # Average predictions across the 4 images in each group
                avg_outputs = outputs_grouped.mean(dim=1)
                _, predicted = torch.max(avg_outputs, dim=1)

                # Group labels (assume all 4 images in a group have the same label)
                labels_grouped = labels.view(num_groups, 4)[:, 0]

                # Build the predictions table
                for i in range(num_groups):
                    sample_id = f"sample_{group_id_counter}"
                    group_id_counter += 1

                    pred_idx = predicted[i].item()
                    label_idx = labels_grouped[i].item()

                    results_table.add_data(
                        sample_id,
                        self.state_abbrev[pred_idx],
                        self.state_abbrev[label_idx]
                    )

                # Collect predictions and labels for metrics
                all_predicted.append(predicted.cpu())
                all_labels.append(labels_grouped.cpu())

        # Concatenate results across batches
        all_predicted = torch.cat(all_predicted, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Compute test accuracy
        total_correct = all_predicted.eq(all_labels).sum().item()
        test_accuracy = total_correct / all_labels.shape[0]

        # Log results
        self.logger.log_metrics({"test/accuracy": test_accuracy}, step=test_step)

        # Log predictions table
        wandb.log({"Test Results Table": results_table}, step=test_step)

        # Log confusion matrix
        y_true = all_labels.numpy()
        y_pred = all_predicted.numpy()
        self.logger.log_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            class_mapping=self.state_abbrev,
            step=test_step,
            title="Test Confusion Matrix"
        )

        print(f"Test Accuracy: {test_accuracy:.4f}")

        return test_accuracy

    def visualize_predictions(self, num_samples=10):
        """
        Visualize model predictions on test samples.

        Args:
            num_samples (int): Number of sample groups to visualize
        """
        self.model.eval()

        # Collect samples to visualize
        sample_images = []
        sample_paths = []
        sample_labels = []
        sample_predictions = []

        with torch.inference_mode():
            for images, labels, paths in self.test_loader:
                batch_size = images.size(0)
                if batch_size % 4 != 0:
                    continue

                num_groups = batch_size // 4

                # Process one batch
                images = images.to(self.device, non_blocking=True)
                outputs = self.model(images)

                # Group outputs and compute predictions
                outputs_grouped = outputs.view(num_groups, 4, -1)
                avg_outputs = outputs_grouped.mean(dim=1)
                _, predicted = torch.max(avg_outputs, dim=1)

                # Group images, labels, and paths
                images_grouped = images.view(num_groups, 4, 3, 224, 224)
                labels_grouped = labels.view(num_groups, 4)[:, 0]
                paths_grouped = [paths[i * 4:(i + 1) * 4] for i in range(num_groups)]

                # Add samples to our collection
                for i in range(min(num_groups, num_samples - len(sample_images))):
                    sample_images.append(images_grouped[i])
                    sample_labels.append(labels_grouped[i])
                    sample_paths.append(paths_grouped[i])
                    sample_predictions.append(predicted[i])

                if len(sample_images) >= num_samples:
                    break

        # Log visualizations
        for i, (images, label, paths, pred) in enumerate(zip(
                sample_images, sample_labels, sample_paths, sample_predictions)):

            # Create a grid of the 4 perspectives
            grid_images = []
            for j in range(4):
                img = images[j].cpu()
                from ..data.transforms import unnormalize_image
                img_unnorm = unnormalize_image(img)
                grid_images.append(wandb.Image(
                    img_unnorm.numpy().transpose(1, 2, 0),
                    caption=f"Perspective {j}"
                ))

            # Get the true and predicted state names
            true_state = self.class_names[label.item()] if self.class_names else self.state_abbrev[label.item()]
            pred_state = self.class_names[pred.item()] if self.class_names else self.state_abbrev[pred.item()]

            # Log the sample
            wandb.log({
                f"test_sample_{i}": {
                    "images": grid_images,
                    "predicted": pred_state,
                    "true": true_state,
                    "correct": pred.item() == label.item()
                }
            })