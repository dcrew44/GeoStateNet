# Model Checkpoints

This directory stores model checkpoints during training. These files are not included in the repository due to their size.

## Expected Files

During training, the following files will be generated:

- `checkpoint_epoch_{N}.pth`: Checkpoint after epoch N
- `best_model.pth`: Best performing model based on validation loss
- `best_phase1.pth`: Best model from phase 1 (head-only training)
- `best_phase2.pth`: Best model from phase 2 (fine-tuning)

## Checkpoint Structure

Each checkpoint file is a dictionary containing:
- `model_state`: Model weights
- `optimizer_state`: Optimizer state
- `scheduler_state`: Learning rate scheduler state
- `epoch`: Epoch number
- `val_loss`: Validation loss
- `scaler_state`: Gradient scaler state for mixed precision training

## Weights & Biases Integration

Checkpoints are also logged to Weights & Biases as artifacts for better experiment tracking.

## Loading Checkpoints

To load a checkpoint:

```python
checkpoint = torch.load("checkpoints/best_model.pth")
model.load_state_dict(checkpoint["model_state"])
```