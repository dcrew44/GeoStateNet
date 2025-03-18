"""Model architecture definitions for state classification."""
import torch
import torch.nn as nn
from torchvision import models


class AdaptiveConcatPool2d(nn.Module):
    """
    Applies both adaptive max pooling and adaptive average pooling, then concatenates them.
    This can help preserve more information for the classifier head.
    """

    def __init__(self, output_size=1):
        """
        Initialize the module.

        Args:
            output_size (int or tuple): Size of the output
        """
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(output_size)
        self.mp = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, x):
        """Forward pass."""
        return torch.cat([self.mp(x), self.ap(x)], dim=1)




def build_state_classifier(num_classes=50, pretrained=True, dropout_rate=0):
    """
    Build a ResNet101 model for state classification with AdaptiveConcatPool2d
    that preserves the original layer structure.
    """
    # Load pretrained model
    if pretrained:
        model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
    else:
        model = models.resnet101(weights=None)

    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    # Get the number of features
    fc_in = model.fc.in_features  # typically 2048 for resnet101

    # Define a custom forward function that ensures proper data flow
    def custom_forward(x, model):
        # Pass x through the body (up to layer4)
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        # Do NOT flatten x here; pass the 4D tensor directly to the head
        x = model.fc(x)
        return x

    # Replace the classifier head
    model.avgpool = nn.Identity()
    model.fc = nn.Sequential(
        # Fastai uses an AdaptiveConcatPool2d to combine max and avg pool outputs.
        AdaptiveConcatPool2d(output_size=1),  # output shape: [batch, fc_in*2, 1, 1]
        nn.Flatten(),  # flatten to shape: [batch, fc_in*2]
        nn.BatchNorm1d(fc_in * 2, ),
        nn.Dropout(dropout_rate),
        nn.Linear(fc_in * 2, 512, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(dropout_rate),
        nn.Linear(512, num_classes, bias=False),
    )

    # Override the forward method
    model.forward = lambda x: custom_forward(x, model)

    return model

def unfreeze_model_layers(model, freeze_conv1=True, freeze_bn1=True, freeze_layer1=True,
                          freeze_layer2=False, freeze_layer3=False, freeze_layer4=False):
    """
    Selectively unfreeze layers of a ResNet model for fine-tuning.

    Args:
        model (nn.Module): ResNet model
        freeze_conv1 (bool): Whether to keep first conv layer frozen
        freeze_bn1 (bool): Whether to keep first batch norm layer frozen
        freeze_layer1-4 (bool): Whether to keep respective layers frozen

    Returns:
        nn.Module: Model with selected layers unfrozen
    """
    # First unfreeze everything
    for param in model.parameters():
        param.requires_grad = True

    # Then freeze selected layers
    if freeze_conv1:
        for param in model.conv1.parameters():
            param.requires_grad = False

    if freeze_bn1:
        for param in model.bn1.parameters():
            param.requires_grad = False

    for param in model.maxpool.parameters():
        param.requires_grad = False

    if freeze_layer1:
        for param in model.layer1.parameters():
            param.requires_grad = False

    if freeze_layer2:
        for param in model.layer2.parameters():
            param.requires_grad = False

    if freeze_layer3:
        for param in model.layer3.parameters():
            param.requires_grad = False

    if freeze_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = False

    # Always ensure the head is trainable
    for param in model.fc.parameters():
        param.requires_grad = True

    return model