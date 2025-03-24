"""Model architecture definitions for state classification."""
import torch
import torch.nn as nn
from torchvision import models
from itertools import chain
from functools import partial

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

norm_types = (AdaptiveConcatPool2d, nn.BatchNorm2d, nn.BatchNorm1d, nn.BatchNorm3d)

def trainable_params(m):
    return filter(lambda p: p.requires_grad, m.parameters())

def norm_bias_params(m, bias=True):
    if isinstance(m, norm_types): return list(m.parameters())
    res = list(chain.from_iterable(
        norm_bias_params(child, bias=bias) for child in m.children()
    ))
    if bias and getattr(m, 'bias', None): res.extend(m.bias)
    return res

def bn_bias_state(model, bias=True):
    return norm_bias_params(model, bias=bias)

def get_parameter_groups(model, weight_decay=1e-2, train_bn=True):
    # Get batch norm parameters (no weight decay)
    bn_bias_params = [p for p in norm_bias_params(model, bias=True) if p.requires_grad]

    bn_bias_params_ids = set(id(p) for p in bn_bias_params)

    # Get all other parameters (with weight decay)
    other_params = [p for p in model.parameters()
                    if id(p) not in bn_bias_params_ids and p.requires_grad]

    # Create parameter groups
    param_groups = [
        {'params': bn_bias_params, 'weight_decay': 0.0},
        {'params': other_params, 'weight_decay': weight_decay}
    ]

    return param_groups

def init_default(m, func=nn.init.kaiming_normal_):
    "Initialize `m` weights with `func` and set `bias` to 0."
    if func:
        if hasattr(m, 'weight') and m.weight is not None:
            func(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    return m

def cond_init(m, func):
    "Apply `init_default` to `m` unless it's a batchnorm module"
    if not isinstance(m, norm_types):
        init_default(m, func)

def apply_leaf(m, f):
    "Apply `f` to children of `m`."
    for l in m.children():
        apply_leaf(l, f)
    f(m)

def apply_init(m, func=nn.init.kaiming_normal_):
    "Initialize all non-batchnorm layers of `m` with `func`."
    apply_leaf(m, partial(cond_init, func=func))

def unfreeze_bn(model):
    bn_params =  bn_bias_state(model, bias=False)
    for p in bn_params:
        p.requires_grad = True


def build_head():
    head = nn.Sequential(
        # Fastai uses an AdaptiveConcatPool2d to combine max and avg pool outputs.
        AdaptiveConcatPool2d(output_size=1),  # output shape: [batch, fc_in*2, 1, 1]
        nn.Flatten(),  # flatten to shape: [batch, fc_in*2]
        nn.BatchNorm1d(4096, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(0.25),
        nn.Linear(4096, 512, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(0.5),
        nn.Linear(512, 50, bias=False),
    )
    return head

def build_state_classifier(num_classes=50, pretrained=True, dropout_rate=0.0):
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
    model.fc = build_head()

    unfreeze_bn(model)

    apply_init(model.fc)
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