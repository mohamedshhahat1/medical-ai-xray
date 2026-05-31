"""
Model Architecture for Chest X-Ray Classification
===================================================

Defines the neural network architecture using pretrained models
with transfer learning for medical image classification.

Supported architectures:
    - ResNet18/50: Good balance of speed and accuracy
    - DenseNet121: Best for medical imaging (used in CheXNet)
"""

import torch
import torch.nn as nn
from torchvision import models

from config import MODEL_ARCH, NUM_CLASSES


def create_model(arch=MODEL_ARCH, num_classes=NUM_CLASSES, pretrained=True):
    """
    Create a classification model with pretrained backbone.

    Uses transfer learning: loads ImageNet-pretrained weights and
    replaces the final classification layer for X-ray diagnosis.

    Args:
        arch (str): Architecture name ('resnet18', 'resnet50', 'densenet121').
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use ImageNet pretrained weights.

    Returns:
        nn.Module: The classification model.
    """
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, num_classes)
        )

    elif arch == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, num_classes)
        )

    elif arch == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier.in_features, num_classes)
        )

    else:
        raise ValueError(f"Unknown architecture: {arch}. "
                         f"Choose from: resnet18, resnet50, densenet121")

    return model


def load_trained_model(model_path, arch=MODEL_ARCH, num_classes=NUM_CLASSES):
    """
    Load a trained model from a checkpoint file.

    Args:
        model_path (str): Path to .pth file.
        arch (str): Model architecture.
        num_classes (int): Number of classes.

    Returns:
        nn.Module: Loaded model in eval mode.
    """
    model = create_model(arch=arch, num_classes=num_classes, pretrained=False)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model
