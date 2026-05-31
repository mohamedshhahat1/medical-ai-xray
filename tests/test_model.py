"""Unit tests for model architecture."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import torch
from model import create_model


def test_resnet18_output_shape():
    model = create_model(arch="resnet18", num_classes=2, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 2), f"Expected (2, 2), got {out.shape}"
    print("✓ ResNet18 output shape correct")


def test_resnet50_output_shape():
    model = create_model(arch="resnet50", num_classes=2, pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 2)
    print("✓ ResNet50 output shape correct")


def test_densenet121_output_shape():
    model = create_model(arch="densenet121", num_classes=3, pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 3)
    print("✓ DenseNet121 output shape correct")


if __name__ == "__main__":
    test_resnet18_output_shape()
    test_resnet50_output_shape()
    test_densenet121_output_shape()
    print("\n✅ All model tests passed!")
