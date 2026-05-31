"""
Data Augmentation Transforms for X-Ray Images
===============================================

Medical image augmentation must be careful — some transforms
that work for natural images can be harmful for medical images.

Safe augmentations for X-rays:
    ✅ Rotation (small, ±15°)
    ✅ Horizontal flip (lungs are roughly symmetric)
    ✅ Brightness/contrast adjustment
    ✅ Random crop + resize
    ✅ Gaussian blur (simulates scan quality variation)

Unsafe augmentations (avoid):
    ❌ Vertical flip (anatomy has fixed orientation)
    ❌ Color jitter (X-rays are grayscale)
    ❌ Large rotation (>20°, unrealistic for real scans)
"""

from torchvision import transforms

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend'))
from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_train_transforms():
    """
    Augmentation transforms for training data.

    Applies safe medical image augmentations to increase
    dataset diversity and reduce overfitting.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms():
    """
    Transforms for validation/test data (no augmentation).

    Only resize and normalize — we want consistent evaluation.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
