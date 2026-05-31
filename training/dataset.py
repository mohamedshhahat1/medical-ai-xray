"""
Dataset Loader for Chest X-Ray Images
=======================================

Supports standard folder structure:
    data/raw/train/Normal/*.png
    data/raw/train/Pneumonia/*.png
    data/raw/val/Normal/*.png
    data/raw/val/Pneumonia/*.png

Handles class imbalance (common in medical datasets) via:
    1. Weighted Random Sampling (oversampling minority class)
    2. Class-weighted loss (computed from dataset statistics)

Compatible with:
    - Kaggle Chest X-Ray Pneumonia dataset
    - NIH ChestX-ray14
    - CheXpert
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from collections import Counter

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend'))
from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS, DEVICE

from transforms import get_train_transforms, get_val_transforms


def get_data_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                     use_oversampling=True):
    """
    Create train and validation data loaders with class imbalance handling.

    Expects ImageFolder structure:
        data/raw/train/<class_name>/<images>
        data/raw/val/<class_name>/<images>

    Args:
        batch_size (int): Batch size.
        num_workers (int): Parallel data loading workers.
        use_oversampling (bool): Whether to oversample minority class.
            Default True — essential for imbalanced medical datasets.

    Returns:
        tuple: (train_loader, val_loader, class_names)
    """
    train_dir = os.path.join(DATA_DIR, "raw", "train")
    val_dir = os.path.join(DATA_DIR, "raw", "val")

    # Validate directories exist
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"Training data not found at: {train_dir}\n"
            f"Download a chest X-ray dataset and organize as:\n"
            f"  {train_dir}/Normal/*.png\n"
            f"  {train_dir}/Pneumonia/*.png\n\n"
            f"Recommended dataset: Kaggle Chest X-Ray Pneumonia\n"
            f"  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
        )

    # Create datasets using ImageFolder (auto-detects classes from subfolders)
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=get_train_transforms()
    )

    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=get_val_transforms()
    ) if os.path.isdir(val_dir) else None

    # Class names from folder structure
    class_names = train_dataset.classes
    print(f"  Classes: {class_names}")
    print(f"  Training samples: {len(train_dataset)}")

    # --- Class distribution analysis ---
    class_counts = Counter(train_dataset.targets)
    print(f"  Class distribution:")
    for cls_idx, count in sorted(class_counts.items()):
        pct = 100.0 * count / len(train_dataset)
        print(f"    {class_names[cls_idx]}: {count} ({pct:.1f}%)")

    # Check imbalance ratio
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else 1
    if imbalance_ratio > 1.5:
        print(f"  ⚠️  Imbalance ratio: {imbalance_ratio:.1f}:1")
        print(f"      → Using oversampling + weighted loss to compensate")
    else:
        print(f"  ✓ Dataset is balanced (ratio: {imbalance_ratio:.1f}:1)")

    if val_dataset:
        print(f"  Validation samples: {len(val_dataset)}")

    # --- Oversampling via WeightedRandomSampler ---
    sampler = None
    shuffle = True

    if use_oversampling and imbalance_ratio > 1.5:
        # Assign higher weight to minority class samples
        # Weight per sample = 1 / (count of its class)
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        sample_weights = [class_weights[label] for label in train_dataset.targets]
        sample_weights = torch.FloatTensor(sample_weights)

        # WeightedRandomSampler draws samples proportional to their weight
        # num_samples = len(dataset) means one "epoch" is still same length
        # but minority class is sampled more frequently
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True  # Allow duplicate sampling of minority class
        )
        shuffle = False  # Sampler and shuffle are mutually exclusive
        print(f"  ✓ Oversampling enabled (WeightedRandomSampler)")

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, class_names


def compute_class_weights(data_dir=None):
    """
    Compute class weights inversely proportional to class frequency.

    Used with nn.CrossEntropyLoss(weight=...) to penalize misclassification
    of the minority class more heavily.

    Formula:
        weight[i] = total_samples / (num_classes * count[i])

    For Kaggle Chest X-Ray (Normal=1341, Pneumonia=3875):
        weight[Normal]    = 5216 / (2 * 1341) = 1.945
        weight[Pneumonia] = 5216 / (2 * 3875) = 0.673

    This means a Normal sample misclassified costs ~3x more than
    a Pneumonia sample misclassified — forcing the model to pay
    attention to the minority class.

    Args:
        data_dir (str): Path to training data directory.

    Returns:
        torch.Tensor: Class weights tensor (move to DEVICE before use).
    """
    if data_dir is None:
        data_dir = os.path.join(DATA_DIR, "raw", "train")

    if not os.path.isdir(data_dir):
        # Return uniform weights if no data
        print("  ⚠️  No training data found, using uniform class weights")
        return torch.ones(2)

    # Count samples per class
    dataset = datasets.ImageFolder(root=data_dir)
    class_counts = Counter(dataset.targets)
    num_classes = len(class_counts)
    total_samples = len(dataset)

    # Compute inverse frequency weights
    weights = []
    for cls_idx in range(num_classes):
        count = class_counts.get(cls_idx, 1)
        w = total_samples / (num_classes * count)
        weights.append(w)

    weights_tensor = torch.FloatTensor(weights)

    # Print for transparency
    class_names = dataset.classes
    print(f"  Class weights (inverse frequency):")
    for i, (name, w) in enumerate(zip(class_names, weights)):
        print(f"    {name}: {w:.4f}")

    return weights_tensor
