"""
Dataset Loader for Chest X-Ray Images
=======================================

Supports standard folder structure:
    data/raw/train/Normal/*.png
    data/raw/train/Pneumonia/*.png
    data/raw/val/Normal/*.png
    data/raw/val/Pneumonia/*.png

Compatible with:
    - Kaggle Chest X-Ray Pneumonia dataset
    - NIH ChestX-ray14
    - CheXpert
"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend'))
from config import DATA_DIR, BATCH_SIZE, NUM_WORKERS

from transforms import get_train_transforms, get_val_transforms


def get_data_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    Create train and validation data loaders.

    Expects ImageFolder structure:
        data/raw/train/<class_name>/<images>
        data/raw/val/<class_name>/<images>

    Args:
        batch_size (int): Batch size.
        num_workers (int): Parallel data loading workers.

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
    if val_dataset:
        print(f"  Validation samples: {len(val_dataset)}")

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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
