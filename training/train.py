"""
Training Pipeline for Chest X-Ray Classification
==================================================

Trains a deep learning model to classify chest X-ray images.
Supports transfer learning from ImageNet pretrained models.

Usage:
    python training/train.py
    python training/train.py --epochs 30 --batch-size 64 --arch resnet50
"""

import os
import sys

import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend'))
from config import (DEVICE, EPOCHS, LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY,
                    PATIENCE, MODEL_DIR, MODEL_ARCH, NUM_CLASSES)
from model import create_model

from dataset import get_data_loaders, compute_class_weights


def train(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE, arch=MODEL_ARCH):
    """
    Main training function.

    Args:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        arch (str): Model architecture.
    """
    print("=" * 60)
    print("  🏥 MEDICAL AI — CHEST X-RAY TRAINING")
    print("=" * 60)
    print(f"\n  Architecture: {arch}")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {WEIGHT_DECAY}")
    print()

    # Load data
    print("  Loading dataset...")
    train_loader, val_loader, class_names = get_data_loaders(batch_size=batch_size)
    print()

    # Create model
    print(f"  Creating {arch} model (pretrained=True)...")
    model = create_model(arch=arch, num_classes=len(class_names), pretrained=True)
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable: {trainable:,}")
    print()

    # Loss (class-weighted to handle imbalance), optimizer, scheduler
    print("  Computing class weights for imbalanced data...")
    class_weights = compute_class_weights().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"  ✓ Using weighted loss: {class_weights.cpu().tolist()}")
    print()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    print("-" * 60)
    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # --- Validate ---
        val_acc = 0.0
        if val_loader:
            val_acc = evaluate(model, val_loader)
        history["val_acc"].append(val_acc)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Epoch {epoch}: Loss={train_loss:.4f} | "
              f"Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}% | "
              f"LR={current_lr:.2e}")

        # --- Save best model ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_model(model, optimizer, epoch, val_acc, arch)
            print(f"  ★ New best model! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1

        # --- Early stopping ---
        if patience_counter >= PATIENCE:
            print(f"\n  ⚡ Early stopping at epoch {epoch} (patience={PATIENCE})")
            break

        print("-" * 60)

    print(f"\n{'=' * 60}")
    print(f"  TRAINING COMPLETE — Best Val Acc: {best_val_acc:.2f}%")
    print(f"{'=' * 60}")

    # Save history
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)


def evaluate(model, data_loader):
    """Evaluate model accuracy on a dataset."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100.0 * correct / total


def save_model(model, optimizer, epoch, val_acc, arch):
    """Save model checkpoint."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, "best_model.pth")

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_acc": val_acc,
        "arch": arch,
    }, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Chest X-Ray classifier")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--arch", type=str, default=MODEL_ARCH,
                        choices=["resnet18", "resnet50", "densenet121"])
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size,
          lr=args.lr, arch=args.arch)
