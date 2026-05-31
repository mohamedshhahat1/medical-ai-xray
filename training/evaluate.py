"""
Model Evaluation for Chest X-Ray Classification
=================================================

Comprehensive evaluation metrics for medical AI:
- Accuracy, Precision, Recall, F1-Score
- Sensitivity (true positive rate) — critical for medical
- Specificity (true negative rate)
- ROC curve & AUC
- Confusion matrix
"""

import os
import sys
import numpy as np
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend'))
from config import DEVICE, MODEL_PATH, MODEL_ARCH, NUM_CLASSES, CLASS_NAMES
from model import load_trained_model

from dataset import get_data_loaders


def evaluate_model(model_path=MODEL_PATH):
    """
    Full evaluation of the trained model.

    Prints classification report, confusion matrix, and key medical metrics.
    """
    print("=" * 60)
    print("  🏥 MODEL EVALUATION — Chest X-Ray Classification")
    print("=" * 60)

    # Load model
    print(f"\n  Loading model from: {model_path}")
    model = load_trained_model(model_path)
    model = model.to(DEVICE)
    model.eval()

    # Load validation data
    print("  Loading validation data...")
    _, val_loader, class_names = get_data_loaders()

    if val_loader is None:
        print("  ERROR: No validation data found!")
        return

    # Run inference on all validation data
    all_preds = []
    all_labels = []
    all_probs = []

    print("  Running inference...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Overall accuracy
    accuracy = 100.0 * (all_preds == all_labels).sum() / len(all_labels)
    print(f"\n  Overall Accuracy: {accuracy:.2f}%")
    print(f"  Total samples: {len(all_labels)}")

    # Classification report
    print(f"\n{'─' * 60}")
    print("  CLASSIFICATION REPORT")
    print(f"{'─' * 60}")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names, digits=4))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"{'─' * 60}")
    print("  CONFUSION MATRIX")
    print(f"{'─' * 60}")
    print(f"  {'':>12} {'Pred Normal':>12} {'Pred Pneumonia':>15}")
    print(f"  {'True Normal':>12} {cm[0][0]:>12} {cm[0][1]:>15}")
    print(f"  {'True Pneum.':>12} {cm[1][0]:>12} {cm[1][1]:>15}")

    # Medical metrics
    if len(class_names) == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        print(f"\n{'─' * 60}")
        print("  MEDICAL METRICS")
        print(f"{'─' * 60}")
        print(f"  Sensitivity (Recall):     {sensitivity:.4f}")
        print(f"  Specificity:              {specificity:.4f}")
        print(f"  PPV (Precision):          {ppv:.4f}")
        print(f"  NPV:                      {npv:.4f}")

        # AUC-ROC
        try:
            auc = roc_auc_score(all_labels, all_probs[:, 1])
            print(f"  AUC-ROC:                  {auc:.4f}")
        except Exception:
            pass

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    evaluate_model()
