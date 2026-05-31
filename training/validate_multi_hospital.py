"""
Multi-Hospital Dataset Validation
====================================

Validates model generalization across multiple hospital/institution datasets.
This is CRITICAL for medical AI — a model trained on one hospital's data
may fail completely on data from another hospital due to differences in:

    - X-ray machine manufacturers (GE, Siemens, Philips)
    - Imaging protocols (kVp, mAs settings)
    - Patient demographics (age, ethnicity distribution)
    - Disease prevalence (high-TB regions vs low-TB regions)
    - Image preprocessing (CLAHE, windowing)

A model that only works on one dataset is NOT deployable.
Cross-dataset validation proves robustness.

Supported Datasets:
    1. COVID-19 Radiography Database (Bangladesh/Qatar) — Primary training
    2. Kaggle Chest X-Ray Pneumonia (Guangzhou, China) — Cross-validation
    3. Montgomery County TB Dataset (USA) — TB validation
    4. Shenzhen Hospital TB Dataset (China) — TB cross-validation
    5. NIH ChestX-ray14 (USA, multi-institution) — Large-scale validation
    6. CheXpert (Stanford, USA) — Stanford Hospital validation

Usage:
    # Validate on all available external datasets
    python training/validate_multi_hospital.py

    # Validate on a specific dataset
    python training/validate_multi_hospital.py --dataset montgomery

    # Add a custom hospital dataset
    python training/validate_multi_hospital.py --custom-dir /path/to/hospital_data

    # Generate validation report
    python training/validate_multi_hospital.py --report
"""

import os
import sys
import json
import argparse
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend'))
from config import DEVICE, MODEL_PATH, MODEL_ARCH, NUM_CLASSES, CLASS_NAMES, MODEL_DIR
from model import load_trained_model

from transforms import get_val_transforms


# =============================================================================
# DATASET REGISTRY — Multi-Hospital Sources
# =============================================================================

DATASET_REGISTRY = {
    "covid_radiography": {
        "name": "COVID-19 Radiography Database",
        "institution": "Qatar University / University of Dhaka / Collaborators",
        "country": "Bangladesh / Qatar",
        "source": "https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database",
        "kaggle_id": "tawsifurrahman/covid19-radiography-database",
        "classes": ["COVID", "Normal", "Pneumonia"],
        "size": "~21,000 images",
        "notes": "Primary training dataset",
    },
    "chest_xray_pneumonia": {
        "name": "Chest X-Ray Pneumonia (Kermany)",
        "institution": "Guangzhou Women and Children's Medical Center",
        "country": "China",
        "source": "https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia",
        "kaggle_id": "paultimothymooney/chest-xray-pneumonia",
        "classes": ["Normal", "Pneumonia"],
        "size": "~5,800 images",
        "notes": "Pediatric patients, different demographic than training data",
    },
    "montgomery": {
        "name": "Montgomery County TB Dataset",
        "institution": "Montgomery County, Department of Health and Human Services",
        "country": "USA",
        "source": "https://openi.nlm.nih.gov/faq#collection",
        "kaggle_id": None,
        "classes": ["Normal", "Tuberculosis"],
        "size": "138 images",
        "notes": "Small but well-curated US hospital data for TB validation",
    },
    "shenzhen": {
        "name": "Shenzhen Hospital TB Dataset",
        "institution": "Shenzhen No.3 People's Hospital",
        "country": "China",
        "source": "https://openi.nlm.nih.gov/faq#collection",
        "kaggle_id": None,
        "classes": ["Normal", "Tuberculosis"],
        "size": "662 images",
        "notes": "Chinese hospital, different equipment and demographics",
    },
    "nih_chestxray": {
        "name": "NIH ChestX-ray14",
        "institution": "National Institutes of Health Clinical Center",
        "country": "USA",
        "source": "https://www.kaggle.com/datasets/nih-chest-xrays/data",
        "kaggle_id": "nih-chest-xrays/data",
        "classes": ["Normal", "Pneumonia", "Infiltration", "Atelectasis"],
        "size": "112,120 images",
        "notes": "Largest public CXR dataset, multi-label, diverse US population",
    },
    "chexpert": {
        "name": "CheXpert",
        "institution": "Stanford University Hospital",
        "country": "USA",
        "source": "https://stanfordmlgroup.github.io/competitions/chexpert/",
        "kaggle_id": None,
        "classes": ["Normal", "Pneumonia", "Cardiomegaly", "Pleural Effusion"],
        "size": "224,316 images",
        "notes": "Stanford Hospital, gold standard for CXR AI validation",
    },
}


# =============================================================================
# VALIDATION ENGINE
# =============================================================================

class MultiHospitalValidator:
    """
    Validates a trained model across multiple hospital datasets.

    Measures:
    - Per-dataset accuracy
    - Cross-dataset AUC-ROC
    - Class-level performance per hospital
    - Generalization gap (training vs external performance)
    """

    def __init__(self, model_path=MODEL_PATH, arch=MODEL_ARCH):
        """
        Initialize validator with a trained model.

        Args:
            model_path (str): Path to trained model weights.
            arch (str): Model architecture name.
        """
        self.model = load_trained_model(model_path, arch=arch, num_classes=NUM_CLASSES)
        self.model = self.model.to(DEVICE)
        self.model.eval()
        self.transform = get_val_transforms()
        self.results = {}

    def validate_dataset(self, dataset_dir, dataset_name="custom"):
        """
        Validate the model on an external dataset.

        Args:
            dataset_dir (str): Path to dataset in ImageFolder format.
            dataset_name (str): Name identifier for this dataset.

        Returns:
            dict: Validation results with accuracy, per-class metrics, etc.
        """
        if not os.path.isdir(dataset_dir):
            print(f"  ⚠️  Dataset not found: {dataset_dir}")
            return None

        print(f"\n  {'─' * 50}")
        print(f"  📊 Validating: {dataset_name}")
        print(f"  {'─' * 50}")

        # Load dataset
        try:
            dataset = datasets.ImageFolder(
                root=dataset_dir,
                transform=self.transform
            )
        except Exception as e:
            print(f"  ❌ Error loading dataset: {e}")
            return None

        if len(dataset) == 0:
            print(f"  ⚠️  Empty dataset: {dataset_dir}")
            return None

        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
        dataset_classes = dataset.classes

        print(f"     Samples: {len(dataset)}")
        print(f"     Classes: {dataset_classes}")

        # Map dataset classes to our model's classes
        class_map = self._map_classes(dataset_classes)
        print(f"     Class mapping: {class_map}")

        # Run inference
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(DEVICE)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Map predictions to dataset's class space for fair comparison
        mapped_preds = self._remap_predictions(all_preds, class_map, dataset_classes)
        mapped_labels = all_labels  # Labels already in dataset's class space

        # Calculate metrics
        accuracy = 100.0 * (mapped_preds == mapped_labels).sum() / len(mapped_labels)

        print(f"     Accuracy: {accuracy:.2f}%")

        # Per-class metrics
        report = classification_report(
            mapped_labels, mapped_preds,
            target_names=dataset_classes,
            output_dict=True, zero_division=0
        )

        # AUC (if possible)
        auc = None
        try:
            if len(dataset_classes) == 2:
                # Map our model's probs to the relevant class
                relevant_idx = self._get_relevant_prob_index(dataset_classes, class_map)
                if relevant_idx is not None:
                    auc = roc_auc_score(mapped_labels, all_probs[:, relevant_idx])
            else:
                auc = roc_auc_score(
                    mapped_labels, all_probs[:, :len(dataset_classes)],
                    multi_class='ovr', average='macro'
                )
        except Exception:
            pass

        if auc:
            print(f"     AUC-ROC: {auc:.4f}")

        result = {
            "dataset_name": dataset_name,
            "dataset_dir": dataset_dir,
            "num_samples": len(dataset),
            "classes": dataset_classes,
            "accuracy": round(accuracy, 2),
            "auc_roc": round(auc, 4) if auc else None,
            "per_class": {
                cls: {
                    "precision": round(report[cls]["precision"], 4),
                    "recall": round(report[cls]["recall"], 4),
                    "f1": round(report[cls]["f1-score"], 4),
                    "support": int(report[cls]["support"]),
                }
                for cls in dataset_classes if cls in report
            },
            "timestamp": datetime.now().isoformat(),
        }

        self.results[dataset_name] = result
        return result

    def validate_all(self, data_root=None):
        """
        Validate on all available external datasets.

        Scans data/ directory for hospital-specific subfolders.

        Args:
            data_root (str): Root data directory. Default: data/external/
        """
        if data_root is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_root = os.path.join(base_dir, "data", "external")

        print("=" * 60)
        print("  🏥 MULTI-HOSPITAL DATASET VALIDATION")
        print("=" * 60)
        print(f"\n  Model: {MODEL_ARCH}")
        print(f"  Classes: {CLASS_NAMES}")
        print(f"  Data root: {data_root}")

        if not os.path.isdir(data_root):
            print(f"\n  ⚠️  External data directory not found: {data_root}")
            print(f"  Create it and add hospital datasets:")
            print(f"    {data_root}/hospital_A/Normal/*.png")
            print(f"    {data_root}/hospital_A/Pneumonia/*.png")
            print(f"    {data_root}/hospital_B/...")
            self._print_available_datasets()
            return

        # Validate each subdirectory as a separate hospital dataset
        hospitals = [d for d in os.listdir(data_root)
                     if os.path.isdir(os.path.join(data_root, d))]

        if not hospitals:
            print(f"\n  ⚠️  No hospital datasets found in {data_root}")
            self._print_available_datasets()
            return

        print(f"\n  Found {len(hospitals)} external dataset(s)")

        for hospital in sorted(hospitals):
            hospital_dir = os.path.join(data_root, hospital)
            # Get display name from registry if available
            info = DATASET_REGISTRY.get(hospital, {})
            display_name = info.get("name", hospital)
            self.validate_dataset(hospital_dir, dataset_name=display_name)

        # Print summary
        self._print_summary()

    def _map_classes(self, dataset_classes):
        """
        Map external dataset classes to our model's class indices.

        Returns dict: {external_class_name: our_model_class_index}
        """
        class_map = {}
        for cls in dataset_classes:
            cls_lower = cls.lower()
            for i, our_cls in enumerate(CLASS_NAMES):
                if our_cls.lower() == cls_lower:
                    class_map[cls] = i
                    break
            else:
                # Fuzzy matching
                if 'normal' in cls_lower or 'healthy' in cls_lower:
                    class_map[cls] = CLASS_NAMES.index("Normal")
                elif 'pneumonia' in cls_lower or 'opacity' in cls_lower:
                    class_map[cls] = CLASS_NAMES.index("Pneumonia")
                elif 'covid' in cls_lower or 'corona' in cls_lower:
                    class_map[cls] = CLASS_NAMES.index("COVID")
                elif 'tb' in cls_lower or 'tuberculosis' in cls_lower:
                    class_map[cls] = CLASS_NAMES.index("Tuberculosis")
                else:
                    class_map[cls] = 0  # Default to first class

        return class_map

    def _remap_predictions(self, predictions, class_map, dataset_classes):
        """Remap model predictions to dataset's class space."""
        # Inverse map: our_class_idx → dataset_class_idx
        inv_map = {}
        for ds_cls, our_idx in class_map.items():
            ds_idx = dataset_classes.index(ds_cls)
            inv_map[our_idx] = ds_idx

        remapped = np.array([inv_map.get(p, 0) for p in predictions])
        return remapped

    def _get_relevant_prob_index(self, dataset_classes, class_map):
        """Get the model probability index for binary classification."""
        if len(dataset_classes) == 2:
            # Return the "positive" class index in our model
            for cls in dataset_classes:
                if cls.lower() != 'normal':
                    return class_map.get(cls, 1)
        return None

    def _print_summary(self):
        """Print a comparison table of all validated datasets."""
        if not self.results:
            return

        print(f"\n\n{'═' * 60}")
        print("  📋 MULTI-HOSPITAL VALIDATION SUMMARY")
        print(f"{'═' * 60}")
        print(f"\n  {'Dataset':<35} {'Accuracy':>10} {'AUC-ROC':>10} {'Samples':>10}")
        print(f"  {'─' * 35} {'─' * 10} {'─' * 10} {'─' * 10}")

        for name, result in self.results.items():
            acc = f"{result['accuracy']:.1f}%"
            auc = f"{result['auc_roc']:.4f}" if result['auc_roc'] else "N/A"
            n = str(result['num_samples'])
            print(f"  {name:<35} {acc:>10} {auc:>10} {n:>10}")

        # Generalization assessment
        accuracies = [r['accuracy'] for r in self.results.values()]
        if len(accuracies) >= 2:
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            print(f"\n  {'─' * 65}")
            print(f"  Mean Accuracy:  {mean_acc:.1f}% ± {std_acc:.1f}%")

            if std_acc < 5:
                print(f"  Assessment:     ✅ GOOD generalization (low variance)")
            elif std_acc < 10:
                print(f"  Assessment:     ⚠️  MODERATE generalization (some variance)")
            else:
                print(f"  Assessment:     ❌ POOR generalization (high variance)")
                print(f"                  Model may be overfitting to training hospital's data")

        print(f"\n{'═' * 60}")

    def _print_available_datasets(self):
        """Print available datasets from the registry."""
        print(f"\n  📚 Available datasets for validation:")
        print(f"  {'─' * 55}")
        for key, info in DATASET_REGISTRY.items():
            print(f"    • {info['name']}")
            print(f"      Institution: {info['institution']} ({info['country']})")
            print(f"      Source: {info['source']}")
            print()

    def save_report(self, output_path=None):
        """
        Save validation results as JSON report.

        Args:
            output_path (str): Output file path.
        """
        if output_path is None:
            os.makedirs(MODEL_DIR, exist_ok=True)
            output_path = os.path.join(MODEL_DIR, "multi_hospital_validation.json")

        report = {
            "model": MODEL_ARCH,
            "model_path": MODEL_PATH,
            "classes": CLASS_NAMES,
            "validation_date": datetime.now().isoformat(),
            "datasets": self.results,
            "summary": {
                "num_datasets": len(self.results),
                "mean_accuracy": round(np.mean([r['accuracy'] for r in self.results.values()]), 2)
                if self.results else 0,
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n  📄 Report saved to: {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate model on multiple hospital datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python training/validate_multi_hospital.py
  python training/validate_multi_hospital.py --custom-dir /path/to/hospital_data
  python training/validate_multi_hospital.py --report
  python training/validate_multi_hospital.py --list-datasets

Dataset setup:
  Place external hospital data in: data/external/<hospital_name>/<class>/images
  Example:
    data/external/montgomery/Normal/*.png
    data/external/montgomery/Tuberculosis/*.png
    data/external/guangzhou/Normal/*.png
    data/external/guangzhou/Pneumonia/*.png
        """
    )

    parser.add_argument("--custom-dir", type=str, default=None,
                        help="Path to a custom hospital dataset (ImageFolder format)")
    parser.add_argument("--custom-name", type=str, default="Custom Hospital",
                        help="Name for the custom dataset")
    parser.add_argument("--report", action="store_true",
                        help="Save validation report as JSON")
    parser.add_argument("--list-datasets", action="store_true",
                        help="List all registered datasets")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Root directory for external datasets (default: data/external/)")

    args = parser.parse_args()

    if args.list_datasets:
        print("\n  📚 Registered Hospital Datasets:\n")
        for key, info in DATASET_REGISTRY.items():
            print(f"  [{key}]")
            print(f"    Name:        {info['name']}")
            print(f"    Institution: {info['institution']}")
            print(f"    Country:     {info['country']}")
            print(f"    Classes:     {info['classes']}")
            print(f"    Size:        {info['size']}")
            print(f"    Source:      {info['source']}")
            print()
        return

    # Initialize validator
    validator = MultiHospitalValidator()

    if args.custom_dir:
        # Validate single custom dataset
        validator.validate_dataset(args.custom_dir, dataset_name=args.custom_name)
    else:
        # Validate all available external datasets
        validator.validate_all(data_root=args.data_root)

    if args.report:
        validator.save_report()


if __name__ == "__main__":
    main()
