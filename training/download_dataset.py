"""
Automatic Dataset Download from Kaggle
========================================

Downloads and extracts the Chest X-Ray Pneumonia dataset from Kaggle.
Handles authentication, download, extraction, and directory organization.

Supported methods:
    1. Kaggle API (pip install kaggle) — fastest, needs API key
    2. Direct URL download (opendatasets) — alternative
    3. Manual instructions — fallback if no API key

Setup Kaggle API Key:
    1. Go to https://www.kaggle.com/settings → Create New Token
    2. Download kaggle.json
    3. Place at ~/.kaggle/kaggle.json (Linux/Mac)
       or set KAGGLE_USERNAME + KAGGLE_KEY environment variables

Usage:
    python training/download_dataset.py
    # or
    from training.download_dataset import download_chest_xray
    download_chest_xray()
"""

import os
import sys

import shutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'backend'))
from config import DATA_DIR


# Kaggle dataset identifiers (multi-disease)
# Primary: COVID-19 Radiography Database (3 classes: COVID, Normal, Pneumonia)
KAGGLE_DATASET = "tawsifurrahman/covid19-radiography-database"
DATASET_URL = "https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database"

# Supplementary: Tuberculosis dataset
KAGGLE_TB_DATASET = "tawsifurrahman/tuberculosis-tb-chest-xray-dataset"

# Alternative datasets:
# - "paultimothymooney/chest-xray-pneumonia" (2-class: Normal, Pneumonia)
# - "nih-chest-xrays/data" (14-class, large)

# Expected output structure
TRAIN_DIR = os.path.join(DATA_DIR, "raw", "train")
VAL_DIR = os.path.join(DATA_DIR, "raw", "val")


def is_dataset_ready():
    """
    Check if the dataset is already downloaded and properly organized.

    Returns:
        bool: True if train/val directories exist with images.
    """
    if not os.path.isdir(TRAIN_DIR):
        return False

    # Check that at least one class folder has images
    for cls in os.listdir(TRAIN_DIR):
        cls_dir = os.path.join(TRAIN_DIR, cls)
        if os.path.isdir(cls_dir):
            files = [f for f in os.listdir(cls_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(files) > 0:
                return True

    return False


def download_chest_xray(force=False):
    """
    Download and extract the Kaggle Chest X-Ray Pneumonia dataset.

    Tries multiple methods in order:
    1. Kaggle API (kaggle datasets download)
    2. Manual instructions (if API not available)

    Args:
        force (bool): Re-download even if data exists.

    Returns:
        bool: True if dataset is ready.
    """
    if is_dataset_ready() and not force:
        print("  ✓ Dataset already downloaded and organized")
        _print_stats()
        return True

    print("=" * 60)
    print("  📥 DOWNLOADING CHEST X-RAY DATASET")
    print("=" * 60)
    print(f"\n  Dataset: {KAGGLE_DATASET}")
    print(f"  Target:  {DATA_DIR}/raw/")
    print()

    # Try Kaggle API first
    if _try_kaggle_api():
        _organize_dataset()
        _print_stats()
        return True

    # Try opendatasets
    if _try_opendatasets():
        _organize_dataset()
        _print_stats()
        return True

    # Fallback: manual instructions
    _print_manual_instructions()
    return False


def _try_kaggle_api():
    """
    Download using the official Kaggle API.

    Requires:
        - pip install kaggle
        - ~/.kaggle/kaggle.json with valid credentials
        OR
        - KAGGLE_USERNAME and KAGGLE_KEY environment variables
    """
    try:
        import kaggle
    except ImportError:
        print("  Kaggle API not installed. Installing...")
        os.system(f"{sys.executable} -m pip install kaggle -q")
        try:
            import kaggle
        except ImportError:
            print("  ❌ Failed to install kaggle package")
            return False

    # Check credentials
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")

    if not os.path.exists(kaggle_json) and not has_env:
        print("  ❌ Kaggle credentials not found")
        print(f"     Expected: {kaggle_json}")
        print(f"     Or set: KAGGLE_USERNAME + KAGGLE_KEY env vars")
        return False

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        download_dir = os.path.join(DATA_DIR, "_download")
        os.makedirs(download_dir, exist_ok=True)

        print("  Downloading from Kaggle (this may take a few minutes)...")
        api.dataset_download_files(
            KAGGLE_DATASET,
            path=download_dir,
            unzip=True
        )
        print("  ✓ COVID/Normal/Pneumonia dataset downloaded!")

        # Also download Tuberculosis dataset
        print("  Downloading Tuberculosis dataset...")
        tb_dir = os.path.join(download_dir, "_tb")
        os.makedirs(tb_dir, exist_ok=True)
        try:
            api.dataset_download_files(
                KAGGLE_TB_DATASET,
                path=tb_dir,
                unzip=True
            )
            print("  ✓ Tuberculosis dataset downloaded!")
        except Exception as e:
            print(f"  ⚠️  TB dataset download failed: {e}")
            print("      You can add TB data manually to data/raw/train/Tuberculosis/")

        return True

    except Exception as e:
        print(f"  ❌ Kaggle API error: {e}")
        return False


def _try_opendatasets():
    """
    Download using opendatasets library (alternative to Kaggle API).
    """
    try:
        import opendatasets
    except ImportError:
        # Don't auto-install opendatasets as it has interactive prompts
        return False

    try:
        download_dir = os.path.join(DATA_DIR, "_download")
        os.makedirs(download_dir, exist_ok=True)

        opendatasets.download(DATASET_URL, data_dir=download_dir)
        print("  ✓ Download complete via opendatasets!")
        return True
    except Exception as e:
        print(f"  ❌ opendatasets error: {e}")
        return False


def _organize_dataset():
    """
    Organize downloaded files into the expected directory structure.

    Handles multiple dataset formats:
    1. COVID-19 Radiography Database: COVID/Normal/Lung_Opacity/Viral Pneumonia/images/
    2. Chest X-Ray Pneumonia: chest_xray/train/Normal|Pneumonia/

    Normalizes to:
        data/raw/train/COVID/
        data/raw/train/Normal/
        data/raw/train/Pneumonia/
        data/raw/train/Tuberculosis/
        data/raw/val/<same classes>/
    """
    print("\n  Organizing dataset structure...")

    download_dir = os.path.join(DATA_DIR, "_download")
    raw_dir = os.path.join(DATA_DIR, "raw")

    # Strategy 1: COVID-19 Radiography Database format
    # Has folders: COVID/images/, Normal/images/, Lung_Opacity/images/, Viral Pneumonia/images/
    covid_classes = {'COVID': 'COVID', 'Normal': 'Normal',
                     'Lung_Opacity': 'Pneumonia', 'Viral Pneumonia': 'Pneumonia',
                     'Tuberculosis': 'Tuberculosis'}

    found_covid_format = False
    for root, dirs, files in os.walk(download_dir):
        if 'COVID' in dirs and 'Normal' in dirs:
            found_covid_format = True
            source_dir = root
            break

    if found_covid_format:
        print("  Detected: COVID-19 Radiography Database format")
        _organize_covid_format(source_dir, covid_classes)

        # Also organize TB data if downloaded separately
        tb_dir = os.path.join(download_dir, "_tb")
        if os.path.isdir(tb_dir):
            _organize_tb_data(tb_dir)

        _cleanup(download_dir)
        return

    # Strategy 2: Standard train/val/test folder structure
    source_dir = None
    for root, dirs, files in os.walk(download_dir):
        if 'train' in dirs and ('test' in dirs or 'val' in dirs):
            source_dir = root
            break

    if source_dir is None:
        print("  ❌ Could not find extracted dataset structure")
        return

    # Copy train data
    src_train = os.path.join(source_dir, "train")
    if os.path.isdir(src_train):
        if os.path.exists(TRAIN_DIR):
            shutil.rmtree(TRAIN_DIR)
        shutil.copytree(src_train, TRAIN_DIR)
        print(f"  ✓ Training data → {TRAIN_DIR}")

    # Copy val data (Kaggle has both 'val' and 'test')
    src_val = os.path.join(source_dir, "val")
    src_test = os.path.join(source_dir, "test")

    val_source = src_val if os.path.isdir(src_val) else src_test
    if os.path.isdir(val_source):
        if os.path.exists(VAL_DIR):
            shutil.rmtree(VAL_DIR)
        shutil.copytree(val_source, VAL_DIR)
        print(f"  ✓ Validation data → {VAL_DIR}")

    # Clean up download folder
    try:
        shutil.rmtree(download_dir)
        print("  ✓ Cleaned up temporary download files")
    except Exception:
        pass

    print("  ✓ Dataset organized successfully!")


def _organize_covid_format(source_dir, class_mapping):
    """
    Organize COVID-19 Radiography Database into train/val splits.

    The dataset has all images in class folders (no train/val split),
    so we create an 80/20 split.

    Args:
        source_dir (str): Root of extracted dataset.
        class_mapping (dict): Maps folder names to our class names.
    """
    import random
    random.seed(42)

    for src_folder, target_class in class_mapping.items():
        src_path = os.path.join(source_dir, src_folder)

        # Some datasets have images in a subfolder called 'images'
        images_subfolder = os.path.join(src_path, "images")
        if os.path.isdir(images_subfolder):
            src_path = images_subfolder

        if not os.path.isdir(src_path):
            continue

        # Get all image files
        images = [f for f in os.listdir(src_path)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not images:
            continue

        random.shuffle(images)
        split_idx = int(len(images) * 0.8)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]

        # Copy to train/val
        for split_name, img_list in [("train", train_imgs), ("val", val_imgs)]:
            dest_dir = os.path.join(DATA_DIR, "raw", split_name, target_class)
            os.makedirs(dest_dir, exist_ok=True)
            for img_name in img_list:
                src_file = os.path.join(src_path, img_name)
                dst_file = os.path.join(dest_dir, img_name)
                if not os.path.exists(dst_file):
                    shutil.copy2(src_file, dst_file)

        print(f"    {src_folder} → {target_class}: "
              f"{len(train_imgs)} train + {len(val_imgs)} val")

    print("  ✓ Dataset organized (80/20 train/val split)")


def _organize_tb_data(tb_dir):
    """
    Organize Tuberculosis dataset into train/val structure.

    The TB dataset may have various structures — we find all images
    in TB-positive folders and organize them.
    """
    import random
    random.seed(42)

    # Find TB images (look for folders named Tuberculosis, TB, etc.)
    tb_images = []
    for root, dirs, files in os.walk(tb_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                tb_images.append(os.path.join(root, f))

    if not tb_images:
        return

    random.shuffle(tb_images)
    split_idx = int(len(tb_images) * 0.8)
    train_imgs = tb_images[:split_idx]
    val_imgs = tb_images[split_idx:]

    for split_name, img_list in [("train", train_imgs), ("val", val_imgs)]:
        dest_dir = os.path.join(DATA_DIR, "raw", split_name, "Tuberculosis")
        os.makedirs(dest_dir, exist_ok=True)
        for img_path in img_list:
            dst = os.path.join(dest_dir, os.path.basename(img_path))
            if not os.path.exists(dst):
                shutil.copy2(img_path, dst)

    print(f"    Tuberculosis: {len(train_imgs)} train + {len(val_imgs)} val")


def _cleanup(download_dir):
    """Remove temporary download directory."""
    try:
        shutil.rmtree(download_dir)
        print("  ✓ Cleaned up temporary download files")
    except Exception:
        pass


def _print_stats():
    """Print dataset statistics."""
    print(f"\n  Dataset location: {os.path.join(DATA_DIR, 'raw')}")
    for split_name, split_dir in [("Train", TRAIN_DIR), ("Val", VAL_DIR)]:
        if os.path.isdir(split_dir):
            total = 0
            for cls in sorted(os.listdir(split_dir)):
                cls_dir = os.path.join(split_dir, cls)
                if os.path.isdir(cls_dir):
                    count = len([f for f in os.listdir(cls_dir)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    total += count
                    print(f"    {split_name}/{cls}: {count} images")
            print(f"    {split_name} total: {total}")
    print()


def _print_manual_instructions():
    """Print manual download instructions."""
    print()
    print("  " + "=" * 56)
    print("  📋 MANUAL DOWNLOAD INSTRUCTIONS")
    print("  " + "=" * 56)
    print()
    print("  Option A: Kaggle API (recommended)")
    print("  ─────────────────────────────────")
    print("  1. pip install kaggle")
    print("  2. Go to https://www.kaggle.com/settings")
    print("  3. Click 'Create New Token' → downloads kaggle.json")
    print("  4. mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/")
    print("  5. chmod 600 ~/.kaggle/kaggle.json")
    print("  6. Run this script again")
    print()
    print("  Option B: Manual download")
    print("  ──────────────────────────")
    print(f"  1. Go to: {DATASET_URL}")
    print("  2. Click 'Download' (requires Kaggle account)")
    print("  3. Extract the zip file")
    print("  4. Copy contents to:")
    print(f"     {TRAIN_DIR}/Normal/")
    print(f"     {TRAIN_DIR}/Pneumonia/")
    print(f"     {VAL_DIR}/Normal/")
    print(f"     {VAL_DIR}/Pneumonia/")
    print()
    print("  Option C: Environment variables")
    print("  ────────────────────────────────")
    print("  export KAGGLE_USERNAME='your_username'")
    print("  export KAGGLE_KEY='your_api_key'")
    print("  Then run this script again.")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Chest X-Ray dataset from Kaggle")
    parser.add_argument("--force", action="store_true", help="Re-download even if exists")
    args = parser.parse_args()

    success = download_chest_xray(force=args.force)
    if success:
        print("  ✅ Dataset ready for training!")
        print("  Run: python training/train.py")
    else:
        print("  ❌ Dataset download failed. Follow the manual instructions above.")
        sys.exit(1)
