"""
Configuration for Medical AI X-Ray Analysis
=============================================

Centralized configuration for model, training, and deployment settings.
"""

import os
import torch

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Model architecture: 'resnet18', 'resnet50', 'densenet121'
MODEL_ARCH = "resnet18"

# Number of output classes
NUM_CLASSES = 4  # Normal, Pneumonia, Tuberculosis, COVID

# Class names (order must match model output / folder names)
CLASS_NAMES = ["COVID", "Normal", "Pneumonia", "Tuberculosis"]

# Class descriptions (for UI display)
CLASS_INFO = {
    "Normal": {
        "description": "No significant abnormality detected",
        "severity": "none",
        "color": "#66bb6a",
    },
    "Pneumonia": {
        "description": "Lung infection causing inflammation in air sacs",
        "severity": "moderate",
        "color": "#ffa726",
    },
    "Tuberculosis": {
        "description": "Bacterial infection (Mycobacterium tuberculosis)",
        "severity": "high",
        "color": "#ef5350",
    },
    "COVID": {
        "description": "COVID-19 related lung patterns (ground-glass opacities)",
        "severity": "high",
        "color": "#ab47bc",
    },
}

# Input image size (model expects this resolution)
IMAGE_SIZE = 224

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pth")

# =============================================================================
# TRAINING HYPERPARAMETERS
# =============================================================================

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

# Early stopping
PATIENCE = 5

# =============================================================================
# DEVICE
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# PREPROCESSING
# =============================================================================

# ImageNet normalization (for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# X-ray specific (grayscale, single channel replicated to 3)
XRAY_MEAN = [0.5]
XRAY_STD = [0.25]

# =============================================================================
# API CONFIGURATION
# =============================================================================

API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".dcm"}
