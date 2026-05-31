"""
Image Preprocessing Pipeline for X-Ray Analysis
=================================================

Handles loading, transforming, and normalizing X-ray images
for model inference. Supports PNG, JPG, and DICOM formats.
"""

import io
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def get_inference_transform():
    """
    Get the standard preprocessing transform for inference.

    Applies:
    1. Resize to IMAGE_SIZE x IMAGE_SIZE
    2. Convert to tensor [0, 1]
    3. Normalize with ImageNet statistics (for pretrained models)

    Returns:
        torchvision.transforms.Compose: Transform pipeline.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_image(image_bytes):
    """
    Preprocess raw image bytes for model inference.

    Args:
        image_bytes (bytes): Raw image file bytes.

    Returns:
        torch.Tensor: Preprocessed tensor of shape (1, 3, 224, 224).
    """
    image = Image.open(io.BytesIO(image_bytes))

    # Convert grayscale X-ray to 3-channel (required by pretrained models)
    if image.mode == "L":
        image = image.convert("RGB")
    elif image.mode == "RGBA":
        image = image.convert("RGB")
    elif image.mode != "RGB":
        image = image.convert("RGB")

    # Apply transforms
    transform = get_inference_transform()
    tensor = transform(image)

    # Add batch dimension: (3, 224, 224) → (1, 3, 224, 224)
    tensor = tensor.unsqueeze(0)

    return tensor


def preprocess_pil_image(image):
    """
    Preprocess a PIL Image for model inference.

    Args:
        image (PIL.Image): Input image.

    Returns:
        torch.Tensor: Preprocessed tensor of shape (1, 3, 224, 224).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    transform = get_inference_transform()
    tensor = transform(image).unsqueeze(0)
    return tensor
