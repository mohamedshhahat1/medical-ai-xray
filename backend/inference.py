"""
Inference Engine for Medical X-Ray Prediction
===============================================

Loads the trained model and runs inference on new X-ray images.
Returns prediction, confidence, and optional Grad-CAM visualization.
"""

import os
import torch
import torch.nn.functional as F

from config import DEVICE, CLASS_NAMES, MODEL_PATH, NUM_CLASSES, MODEL_ARCH
from model import create_model, load_trained_model
from utils.preprocess import preprocess_image, preprocess_pil_image
from utils.visualize import GradCAM


# Global model instance (loaded once at startup)
_model = None
_grad_cam = None


def load_model():
    """Load the trained model into memory (called once at startup)."""
    global _model, _grad_cam

    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  No trained model found at: {MODEL_PATH}")
        print(f"   Train first with: python training/train.py")
        return False

    _model = load_trained_model(MODEL_PATH, arch=MODEL_ARCH, num_classes=NUM_CLASSES)
    _model = _model.to(DEVICE)
    _model.eval()

    # Initialize Grad-CAM
    _grad_cam = GradCAM(_model, target_layer_name="layer4")

    print(f"✓ Model loaded: {MODEL_ARCH} ({NUM_CLASSES} classes)")
    print(f"  Device: {DEVICE}")
    return True


def predict_image(image_bytes):
    """
    Run inference on raw image bytes.

    Args:
        image_bytes (bytes): Raw image file content.

    Returns:
        dict: {
            "prediction": str,      # Class name
            "confidence": float,    # 0.0 - 1.0
            "probabilities": dict,  # All class probabilities
            "class_index": int      # Predicted class index
        }
    """
    if _model is None:
        return {"error": "Model not loaded. Train first with: python training/train.py"}

    # Preprocess
    tensor = preprocess_image(image_bytes).to(DEVICE)

    # Inference
    with torch.no_grad():
        output = _model(tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)

    predicted_class = CLASS_NAMES[predicted_idx.item()]
    conf = confidence.item()

    # Build probability dict
    prob_dict = {
        CLASS_NAMES[i]: round(probabilities[0][i].item(), 4)
        for i in range(len(CLASS_NAMES))
    }

    return {
        "prediction": predicted_class,
        "confidence": round(conf, 4),
        "probabilities": prob_dict,
        "class_index": predicted_idx.item(),
    }


def predict_with_gradcam(image_bytes):
    """
    Run inference with Grad-CAM visualization.

    Args:
        image_bytes (bytes): Raw image file content.

    Returns:
        tuple: (prediction_dict, gradcam_heatmap)
    """
    if _model is None or _grad_cam is None:
        return {"error": "Model not loaded"}, None

    from PIL import Image
    import io

    # Get prediction
    result = predict_image(image_bytes)

    # Generate Grad-CAM
    tensor = preprocess_image(image_bytes).to(DEVICE)
    tensor.requires_grad_(True)
    heatmap = _grad_cam.generate(tensor, class_idx=result["class_index"])

    # Create overlay
    original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    overlay = _grad_cam.overlay_on_image(original_image, heatmap)

    return result, overlay
