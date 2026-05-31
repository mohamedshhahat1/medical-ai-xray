"""
RunPod Serverless Handler
===========================

Handles inference requests on RunPod serverless GPU infrastructure.
Receives base64 images, runs prediction, returns results.

RunPod serverless is pay-per-request GPU inference — you only pay
when predictions are running (no idle costs).
"""

import os
import sys
import base64

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

import runpod
from inference import load_model, predict_image


# Load model once at cold start
print("Loading model...")
load_model()
print("✓ Model ready for inference")


def handler(event):
    """
    RunPod serverless handler function.

    Input (event["input"]):
        {
            "image": "<base64_encoded_image>",
            "include_gradcam": true/false
        }

    Output:
        {
            "prediction": "Pneumonia",
            "confidence": 0.93,
            "probabilities": {...}
        }
    """
    try:
        input_data = event.get("input", {})

        # Get image from base64
        image_b64 = input_data.get("image")
        if not image_b64:
            return {"error": "No image provided. Send base64 in 'image' field."}

        # Decode base64
        image_bytes = base64.b64decode(image_b64)

        # Run prediction
        result = predict_image(image_bytes)

        return result

    except Exception as e:
        return {"error": str(e)}


# Start RunPod serverless worker
runpod.serverless.start({"handler": handler})
