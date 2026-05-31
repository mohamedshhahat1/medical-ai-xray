"""
API Route Definitions
======================

Defines all REST API endpoints for the medical AI system.
"""

import io
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference import predict_image, predict_with_gradcam
from config import CLASS_NAMES, ALLOWED_EXTENSIONS


router = APIRouter()


# Response models
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict
    class_index: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list


# Routes
@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict diagnosis from an uploaded X-ray image.

    Accepts: PNG, JPG, JPEG, BMP
    Returns: prediction, confidence, and per-class probabilities.
    """
    # Validate file extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}"
        )

    try:
        contents = await file.read()
        result = predict_image(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

    if "error" in result:
        raise HTTPException(status_code=503, detail=result["error"])

    return result


@router.post("/predict/gradcam")
async def predict_gradcam(file: UploadFile = File(...)):
    """
    Predict with Grad-CAM visualization overlay.

    Returns the X-ray image with a heatmap showing where the model
    is focusing (important for medical professionals).
    """
    try:
        contents = await file.read()
        result, overlay_image = predict_with_gradcam(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    if "error" in result or overlay_image is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Return overlay image as PNG
    img_buffer = io.BytesIO()
    overlay_image.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    return StreamingResponse(
        img_buffer,
        media_type="image/png",
        headers={"X-Prediction": result["prediction"],
                 "X-Confidence": str(result["confidence"])}
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with model status."""
    from inference import _model
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "classes": CLASS_NAMES,
    }


@router.get("/classes")
async def get_classes():
    """Return the list of diagnostic classes."""
    return {"classes": CLASS_NAMES, "count": len(CLASS_NAMES)}
