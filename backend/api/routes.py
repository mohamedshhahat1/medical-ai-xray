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


@router.post("/dicom/metadata")
async def get_dicom_metadata(file: UploadFile = File(...)):
    """
    Extract metadata from a DICOM file.

    Returns patient info (anonymized), study details, and technical parameters.
    Useful for displaying scan info alongside AI predictions.
    """
    try:
        from utils.dicom_handler import PYDICOM_AVAILABLE, get_metadata
        if not PYDICOM_AVAILABLE:
            raise HTTPException(status_code=501,
                                detail="DICOM support requires pydicom. Install: pip install pydicom")

        import tempfile
        contents = await file.read()

        with tempfile.NamedTemporaryFile(suffix='.dcm', delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            metadata = get_metadata(tmp_path, anonymize=True)
        finally:
            os.remove(tmp_path)

        return {"metadata": metadata}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read DICOM: {str(e)}")


@router.post("/report")
async def generate_pdf_report(file: UploadFile = File(...),
                              patient_id: str = None, patient_name: str = None,
                              notes: str = None):
    """
    Generate a professional PDF report for an X-ray analysis.

    Returns a downloadable PDF with:
    - AI diagnosis and confidence
    - Classification probabilities for all diseases
    - Grad-CAM heatmap visualization
    - Highlighted region description
    - Medical disclaimer

    Query params:
        patient_id (str, optional): Patient identifier for the report.
        patient_name (str, optional): Patient name.
        notes (str, optional): Additional clinical notes.
    """
    from report_generator import generate_report
    from PIL import Image as PILImage

    try:
        contents = await file.read()

        # Run prediction
        result = predict_image(contents)
        if "error" in result:
            raise HTTPException(status_code=503, detail=result["error"])

        # Get Grad-CAM
        _, heatmap_overlay = predict_with_gradcam(contents)

        # Original image
        original_img = PILImage.open(io.BytesIO(contents)).convert("RGB")

        # Generate PDF
        pdf_bytes = generate_report(
            prediction=result,
            original_image=original_img,
            heatmap_image=heatmap_overlay,
            patient_id=patient_id,
            patient_name=patient_name,
            notes=notes,
        )

        # Return PDF
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=xray_report_{result['prediction'].lower()}.pdf"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation error: {str(e)}")
