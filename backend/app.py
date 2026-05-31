"""
Medical AI X-Ray Analysis — FastAPI Application
=================================================

Main application entry point. Configures the FastAPI server,
loads the trained model, and mounts all API routes.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from inference import load_model
from config import API_HOST, API_PORT


# Create FastAPI application
app = FastAPI(
    title="Medical AI X-Ray Analysis",
    description="Deep learning API for chest X-ray diagnosis (Pneumonia, Normal)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS (allow frontend connections)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.on_event("startup")
async def startup():
    """Load model on server startup."""
    print("\n🏥 Medical AI X-Ray Analysis Server Starting...")
    load_model()
    print(f"🚀 Server ready at http://{API_HOST}:{API_PORT}")
    print(f"📖 API docs at http://{API_HOST}:{API_PORT}/docs\n")


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Medical AI X-Ray Analysis",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict — Upload X-ray for diagnosis",
            "gradcam": "POST /predict/gradcam — Prediction with heatmap",
            "health": "GET /health — Server & model status",
            "docs": "GET /docs — Interactive API documentation",
        }
    }
