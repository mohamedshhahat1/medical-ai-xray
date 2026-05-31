"""
Medical AI X-Ray Analysis — FastAPI Application
=================================================

Main application entry point. Configures the FastAPI server,
loads the trained model, serves the web UI, and mounts all API routes.

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from inference import load_model
from config import API_HOST, API_PORT


# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_DIR = os.path.join(BASE_DIR, "frontend", "web")


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
    print(f"🌐 Web UI at http://{API_HOST}:{API_PORT}/")
    print(f"📖 API docs at http://{API_HOST}:{API_PORT}/docs\n")


@app.get("/")
async def serve_ui():
    """Serve the Web UI for X-ray upload and analysis."""
    index_path = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "name": "Medical AI X-Ray Analysis",
        "version": "1.0.0",
        "message": "Web UI not found. Use /docs for API documentation.",
    }
