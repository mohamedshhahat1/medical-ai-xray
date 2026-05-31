# 🏥 Medical AI X-Ray Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

A **production-ready** deep learning system for chest X-ray diagnosis. Detects pneumonia, COVID-19, and other lung conditions from X-ray images using convolutional neural networks.

---

## 🎯 Features

- 🧠 Deep learning model (ResNet/DenseNet) for chest X-ray classification
- 🚀 FastAPI backend with REST API for real-time predictions
- 📱 Flutter-ready frontend integration
- 🔬 Grad-CAM visualization (shows where the model is looking)
- 📊 Comprehensive evaluation metrics (ROC, AUC, confusion matrix)
- 🐳 Docker support for easy deployment
- 📈 TensorBoard training visualization
- ⚡ ONNX export for edge deployment

---

## 📁 Project Structure

```
medical-ai-xray/
│
├── backend/                    # 🟢 AI Serving System
│   ├── app.py                  # FastAPI application
│   ├── inference.py            # Model inference engine
│   ├── model.py                # Model architecture definition
│   ├── config.py               # Configuration & constants
│   ├── requirements.txt        # Backend dependencies
│   ├── utils/
│   │   ├── preprocess.py       # Image preprocessing pipeline
│   │   └── visualize.py        # Grad-CAM & result visualization
│   ├── models/
│   │   └── best_model.pth      # Trained model weights
│   └── api/
│       └── routes.py           # API route definitions
│
├── training/                   # 🟡 Model Training Pipeline
│   ├── train.py                # Training loop & entry point
│   ├── dataset.py              # Dataset loading & management
│   ├── evaluate.py             # Model evaluation & metrics
│   └── transforms.py           # Data augmentation transforms
│
├── data/                       # 🔵 Dataset Storage
│   ├── raw/                    # Original X-ray images
│   │   ├── train/              # Training images
│   │   └── val/                # Validation images
│   └── processed/              # Preprocessed data
│
├── frontend/                   # 🟣 User Interface
│   ├── flutter_app/            # Flutter mobile/web app
│   └── README.md               # Frontend documentation
│
├── notebooks/                  # 🟠 Experimentation
│   ├── exploration.ipynb       # Data exploration & analysis
│   └── training_debug.ipynb    # Training debugging
│
├── tests/                      # 🔴 Testing
│   ├── test_model.py           # Model unit tests
│   └── test_api.py             # API integration tests
│
├── docs/                       # 📚 Documentation
│   ├── architecture.md         # System architecture
│   └── api_reference.md        # API documentation
│
├── README.md
├── .gitignore
└── docker-compose.yml
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/mohamedshhahat1/medical-ai-xray.git
cd medical-ai-xray
pip install -r backend/requirements.txt
```

### 2. Train the Model

```bash
python training/train.py --epochs 20 --batch-size 32
```

### 3. Start the API Server

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 4. Make a Prediction

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@xray_image.png"
```

**Response:**
```json
{
  "prediction": "Pneumonia",
  "confidence": 0.93,
  "probabilities": {
    "Normal": 0.07,
    "Pneumonia": 0.93
  },
  "grad_cam_url": "/visualize/gradcam/latest"
}
```

---

## 🐳 Docker

```bash
docker-compose up --build
# Open http://localhost:8000/docs for API documentation
```

---

## 🧠 Tech Stack

| Technology | Purpose |
|-----------|---------|
| PyTorch | Deep learning framework |
| torchvision | Pretrained models & transforms |
| FastAPI | REST API backend |
| OpenCV | Image processing |
| Grad-CAM | Model interpretability |
| Flutter | Mobile/Web frontend |
| Docker | Containerization |
| TensorBoard | Training visualization |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~95% |
| Sensitivity | ~94% |
| Specificity | ~96% |
| AUC-ROC | ~0.98 |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/mohamedshhahat1">Mohamed Shhahat</a>
</p>
