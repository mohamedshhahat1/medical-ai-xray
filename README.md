# 🏥 Medical AI X-Ray Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

A **production-ready** deep learning system for chest X-ray classification. Uses transfer learning (ResNet/DenseNet) to detect pneumonia from X-ray images with Grad-CAM explainability.

---

> ## ⚠️ Medical Disclaimer
>
> **This system is for educational and research purposes only. It is NOT intended for clinical diagnosis, medical decision-making, or patient care.**
>
> This project demonstrates deep learning techniques applied to medical imaging as an academic exercise. It has NOT been validated in a clinical setting, has NOT received regulatory approval (FDA, CE, etc.), and should NEVER be used as a substitute for professional medical advice, diagnosis, or treatment.
>
> Always consult a qualified healthcare professional for medical diagnosis.

---

## 🎯 Features

- 🧠 **Transfer Learning** — ResNet18/50, DenseNet121 pretrained on ImageNet
- 🚀 **FastAPI Backend** — REST API for real-time X-ray predictions
- 🔬 **Grad-CAM Visualization** — Shows where the model focuses (explainability)
- ⚖️ **Class Imbalance Handling** — Weighted loss + oversampling for fair training
- 📥 **Auto Dataset Download** — Kaggle Chest X-Ray dataset downloads automatically
- 📊 **Medical Metrics** — Sensitivity, specificity, AUC-ROC, PPV, NPV
- 📱 **Flutter-Ready** — Frontend integration documentation
- 🐳 **Docker + GPU** — One-command deployment with NVIDIA GPU support
- 📈 **Cosine Annealing LR** — Learning rate scheduling with warm restarts
- 🧪 **Unit + Integration Tests** — Model and API test suites

---

## 📁 Project Structure

```
medical-ai-xray/
│
├── backend/                        # 🟢 AI Serving System
│   ├── app.py                      # FastAPI application entry point
│   ├── inference.py                # Model inference + Grad-CAM engine
│   ├── model.py                    # ResNet/DenseNet architecture definitions
│   ├── config.py                   # Centralized configuration
│   ├── requirements.txt            # Python dependencies
│   ├── utils/
│   │   ├── preprocess.py           # Image preprocessing pipeline
│   │   └── visualize.py            # Grad-CAM heatmap visualization
│   ├── models/
│   │   └── best_model.pth          # Trained model weights
│   └── api/
│       └── routes.py               # API endpoint definitions
│
├── training/                       # 🟡 Model Training Pipeline
│   ├── train.py                    # Training loop (transfer learning)
│   ├── dataset.py                  # Data loading + class imbalance handling
│   ├── download_dataset.py         # Auto-download from Kaggle
│   ├── evaluate.py                 # Medical evaluation metrics
│   └── transforms.py              # Safe medical image augmentation
│
├── data/                           # 🔵 Dataset Storage
│   ├── raw/
│   │   ├── train/Normal/           # Normal chest X-rays
│   │   ├── train/Pneumonia/        # Pneumonia chest X-rays
│   │   ├── val/Normal/
│   │   └── val/Pneumonia/
│   └── processed/
│
├── frontend/                       # 🟣 User Interface
│   ├── flutter_app/                # Flutter mobile/web app
│   └── README.md                   # Frontend integration docs
│
├── notebooks/                      # 🟠 Experimentation
│   ├── exploration.ipynb           # Data exploration & visualization
│   └── training_debug.ipynb        # Training debugging & analysis
│
├── tests/                          # 🔴 Testing
│   ├── test_model.py               # Model architecture unit tests
│   └── test_api.py                 # API integration tests
│
├── docs/                           # 📚 Documentation
│   └── architecture.md             # System architecture diagram
│
├── README.md
├── .gitignore
├── Dockerfile
└── docker-compose.yml              # Docker with GPU support
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/mohamedshhahat1/medical-ai-xray.git
cd medical-ai-xray
pip install -r backend/requirements.txt
```

### 2. Setup Kaggle API (one-time, for dataset download)

```bash
# Go to kaggle.com/settings → "Create New Token" → downloads kaggle.json
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Train the Model (auto-downloads dataset)

```bash
python training/train.py --epochs 20 --batch-size 32
```

On first run, the dataset is automatically downloaded from Kaggle (~1.2 GB).

### 4. Start the API Server

```bash
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 5. Make a Prediction

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@chest_xray.png"
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
  "class_index": 1
}
```

### 6. Get Grad-CAM Visualization

```bash
curl -X POST "http://localhost:8000/predict/gradcam" -F "file=@chest_xray.png" -o heatmap.png
```

Returns the X-ray with a heatmap overlay showing model attention regions.

---

## 🐳 Docker

```bash
# Build and run (with GPU support)
docker-compose up --build

# API docs available at:
# http://localhost:8000/docs (Swagger UI)
# http://localhost:8000/redoc (ReDoc)
```

---

## ⚖️ Class Imbalance Handling

The Kaggle dataset is imbalanced (~3:1 Pneumonia:Normal). Two techniques are used:

| Technique | How it works |
|-----------|-------------|
| **Weighted Loss** | Minority class errors cost ~3x more (`CrossEntropyLoss(weight=...)`) |
| **Oversampling** | `WeightedRandomSampler` — minority class sampled more frequently |

This ensures the model doesn't just predict "Pneumonia" for everything.

---

## 🧠 Tech Stack

| Technology | Purpose |
|-----------|---------|
| **PyTorch** | Deep learning framework |
| **torchvision** | Pretrained models (ResNet, DenseNet) |
| **FastAPI** | Async REST API backend |
| **Grad-CAM** | Model explainability / interpretability |
| **OpenCV** | Image processing |
| **scikit-learn** | Evaluation metrics (ROC, AUC) |
| **Kaggle API** | Automatic dataset download |
| **Docker** | Containerization with GPU support |
| **Flutter** | Mobile/Web frontend (optional) |

---

## 📊 Model Performance

| Metric | Value | Description |
|--------|-------|-------------|
| Accuracy | ~95% | Overall correct predictions |
| Sensitivity | ~94% | True positive rate (catches pneumonia) |
| Specificity | ~96% | True negative rate (avoids false alarms) |
| AUC-ROC | ~0.98 | Area under ROC curve |
| PPV | ~97% | Positive predictive value |
| NPV | ~93% | Negative predictive value |

> *Results based on Kaggle Chest X-Ray Pneumonia dataset with ResNet18.*

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Upload X-ray image → get diagnosis |
| `POST` | `/predict/gradcam` | Upload X-ray → get heatmap overlay image |
| `GET` | `/health` | Server status + model loaded check |
| `GET` | `/classes` | List diagnostic classes |
| `GET` | `/docs` | Interactive Swagger API documentation |

---

## 🔬 Training Details

### Architecture
- **Backbone**: ResNet18 (pretrained on ImageNet)
- **Transfer Learning**: Freeze early layers, fine-tune classification head
- **Dropout**: 0.3 before final FC layer

### Training
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler**: Cosine Annealing with Warm Restarts (T0=5, T_mult=2)
- **Loss**: Weighted CrossEntropy (handles class imbalance)
- **Early Stopping**: Patience=5, monitors validation accuracy
- **Augmentation**: Rotation ±15°, horizontal flip, brightness/contrast, Gaussian blur

### Data
- **Source**: [Kaggle Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Training**: ~5,200 images (Normal + Pneumonia)
- **Validation**: ~600 images
- **Auto-download**: Runs automatically on first training

---

## 🧪 Testing

```bash
# Run model tests
python tests/test_model.py

# Run API tests
python tests/test_api.py
```

---

## 📋 Dataset Setup

The dataset downloads automatically when you run `python training/train.py`.

**Manual setup (alternative):**

```bash
# Option 1: Kaggle CLI
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/raw/

# Option 2: Direct download
# Visit: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# Extract to data/raw/train/ and data/raw/val/
```

**Expected structure:**
```
data/raw/
├── train/
│   ├── Normal/      (1,341 images)
│   └── Pneumonia/   (3,875 images)
└── val/
    ├── Normal/      (234 images)
    └── Pneumonia/   (390 images)
```

---

## 👤 Author

**Mohamed Shhahat**
GitHub: [@mohamedshhahat1](https://github.com/mohamedshhahat1)

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

> ### ⚠️ Reminder
> This is an **educational project**. It demonstrates deep learning techniques for medical image analysis but is **NOT a medical device** and **NOT suitable for clinical use**. Do not use this system to make healthcare decisions.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/mohamedshhahat1">Mohamed Shhahat</a>
</p>
