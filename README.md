# 🏥 Medical AI X-Ray Analysis

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Flutter](https://img.shields.io/badge/Flutter-3.x-02569B?style=for-the-badge&logo=flutter&logoColor=white)](https://flutter.dev)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](Dockerfile)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

A **production-ready** deep learning system for chest X-ray classification. Detects multiple lung conditions — **COVID-19, Pneumonia, Tuberculosis** — using transfer learning (ResNet/DenseNet) with Grad-CAM explainability, real-time web interface, Flutter mobile app, and cloud deployment support.

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

### Core AI
- 🧠 **Multi-Disease Classification** — COVID-19, Pneumonia, Normal (expandable to Tuberculosis)
- 🔬 **Transfer Learning** — ResNet18/50, DenseNet121 pretrained on ImageNet
- 🔥 **Grad-CAM Visualization** — Shows where the model focuses (explainability)
- ⚖️ **Class Imbalance Handling** — Weighted loss + oversampling
- 📈 **Cosine Annealing LR** — Learning rate scheduling with warm restarts
- 🛑 **Early Stopping** — Prevents overfitting automatically

### Deployment & API
- 🚀 **FastAPI Backend** — REST API for real-time X-ray predictions
- 🌐 **Web UI** — Drag & drop upload, instant diagnosis, heatmap, PDF report
- 📱 **Flutter Mobile App** — 4 screens (Upload, Result, Grad-CAM, PDF)
- 🐳 **Docker + GPU** — One-command deployment with NVIDIA support
- ☁️ **Cloud Deployment** — AWS (ECS/SageMaker), GCP (K8s/Vertex AI), RunPod
- 🔄 **CI/CD** — GitHub Actions auto-deploy pipeline

### Medical Features
- 📄 **AI Report Generator** — Professional PDF with findings, Grad-CAM, disclaimer
- 🏥 **DICOM Support** — Native hospital image format (real clinical data)
- 🏥 **Multi-Hospital Validation** — Cross-dataset generalization testing (6 hospitals)
- 📥 **Auto Dataset Download** — Kaggle datasets download automatically

### Data & Training
- 📊 **Medical Metrics** — Sensitivity, specificity, AUC-ROC, PPV, NPV per class
- 🔢 **Multi-Hospital Validation** — Test on 6 different hospital datasets
- 🧪 **Safe Augmentation** — Medical-appropriate transforms (no harmful flips)
- 📥 **Kaggle Auto-Download** — COVID-19 Radiography + TB datasets

---

## 📁 Project Structure

```
medical-ai-xray/
│
├── backend/                        # 🟢 AI Serving System
│   ├── app.py                      # FastAPI application
│   ├── inference.py                # Model inference + Grad-CAM
│   ├── model.py                    # ResNet/DenseNet architectures
│   ├── config.py                   # Centralized configuration
│   ├── report_generator.py         # PDF report generation
│   ├── requirements.txt            # Python dependencies
│   ├── utils/
│   │   ├── preprocess.py           # Image preprocessing (PNG/JPG/DICOM)
│   │   ├── dicom_handler.py        # DICOM file loading & metadata
│   │   └── visualize.py            # Grad-CAM heatmap visualization
│   ├── models/
│   │   └── best_model.pth          # Trained model weights
│   └── api/
│       └── routes.py               # API endpoint definitions
│
├── training/                       # 🟡 Model Training Pipeline
│   ├── train.py                    # Training loop (transfer learning)
│   ├── dataset.py                  # Data loading + class imbalance
│   ├── download_dataset.py         # Auto-download from Kaggle
│   ├── evaluate.py                 # Medical evaluation metrics
│   ├── validate_multi_hospital.py  # Cross-hospital generalization
│   └── transforms.py              # Safe medical augmentation
│
├── data/                           # 🔵 Dataset Storage
│   ├── raw/train/                  # Training images (per class)
│   ├── raw/val/                    # Validation images
│   └── external/                   # Multi-hospital validation data
│
├── frontend/                       # 🟣 User Interfaces
│   ├── web/index.html              # Web UI (drag & drop, real-time)
│   ├── flutter_app/                # Flutter mobile app (4 screens)
│   └── README.md                   # Frontend docs
│
├── notebooks/                      # 🟠 Experimentation
│   ├── exploration.ipynb           # Data exploration
│   └── training_debug.ipynb        # Training debugging
│
├── tests/                          # 🔴 Testing
│   ├── test_model.py               # Model unit tests
│   └── test_api.py                 # API integration tests
│
├── deploy/                         # ☁️ Cloud Deployment
│   ├── README.md                   # Full deployment guide
│   ├── aws/                        # AWS ECS + SageMaker
│   ├── gcp/                        # GCP Kubernetes + Vertex AI
│   └── runpod/                     # RunPod serverless GPU
│
├── docs/                           # 📚 Documentation
│   └── architecture.md             # System architecture
│
├── .github/workflows/deploy.yml    # 🔄 CI/CD pipeline
├── Dockerfile
├── docker-compose.yml
├── README.md
└── .gitignore
```

---

## 🚀 Setup Instructions

### Option 1: Windows (Local)

#### Prerequisites
- Python 3.8+ → [Download](https://www.python.org/downloads/)
- Git → [Download](https://git-scm.com/download/win)
- (Optional) NVIDIA GPU + CUDA → [Download](https://developer.nvidia.com/cuda-downloads)

#### Step-by-step

```powershell
# 1. Clone the repository
git clone https://github.com/mohamedshhahat1/medical-ai-xray.git
cd medical-ai-xray

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install PyTorch (CPU version — works on any PC)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3b. OR install PyTorch with GPU (if you have NVIDIA GPU + CUDA)
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install other dependencies
pip install -r backend\requirements.txt

# 5. Setup Kaggle API key (for dataset download)
#    Go to https://www.kaggle.com/settings → Create New Token
#    Move downloaded kaggle.json to:
mkdir %USERPROFILE%\.kaggle
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\

# 6. Train the model (auto-downloads dataset)
python training\train.py --epochs 15

# 7. Start the server
cd backend
uvicorn app:app --host 0.0.0.0 --port 8000

# 8. Open browser → http://localhost:8000
```

#### Quick test (PowerShell):
```powershell
# Upload an X-ray for prediction
Invoke-WebRequest -Method POST -Uri "http://localhost:8000/predict" `
  -Form @{file = Get-Item "path\to\xray.png"} | Select-Object -Expand Content
```

---

### Option 2: macOS / Linux

```bash
# 1. Clone
git clone https://github.com/mohamedshhahat1/medical-ai-xray.git
cd medical-ai-xray

# 2. Virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Kaggle API key
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 5. Train (auto-downloads dataset from Kaggle)
python training/train.py --epochs 15

# 6. Start server
cd backend && uvicorn app:app --host 0.0.0.0 --port 8000

# 7. Open http://localhost:8000
```

---

### Option 3: Docker (Any OS — Recommended)

#### CPU Version:
```bash
# One command — downloads, trains, serves
docker-compose up --build

# Wait for training (~15 min first run), then open:
# http://localhost:8000
```

#### GPU Version (NVIDIA):
```bash
# 1. Install NVIDIA Container Toolkit
#    Windows: Install Docker Desktop + WSL2 + NVIDIA drivers
#    Linux: sudo apt install nvidia-container-toolkit

# 2. Run with GPU support (already configured in docker-compose.yml)
docker-compose up --build

# Training: ~2 min with GPU vs ~15 min on CPU
```

#### Manual Docker commands:
```bash
# Build
docker build -t medical-ai-xray .

# Run (CPU)
docker run -p 8000:8000 -v ./data:/app/data medical-ai-xray

# Run (GPU)
docker run --gpus all -p 8000:8000 -v ./data:/app/data medical-ai-xray
```

---

### Option 4: Google Colab (Free GPU)

```python
# In a Colab notebook:
!git clone https://github.com/mohamedshhahat1/medical-ai-xray.git
%cd medical-ai-xray
!pip install -r backend/requirements.txt

# Setup Kaggle
import os
os.environ['KAGGLE_USERNAME'] = 'your_username'
os.environ['KAGGLE_KEY'] = 'your_key'

# Train with free GPU
!python training/train.py --epochs 20 --batch-size 64

# Test
!python -c "
from backend.inference import load_model, predict_image
load_model()
# ... test prediction
"
```

---

## 📂 Dataset Setup & Training

### Automatic (Recommended)

If you have a Kaggle API key set up, just run:
```bash
python training/train.py --epochs 15
```
The dataset downloads and organizes automatically.

### Manual Download

1. Download from: https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
2. Extract the zip file
3. Organize into this structure:

```
data/raw/
│
├── train/                    ← 80% of images (model learns from these)
│   ├── COVID/                ← COVID-19 X-ray images
│   │   ├── COVID-1.png
│   │   ├── COVID-2.png
│   │   └── ... (2,800+ images)
│   ├── Normal/               ← Healthy chest X-rays
│   │   ├── Normal-1.png
│   │   └── ... (8,000+ images)
│   └── Pneumonia/            ← Pneumonia X-rays (bacterial + viral)
│       ├── Pneumonia-1.png
│       └── ... (4,600+ images)
│
└── val/                      ← 20% of images (model is tested on these)
    ├── COVID/                ← COVID validation images
    │   └── ... (700+ images)
    ├── Normal/               ← Normal validation images
    │   └── ... (2,000+ images)
    └── Pneumonia/            ← Pneumonia validation images
        └── ... (1,100+ images)
```

### Why 80/20 Split?

| Folder | Purpose | % | Why |
|--------|---------|:-:|-----|
| `train/` | Model learns patterns from these images | 80% | More data = better learning |
| `val/` | Tests accuracy on images the model has **NEVER seen** | 20% | Proves the model generalizes, doesn't just memorize |

> **Critical Rule:** Train and val images must be DIFFERENT images — never the same image in both folders. This prevents **data leakage** (the model memorizing answers instead of learning patterns).

### How to split manually

If your Kaggle download is all in one folder (no train/val split):

```bash
# Automatic: the download script does this for you (80/20 random split)
python training/download_dataset.py

# Or manually: move ~20% of each class folder to val/
# Example: if COVID/ has 3,616 images:
#   Move first 2,893 → train/COVID/  (80%)
#   Move last 723   → val/COVID/     (20%)
```

### Train the Model

```bash
# CPU training (~15-20 minutes)
python training/train.py --epochs 15 --batch-size 32

# GPU training (~2 minutes) — if you have NVIDIA GPU
python training/train.py --epochs 20 --batch-size 64 --arch resnet18

# Use DenseNet121 (better accuracy, slower)
python training/train.py --epochs 20 --arch densenet121
```

### What happens during training:

```
Epoch  1/15 | Loss: 0.85 | Train: 72.3% | Val: 68.5%   ← Learning starts
Epoch  5/15 | Loss: 0.32 | Train: 89.1% | Val: 85.2%   ← Getting better
Epoch 10/15 | Loss: 0.12 | Train: 96.4% | Val: 92.8%   ← Nearly converged
Epoch 15/15 | Loss: 0.08 | Train: 98.1% | Val: 93.5%   ← Done!
  ★ Best model saved: backend/models/best_model.pth
```

---

## 🎯 Usage

### Web UI (Recommended)

1. Start server: `cd backend && uvicorn app:app --port 8000`
2. Open: **http://localhost:8000**
3. Drag & drop an X-ray image
4. Get instant diagnosis + Grad-CAM heatmap
5. Download PDF report

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Upload X-ray → diagnosis + confidence |
| `POST` | `/predict/gradcam` | Upload X-ray → heatmap overlay image |
| `POST` | `/report` | Upload X-ray → downloadable PDF report |
| `POST` | `/dicom/metadata` | Upload .dcm → extract study metadata |
| `GET` | `/health` | Server status + model check |
| `GET` | `/classes` | List diagnostic classes |
| `GET` | `/docs` | Swagger API documentation |

### CLI Examples

```bash
# Predict
curl -X POST "http://localhost:8000/predict" -F "file=@xray.png"

# Get Grad-CAM heatmap
curl -X POST "http://localhost:8000/predict/gradcam" -F "file=@xray.png" -o heatmap.png

# Download PDF report
curl -X POST "http://localhost:8000/report" -F "file=@xray.png" -o report.pdf

# Upload DICOM file (hospital format)
curl -X POST "http://localhost:8000/predict" -F "file=@scan.dcm"
```

### Flutter Mobile App

```bash
cd frontend/flutter_app
flutter pub get
flutter run
```

4 screens: Upload (camera/gallery) → Result → Grad-CAM → PDF Report

---

## 🧠 Model Architecture

| Component | Details |
|-----------|---------|
| **Backbone** | ResNet18 (pretrained ImageNet) |
| **Transfer Learning** | Fine-tune classification head |
| **Classes** | COVID, Normal, Pneumonia |
| **Input** | 224×224 RGB (auto-resized) |
| **Dropout** | 0.3 before final FC |
| **Loss** | Weighted CrossEntropy (handles imbalance) |
| **Optimizer** | Adam (lr=0.001, weight_decay=1e-4) |
| **Scheduler** | Cosine Annealing (T_max=15) |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~93% |
| COVID Sensitivity | ~95% |
| Pneumonia Sensitivity | ~94% |
| Normal Specificity | ~91% |

> Trained on COVID-19 Radiography Database (Kaggle). Results improve with more epochs and GPU training.

---

## 📄 AI Report (PDF Output)

```
┌─────────────────────────────────────────────┐
│  🏥 Medical AI — X-Ray Analysis Report      │
│  Date: May 31, 2026                        │
│                                             │
│  ■ AI Diagnosis: PNEUMONIA                  │
│    Confidence: 93.2%                        │
│    Severity: MODERATE                       │
│                                             │
│  ■ Findings                                 │
│    • Lower right lung — consolidation       │
│    • Grad-CAM highlights infection area     │
│                                             │
│  ■ Probabilities                            │
│    Pneumonia  93.2%  ████████████████       │
│    Normal      4.1%  █                      │
│    COVID       2.7%  █                      │
│                                             │
│  [Original X-Ray]    [Grad-CAM Heatmap]     │
│                                             │
│  ⚠️ EDUCATIONAL PURPOSES ONLY               │
└─────────────────────────────────────────────┘
```

---

## ☁️ Cloud Deployment

| Platform | GPU | Cost | Setup |
|----------|:---:|------|-------|
| **RunPod** | A100/H100 | $0.39/hr | Easiest GPU |
| **AWS ECS** | T4/A10G | $0.50/hr | Production |
| **GCP Cloud Run** | T4/A100 | $0.35/hr | Auto-scale to zero |
| **Railway** | CPU only | $7/mo | Demos |

See [`deploy/README.md`](deploy/README.md) for full deployment guide.

---

## 🧪 Testing

### Unit & Integration Tests
```bash
python tests/test_model.py    # Model architecture tests
python tests/test_api.py      # API endpoint tests
```

### 📊 Full Evaluation (All Validation Images)

Tests ALL images in `data/raw/val/` and gives medical metrics:

```bash
python training/evaluate.py
```

**Output:**
```
  Overall Accuracy: 93.5%

  PER-CLASS MEDICAL METRICS (One-vs-Rest)
  Class          Sensitivity  Specificity     PPV      NPV
  COVID               0.9500       0.9700  0.9400   0.9800
  Normal              0.9100       0.9300  0.9000   0.9500
  Pneumonia           0.9400       0.9600  0.9500   0.9400

  Macro AUC-ROC (one-vs-rest): 0.9650
```

### 📁 Batch Test a Folder (Predict All Images)

Test every image in a folder and see individual results:

```bash
python -c "
import os, sys
sys.path.insert(0, 'backend')
from inference import load_model, predict_image

load_model()

folder = 'path/to/your/xray/folder'  # ← Change this
correct = 0
total = 0
for img in os.listdir(folder):
    if not img.endswith(('.png', '.jpg', '.jpeg')): continue
    with open(os.path.join(folder, img), 'rb') as f:
        result = predict_image(f.read())
    total += 1
    print(f'  {img} → {result[\"prediction\"]} ({result[\"confidence\"]*100:.1f}%)')

print(f'\n  Total tested: {total} images')
"
```

### 📡 Batch Test via API (cURL)

Test a folder of images using the running server:

```bash
# Start server first: cd backend && uvicorn app:app --port 8000

# Test all images in a folder
for img in data/raw/val/COVID/*.png; do
  echo -n "$(basename $img) → "
  curl -s -X POST "http://localhost:8000/predict" -F "file=@$img" | \
    python3 -c "import sys,json;d=json.load(sys.stdin);print(f\"{d['prediction']} ({d['confidence']*100:.1f}%)\")"
done
```

**Windows PowerShell:**
```powershell
# Test all images in a folder
Get-ChildItem "data\raw\val\COVID\*.png" | ForEach-Object {
    $result = Invoke-WebRequest -Method POST -Uri "http://localhost:8000/predict" `
        -Form @{file = $_} | Select-Object -Expand Content | ConvertFrom-Json
    Write-Host "$($_.Name) → $($result.prediction) ($([math]::Round($result.confidence*100,1))%)"
}
```

### 🏥 Multi-Hospital Validation

Test model on data from different hospitals to prove it generalizes:

```bash
# List all registered hospital datasets
python training/validate_multi_hospital.py --list-datasets

# Validate on your own hospital data
python training/validate_multi_hospital.py --custom-dir path/to/hospital_images

# Generate validation report
python training/validate_multi_hospital.py --report
```

---

## 🏥 Multi-Hospital Validation

Validates model generalization across hospitals:

```bash
python training/validate_multi_hospital.py
```

Registered datasets from 6 institutions across 4 countries (USA, China, Bangladesh, Qatar).

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **PyTorch** | Deep learning framework |
| **torchvision** | Pretrained models (ResNet, DenseNet) |
| **FastAPI** | Async REST API |
| **Grad-CAM** | Model explainability |
| **pydicom** | DICOM medical image format |
| **ReportLab** | PDF report generation |
| **OpenCV** | Image processing |
| **scikit-learn** | Evaluation metrics |
| **Kaggle API** | Auto dataset download |
| **Docker** | Containerization + GPU |
| **Flutter** | Mobile/Web app |
| **GitHub Actions** | CI/CD pipeline |

---

## 🗺️ Roadmap

- [x] Multi-disease classification (COVID, Pneumonia, Normal)
- [x] Grad-CAM visualization
- [x] FastAPI REST API
- [x] Web UI (drag & drop, real-time)
- [x] PDF report generation
- [x] DICOM support
- [x] Docker + GPU deployment
- [x] Flutter mobile app
- [x] Multi-hospital validation
- [x] Cloud deployment (AWS, GCP, RunPod)
- [x] Class imbalance handling
- [x] Auto dataset download
- [ ] Add Tuberculosis class (separate dataset)
- [ ] Model ensemble (multiple architectures voting)
- [ ] ONNX export for edge deployment
- [ ] Patient history tracking
- [ ] HIPAA-compliant data handling

---

## 👤 Author

**Mohamed Shhahat**
GitHub: [@mohamedshhahat1](https://github.com/mohamedshhahat1)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

> ### ⚠️ Reminder
> This is an **educational project**. It demonstrates deep learning for medical imaging but is **NOT a medical device** and **NOT suitable for clinical use**. Do not use this system to make healthcare decisions.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/mohamedshhahat1">Mohamed Shhahat</a>
</p>
