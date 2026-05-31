# 🏗️ System Architecture

## Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Frontend  │────▶│   FastAPI    │────▶│   PyTorch   │
│  (Flutter)  │◀────│   Backend    │◀────│    Model    │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    │  Grad-CAM   │
                    │ Visualization│
                    └─────────────┘
```

## Processing Pipeline

```
X-Ray Image Upload
       │
       ▼
┌─────────────────┐
│  Preprocessing  │  Resize 224×224, Normalize
└─────────────────┘
       │
       ▼
┌─────────────────┐
│  ResNet/Dense   │  Feature extraction
│  Net Backbone   │  (pretrained ImageNet)
└─────────────────┘
       │
       ▼
┌─────────────────┐
│  Classification │  Normal vs Pneumonia
│  Head           │  (with confidence)
└─────────────────┘
       │
       ▼
┌─────────────────┐
│   Grad-CAM      │  Explainability heatmap
└─────────────────┘
       │
       ▼
  JSON Response + Heatmap Image
```

## Key Design Decisions

1. **Transfer Learning**: ImageNet pretrained → fine-tune on X-rays
2. **Grad-CAM**: Required for medical AI (explainability)
3. **FastAPI**: Async, fast, auto-docs (Swagger)
4. **Separation**: Training ≠ Serving (different requirements)
