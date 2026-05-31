# ☁️ Cloud Deployment Guide

## Quick Reference

| Platform | GPU | Cost | Best For |
|----------|-----|------|----------|
| **RunPod** | ✅ A100/H100 | $0.39/hr | Fast GPU inference, easy setup |
| **AWS** | ✅ T4/A10G | $0.50+/hr | Production, autoscaling |
| **GCP** | ✅ T4/A100 | $0.35+/hr | ML-optimized, Vertex AI |
| **Railway/Render** | ❌ CPU | $7/mo | Demo, low-traffic |

---

## 1. 🚀 RunPod (Recommended for GPU)

Fastest way to deploy with GPU inference.

### Option A: Serverless (Pay per request)

```bash
# 1. Create RunPod account: https://runpod.io
# 2. Build & push Docker image
docker build -t your-dockerhub/medical-ai-xray .
docker push your-dockerhub/medical-ai-xray

# 3. Create Serverless Endpoint on RunPod dashboard
#    - Template: your-dockerhub/medical-ai-xray
#    - GPU: RTX 3090 or A100
#    - Container port: 8000
```

### Option B: Pod (Always running)

```bash
# Deploy via RunPod CLI
pip install runpodctl
runpodctl create pod \
  --name medical-ai-xray \
  --image your-dockerhub/medical-ai-xray \
  --gpu-type "NVIDIA RTX A5000" \
  --ports "8000/http"
```

---

## 2. 🔶 AWS (Production)

### Option A: ECS with Fargate (Serverless containers)

```bash
# 1. Push image to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag medical-ai-xray:latest <account>.dkr.ecr.us-east-1.amazonaws.com/medical-ai-xray:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/medical-ai-xray:latest

# 2. Deploy with the provided task definition
aws ecs create-service --cluster medical-ai --task-definition medical-ai-xray --desired-count 1
```

### Option B: EC2 with GPU

```bash
# Launch g4dn.xlarge (NVIDIA T4, ~$0.53/hr)
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type g4dn.xlarge \
  --key-name your-key

# SSH in and run
ssh -i your-key.pem ec2-user@<ip>
git clone https://github.com/mohamedshhahat1/medical-ai-xray.git
cd medical-ai-xray
docker-compose up --build -d
```

### Option C: SageMaker Endpoint

See `deploy/aws/sagemaker_deploy.py` for SageMaker-specific deployment.

---

## 3. 🔵 GCP (Google Cloud)

### Option A: Cloud Run (Serverless, auto-scales to zero)

```bash
# 1. Build and push to GCR
gcloud builds submit --tag gcr.io/YOUR_PROJECT/medical-ai-xray

# 2. Deploy to Cloud Run
gcloud run deploy medical-ai-xray \
  --image gcr.io/YOUR_PROJECT/medical-ai-xray \
  --port 8000 \
  --memory 4Gi \
  --cpu 2 \
  --allow-unauthenticated
```

### Option B: GKE with GPU (Kubernetes)

```bash
# Create GPU cluster
gcloud container clusters create medical-ai \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --machine-type n1-standard-4

# Deploy
kubectl apply -f deploy/gcp/kubernetes.yaml
```

### Option C: Vertex AI (Managed ML)

See `deploy/gcp/vertex_deploy.py` for Vertex AI endpoint deployment.

---

## 4. 🟣 Railway / Render (Budget CPU)

For demos and low-traffic (no GPU):

```bash
# Railway (one-click)
# 1. Connect GitHub repo at https://railway.app
# 2. Set start command: cd backend && uvicorn app:app --host 0.0.0.0 --port $PORT
# Done!

# Render
# 1. Create Web Service at https://render.com
# 2. Docker runtime, port 8000
# 3. Auto-deploys from GitHub
```

---

## Environment Variables (All Platforms)

```bash
PYTHONUNBUFFERED=1
MODEL_ARCH=resnet18       # or resnet50, densenet121
NUM_CLASSES=4
PORT=8000
```

---

## Production Checklist

- [ ] Enable HTTPS (TLS certificate)
- [ ] Set up health check monitoring
- [ ] Configure auto-scaling rules
- [ ] Add rate limiting
- [ ] Enable logging (CloudWatch / Cloud Logging)
- [ ] Set up CI/CD (GitHub Actions → auto-deploy)
- [ ] Load test before launch
- [ ] Add authentication (API keys or OAuth)
