#!/bin/bash
# RunPod Serverless Handler Setup
# This script prepares the container for RunPod serverless inference.
#
# Usage: Set as the Docker CMD for RunPod serverless template
#
# RunPod serverless expects:
#   - A handler function that accepts requests
#   - Response returned as JSON

echo "🏥 Medical AI X-Ray — RunPod Serverless Starting..."

# Install runpod SDK if not present
pip install runpod --quiet 2>/dev/null

# Start the handler
python /app/deploy/runpod/handler.py
