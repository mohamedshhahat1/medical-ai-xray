"""
GCP Vertex AI Deployment Script
==================================

Deploys the model to Google Cloud Vertex AI for managed inference.
Provides auto-scaling, monitoring, and model versioning.

Prerequisites:
    pip install google-cloud-aiplatform
    gcloud auth login
    gcloud config set project YOUR_PROJECT

Usage:
    python deploy/gcp/vertex_deploy.py
    python deploy/gcp/vertex_deploy.py --machine n1-standard-4-t4
"""

import os
import sys
import argparse


def deploy_to_vertex(project_id, region="us-central1",
                     machine_type="n1-standard-4",
                     accelerator="NVIDIA_TESLA_T4"):
    """
    Deploy model to Vertex AI endpoint.

    Args:
        project_id (str): GCP project ID.
        region (str): GCP region.
        machine_type (str): Compute instance type.
        accelerator (str): GPU type.
    """
    try:
        from google.cloud import aiplatform
    except ImportError:
        print("Install: pip install google-cloud-aiplatform")
        return

    aiplatform.init(project=project_id, location=region)

    print("=" * 60)
    print("  🔵 DEPLOYING TO VERTEX AI")
    print("=" * 60)
    print(f"  Project: {project_id}")
    print(f"  Region: {region}")
    print(f"  Machine: {machine_type}")
    print(f"  GPU: {accelerator}")
    print()

    # Upload model
    print("  Uploading model...")
    model = aiplatform.Model.upload(
        display_name="medical-ai-xray",
        serving_container_image_uri=f"gcr.io/{project_id}/medical-ai-xray:latest",
        serving_container_ports=[8000],
        serving_container_health_route="/health",
        serving_container_predict_route="/predict",
    )
    print(f"  ✓ Model uploaded: {model.resource_name}")

    # Create endpoint
    print("  Creating endpoint...")
    endpoint = aiplatform.Endpoint.create(display_name="medical-ai-xray-endpoint")
    print(f"  ✓ Endpoint created: {endpoint.resource_name}")

    # Deploy model to endpoint
    print("  Deploying model to endpoint...")
    model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        accelerator_type=accelerator,
        accelerator_count=1,
        min_replica_count=1,
        max_replica_count=3,
    )

    print(f"\n  ✅ Deployed successfully!")
    print(f"  Endpoint: {endpoint.resource_name}")
    print(f"  Predict URL: https://{region}-aiplatform.googleapis.com/v1/{endpoint.resource_name}:predict")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy to GCP Vertex AI")
    parser.add_argument("--project", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region")
    parser.add_argument("--machine", default="n1-standard-4", help="Machine type")
    parser.add_argument("--gpu", default="NVIDIA_TESLA_T4", help="GPU type")
    args = parser.parse_args()

    deploy_to_vertex(args.project, args.region, args.machine, args.gpu)
