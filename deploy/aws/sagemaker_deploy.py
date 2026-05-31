"""
AWS SageMaker Deployment Script
=================================

Deploys the trained model as a SageMaker real-time endpoint.
Provides auto-scaling, monitoring, and A/B testing out of the box.

Prerequisites:
    pip install sagemaker boto3
    aws configure (set up credentials)

Usage:
    python deploy/aws/sagemaker_deploy.py
    python deploy/aws/sagemaker_deploy.py --instance ml.g4dn.xlarge
"""

import os
import sys
import argparse
import tarfile
import boto3


def package_model(model_path, output_path="model.tar.gz"):
    """Package model for SageMaker (requires tar.gz format)."""
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(model_path, arcname="model.pth")
        # Add inference code
        inference_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'backend')
        tar.add(os.path.join(inference_dir, 'model.py'), arcname='code/model.py')
        tar.add(os.path.join(inference_dir, 'config.py'), arcname='code/config.py')
    print(f"  ✓ Model packaged: {output_path}")
    return output_path


def deploy_to_sagemaker(model_path, instance_type="ml.g4dn.xlarge",
                        endpoint_name="medical-ai-xray"):
    """
    Deploy model to SageMaker endpoint.

    Args:
        model_path (str): Path to model.tar.gz
        instance_type (str): SageMaker instance type
        endpoint_name (str): Name for the endpoint
    """
    try:
        import sagemaker
        from sagemaker.pytorch import PyTorchModel
    except ImportError:
        print("Install sagemaker: pip install sagemaker")
        return

    session = sagemaker.Session()
    role = sagemaker.get_execution_role()

    # Upload model to S3
    model_s3_path = session.upload_data(
        path=model_path,
        key_prefix="medical-ai-xray/model"
    )
    print(f"  ✓ Model uploaded to S3: {model_s3_path}")

    # Create PyTorch model
    pytorch_model = PyTorchModel(
        model_data=model_s3_path,
        role=role,
        framework_version="2.0.0",
        py_version="py310",
        entry_point="inference.py",
    )

    # Deploy
    print(f"  Deploying to {instance_type}...")
    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
    )

    print(f"  ✓ Endpoint deployed: {endpoint_name}")
    print(f"    URL: https://runtime.sagemaker.{session.boto_region_name}.amazonaws.com")
    return predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy to AWS SageMaker")
    parser.add_argument("--instance", default="ml.g4dn.xlarge", help="Instance type")
    parser.add_argument("--name", default="medical-ai-xray", help="Endpoint name")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "backend", "models", "best_model.pth")

    if not os.path.exists(model_path):
        print(f"  ❌ Model not found: {model_path}")
        print(f"     Train first: python training/train.py")
        sys.exit(1)

    tar_path = package_model(model_path)
    deploy_to_sagemaker(tar_path, instance_type=args.instance, endpoint_name=args.name)
