"""
Modal backend for T-MoE training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infra.config.config import PipelineConfig

logger = logging.getLogger(__name__)


def run_modal_training(
    config: PipelineConfig, experiment_config_name: str, dry_run: bool = False
) -> None:
    """
    Execute training on Modal.

    Workflow:
        1. Ensure dataset exists in S3 (via shared utility)
        2. Launch Modal function with S3 dataset path and config
        3. Modal downloads from S3, trains, uploads results

    Args:
        config: PipelineConfig instance with Modal settings.
        experiment_config_name: Name of the experiment config.
        dry_run: If True, log actions but don't execute.

    Raises:
        ImportError: If modal package is not installed.
        RuntimeError: If Modal training fails.
    """
    import importlib.util

    if importlib.util.find_spec("modal") is None:
        logger.error("Modal package not installed")
        logger.error("Install with: pip install modal")
        raise ImportError("Modal package required for Modal backend")

    from infra.dataset.ensure_dataset import ensure_dataset_in_s3

    logger.info("=" * 70)
    logger.info("MODAL TRAINING MODE")
    logger.info("  Config: %s", experiment_config_name)
    logger.info("  Dry Run: %s", dry_run)
    logger.info("  GPU: %s", config.modal_gpu or "default")
    logger.info("  CPU: %s", config.modal_cpu or "default")
    logger.info("  Timeout: %s", config.modal_timeout or "default")
    logger.info("=" * 70)

    # Step 1: Ensure dataset in S3
    s3_path = ensure_dataset_in_s3(config)
    logger.info("Dataset ready at: %s", s3_path)

    if dry_run:
        logger.info("")
        logger.info("=" * 70)
        logger.info("DRY RUN — Would launch Modal training")
        logger.info("  Dataset: %s", s3_path)
        logger.info("  Config: %s", experiment_config_name)
        logger.info("=" * 70)
        return

    # Step 2: Create Modal app and launch training
    app = _create_modal_app(config)

    logger.info("Launching Modal training function...")
    with app.run():
        app.train_tmoe.remote(
            s3_dataset_path=s3_path,
            experiment_config_name=experiment_config_name,
            aws_region=config.aws_region,
            raw_data_bucket=config.raw_data_bucket,
            dataset_name=config.dataset_name,
        )

    logger.info("")
    logger.info("=" * 70)
    logger.info("MODAL TRAINING COMPLETED")
    logger.info("  Check Modal dashboard for details")
    logger.info("=" * 70)


# Global Modal resources (created on demand)
_modal_app = None
_modal_image = None
_modal_volume = None


def _get_modal_image():
    """Get or create Modal image."""
    global _modal_image
    if _modal_image is None:
        import modal

        _modal_image = (
            modal.Image.debian_slim(python_version="3.11")
            .pip_install(
                "numpy==2.2.6",
                "torch==2.10.0",
                "transformers>=4.35.0",
                "datasets>=2.14.0",
                "tokenizers>=0.15.0",
                "hydra-core>=1.3.0",
                "omegaconf>=2.3.0",
                "wandb==0.24.1",
                "boto3",
            )
            .workdir("/root/tmoe")
            .add_local_dir(
                local_path=str(Path(__file__).resolve().parent.parent.parent),
                remote_path="/root/tmoe",
            )
        )
    return _modal_image


def _create_modal_app(config: PipelineConfig):
    """Create and configure Modal app with training function."""
    import modal

    global _modal_app, _modal_volume

    if _modal_app is None:
        _modal_app = modal.App("tmoe-training")

    volume_name = f"tmoe-results-{config.environment}"
    _modal_volume = modal.Volume.from_name(
        volume_name,
        create_if_missing=True,
    )

    image = _get_modal_image()

    # Always attach AWS credentials so the remote container can access S3
    aws_secret = modal.Secret.from_name("aws-credentials")
    secrets = [aws_secret]

    # Check if WandB secret exists before trying to use it
    try:
        # Try to get the secret - this will raise if it doesn't exist
        import subprocess

        result = subprocess.run(
            ["modal", "secret", "list"], capture_output=True, text=True, timeout=5
        )
        if "wandb-secret" in result.stdout:
            wandb_secret = modal.Secret.from_name("wandb-secret")
            secrets.append(wandb_secret)
            logger.info("WandB secret found, will enable WandB logging")
        else:
            logger.info("WandB secret not found, will disable WandB")
    except Exception as e:
        logger.info(f"Could not check for WandB secret: {e}, will disable WandB")

    # Attach the global training function to the app
    _modal_app.train_tmoe = _modal_app.function(
        gpu=config.modal_gpu or "A10G:1",
        cpu=config.modal_cpu or 4,
        timeout=config.modal_timeout or 14400,
        volumes={"/root/data": _modal_volume},
        secrets=secrets,
        image=image,
    )(_modal_train_tmoe_impl)

    return _modal_app


# Global training implementation
def _modal_train_tmoe_impl(
    s3_dataset_path: str,
    experiment_config_name: str,
    aws_region: str,
    raw_data_bucket: str,
    dataset_name: str,
):
    """Remote training function that runs on Modal."""
    import sys

    sys.path.insert(0, "/root/tmoe")

    from pathlib import Path
    from omegaconf import OmegaConf
    from src.utils.config_loader import load_experiment_config
    from src.utils.training_workflow import execute_training_workflow

    print("=" * 70)
    print("MODAL REMOTE TRAINING")
    print("=" * 70)

    print("\nStep 1: Downloading dataset from S3...")
    dataset_cache = Path("/tmp/tmoe_dataset")
    dataset_cache.mkdir(parents=True, exist_ok=True)

    _download_from_s3(
        s3_path=s3_dataset_path,
        local_dir=str(dataset_cache),
        aws_region=aws_region,
    )

    print("\nStep 2: Starting training...")
    print(f"Loading experiment config: {experiment_config_name}")
    experiment_config = load_experiment_config(experiment_config_name, [])

    OmegaConf.update(experiment_config, "execution_env", "aws", merge=True)

    output_dir, final_metrics = execute_training_workflow(
        experiment_config=experiment_config,
        cache_dir=str(dataset_cache),
    )

    print("\nStep 3: Uploading results to S3...")
    from infra.s3client.s3_sync import upload_experiment_dir

    output_path = Path(output_dir)
    s3_prefix = f"experiments/{output_path.name}/"

    result = upload_experiment_dir(
        local_dir=output_dir,
        bucket=raw_data_bucket,
        s3_prefix=s3_prefix,
        aws_region=aws_region,
        max_retries=3,
    )

    uploaded_count = len(result["uploaded"])
    failed_count = len(result["failed"])

    if failed_count > 0:
        checkpoint_failures = [
            p for p in result["failed"] if "checkpoint" in str(p).lower()
        ]
        if checkpoint_failures:
            raise RuntimeError(
                f"Critical checkpoint upload failed: {checkpoint_failures}"
            )

    print("\n" + "=" * 70)
    print("Modal training complete!")
    print(f"Results uploaded to s3://{raw_data_bucket}/{s3_prefix}")
    print(f"Uploaded: {uploaded_count}, Failed: {failed_count}")
    print("=" * 70)


def _download_from_s3(s3_path: str, local_dir: str, aws_region: str) -> None:
    """
    Download dataset from S3 to local directory.

    Args:
        s3_path: S3 path in format s3://bucket/prefix
        local_dir: Local directory to download to.
        aws_region: AWS region.
    """
    import boto3
    from pathlib import Path

    # Parse S3 path
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}")

    parts = s3_path[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""

    s3_client = boto3.client("s3", region_name=aws_region)

    # List and download objects
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Skip directories
            if key.endswith("/"):
                continue

            # Determine local path
            relative_path = key[len(prefix) :].lstrip("/")
            local_file = Path(local_dir) / relative_path
            local_file.parent.mkdir(parents=True, exist_ok=True)

            # Download
            print(f"  Downloading: {key}")
            s3_client.download_file(bucket, key, str(local_file))

    print(f"Downloaded files from {s3_path} to {local_dir}")
