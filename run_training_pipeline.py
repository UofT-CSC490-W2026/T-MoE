"""
T-MoE Unified Training Pipeline — Conditional Dataset Handling.

Orchestrates the full workflow:
  1. Check if dataset already exists in S3
  2. If not → run data ingestion pipeline (run_pipeline.py logic)
  3. Download dataset from S3 to local cache
  4. Run HuggingFace model training (reuses train.py logic)
  5. Upload training outputs (checkpoints, logs, config) to S3

Usage:
    python run_training_pipeline.py --config gptneo_125m_lora
    python run_training_pipeline.py --config gptneo_125m_lora --dry-run

Environment:
    Requires AWS credentials and the following env vars / config:
      - RAW_DATA_BUCKET: S3 bucket for datasets and outputs
      - AWS_REGION: AWS region (default: us-east-1)
    See .env.example or run `cd infra/terraform && terraform output env_configuration`
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("tmoe.training_pipeline")


# =====================================================================
# Step 1: Configuration
# =====================================================================

def load_configs(args) -> tuple:
    """
    Load both pipeline config (for S3/ingestion) and experiment config
    (for training).

    Returns:
        (pipeline_config, experiment_config)
    """
    # Pipeline config (S3 bucket, dataset, AWS region)
    from infra.config.config import load_pipeline_config
    pipeline_config = load_pipeline_config()
    logger.info("Pipeline config loaded: bucket=%s, dataset=%s",
                pipeline_config.raw_data_bucket, pipeline_config.dataset_name)

    # Experiment config (model, training params)
    from src.utils.config_loader import load_experiment_config
    experiment_config = load_experiment_config(args.config, args.overrides or [])

    # Force execution_env to aws for S3-backed training
    from omegaconf import OmegaConf
    OmegaConf.update(experiment_config, "execution_env", "aws", merge=True)

    logger.info("Experiment config loaded: %s", experiment_config.experiment_name)
    return pipeline_config, experiment_config


# =====================================================================
# Step 2: S3 Dataset Check
# =====================================================================

def check_dataset_in_s3(pipeline_config) -> bool:
    """
    Check if the dataset already exists in S3.

    Uses S3Client.dataset_exists() to look for data files under the
    configured raw_data_prefix.

    Args:
        pipeline_config: PipelineConfig with S3 bucket/prefix info.

    Returns:
        True if dataset data files exist in S3, False otherwise.
    """
    from infra.s3client.client import S3Client

    s3_client = S3Client(
        region=pipeline_config.aws_region,
        max_retries=pipeline_config.max_retries,
    )

    prefix = pipeline_config.raw_data_prefix
    bucket = pipeline_config.raw_data_bucket

    logger.info("Checking for existing dataset in s3://%s/%s", bucket, prefix)
    return s3_client.dataset_exists(bucket, prefix)


# =====================================================================
# Step 3: Data Ingestion (conditional)
# =====================================================================

def run_data_ingestion(pipeline_config) -> Dict[str, Any]:
    """
    Run the data ingestion pipeline to upload dataset to S3.

    Reuses the existing fallback ingestion logic from run_pipeline.py.

    Args:
        pipeline_config: PipelineConfig instance.

    Returns:
        Ingestion result summary dict.
    """
    logger.info("=" * 70)
    logger.info("STEP: Running data ingestion (dataset not found in S3)")
    logger.info("=" * 70)

    from infra.data_ingestion.fallback_ingestion import FallbackIngestion

    ingestion = FallbackIngestion(
        dataset_name=pipeline_config.dataset_name,
        s3_bucket=pipeline_config.raw_data_bucket,
        s3_prefix=pipeline_config.raw_data_prefix,
        aws_region=pipeline_config.aws_region,
        dataset_config=pipeline_config.dataset_config,
        dataset_splits=pipeline_config.dataset_splits,
        output_format=pipeline_config.output_format,
        max_retries=pipeline_config.max_retries,
        log_level=pipeline_config.log_level,
    )

    result = ingestion.run()
    logger.info("Data ingestion complete: %d records uploaded", result.get("total_records", 0))
    return result


# =====================================================================
# Step 4: Download Dataset from S3 to Local Cache
# =====================================================================

def download_dataset_from_s3(pipeline_config, cache_dir: str) -> None:
    """
    Download dataset files from S3 to the local cache directory.

    Args:
        pipeline_config: PipelineConfig with S3 bucket/prefix info.
        cache_dir: Local directory to download files into.
    """
    logger.info("=" * 70)
    logger.info("STEP: Downloading dataset from S3 to local cache")
    logger.info("  From: s3://%s/%s", pipeline_config.raw_data_bucket,
                pipeline_config.raw_data_prefix)
    logger.info("  To:   %s", cache_dir)
    logger.info("=" * 70)

    from infra.s3client.s3_sync import download_s3_prefix

    downloaded = download_s3_prefix(
        bucket=pipeline_config.raw_data_bucket,
        s3_prefix=pipeline_config.raw_data_prefix,
        local_dir=cache_dir,
        aws_region=pipeline_config.aws_region,
        max_retries=pipeline_config.max_retries,
    )

    if not downloaded:
        raise RuntimeError(
            f"No files downloaded from s3://{pipeline_config.raw_data_bucket}/"
            f"{pipeline_config.raw_data_prefix}. "
            "Check that the ingestion pipeline ran successfully."
        )

    logger.info("Downloaded %d files to %s", len(downloaded), cache_dir)


# =====================================================================
# Step 5: Run Training
# =====================================================================

def run_training(experiment_config, cache_dir: str) -> tuple:
    """
    Run HuggingFace model training using the existing training logic.

    The experiment config is set to execution_env=aws with the cache_dir
    pointing to the downloaded S3 dataset.

    Args:
        experiment_config: OmegaConf DictConfig for training.
        cache_dir: Local cache directory containing the dataset.

    Returns:
        (output_dir, final_metrics) tuple.
    """
    import random
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    from src.utils.experiment import setup_experiment, build_model, build_dataloaders, build_optimizer
    from src.utils.logging import initialize_wandb, finalize_wandb
    from src.training.trainer import Trainer

    logger.info("=" * 70)
    logger.info("STEP: Running model training")
    logger.info("=" * 70)

    # Point AWS cache_dir to our downloaded dataset
    OmegaConf.update(experiment_config, "compute.aws.cache_dir", cache_dir, merge=True)

    # Set random seed
    seed = experiment_config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to: %d", seed)

    # Setup experiment directory
    output_dir = setup_experiment(experiment_config)
    logger.info("Output directory: %s", output_dir)

    # Save config
    config_path = Path(output_dir) / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(experiment_config, f)
    logger.info("Configuration saved to: %s", config_path)

    # Initialize WandB
    initialize_wandb(experiment_config, output_dir)

    # Build model
    logger.info("Building model...")
    model = build_model(experiment_config)
    logger.info("Model built: %d total params, %d trainable",
                model.get_total_params(), model.get_trainable_params())

    # Build dataloaders (will use S3-backed local files)
    logger.info("Loading datasets from local cache...")
    train_dataloader, val_dataloader = build_dataloaders(experiment_config)
    logger.info("Training batches: %d", len(train_dataloader))
    if val_dataloader:
        logger.info("Validation batches: %d", len(val_dataloader))

    # Build optimizer
    optimizer = build_optimizer(model, experiment_config)
    logger.info("Optimizer: %s, lr=%s", experiment_config.training.optimizer,
                experiment_config.training.lr)

    # Resolve device
    device = experiment_config.device.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training device: %s", device)

    # Create trainer and run
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        config=experiment_config,
        output_dir=output_dir,
        device=device,
    )

    try:
        final_metrics = trainer.train()
        logger.info("Training complete! Final loss: %.4f, Best loss: %.4f",
                     final_metrics["loss"], final_metrics["best_loss"])
    finally:
        finalize_wandb()

    return output_dir, final_metrics


# =====================================================================
# Step 6: Upload Outputs to S3
# =====================================================================

def upload_outputs_to_s3(pipeline_config, output_dir: str) -> None:
    """
    Upload the training output directory to S3.

    Preserves the full directory structure under:
        s3://{bucket}/experiments/{experiment_dir_name}/

    Args:
        pipeline_config: PipelineConfig with S3 bucket info.
        output_dir: Local experiment output directory.
    """
    logger.info("=" * 70)
    logger.info("STEP: Uploading training outputs to S3")
    logger.info("=" * 70)

    from infra.s3client.s3_sync import upload_experiment_dir

    output_path = Path(output_dir)
    s3_prefix = f"experiments/{output_path.name}/"

    result = upload_experiment_dir(
        local_dir=output_dir,
        bucket=pipeline_config.raw_data_bucket,
        s3_prefix=s3_prefix,
        aws_region=pipeline_config.aws_region,
        max_retries=pipeline_config.max_retries,
    )

    uploaded_count = len(result["uploaded"])
    failed_count = len(result["failed"])

    if failed_count > 0:
        logger.warning("Some files failed to upload: %d succeeded, %d failed",
                       uploaded_count, failed_count)
        for path in result["failed"]:
            logger.warning("  FAILED: %s", path)
    else:
        logger.info("All %d files uploaded to s3://%s/%s",
                    uploaded_count, pipeline_config.raw_data_bucket, s3_prefix)


# =====================================================================
# Main Entry Point
# =====================================================================

def main() -> None:
    """
    Main entry point — orchestrates the full conditional training pipeline.

    Exit codes:
        0: Success
        1: Configuration error
        2: Pipeline execution error
        130: Keyboard interrupt
    """
    parser = argparse.ArgumentParser(
        description="T-MoE Unified Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_training_pipeline.py --config gptneo_125m_lora
  python run_training_pipeline.py --config gptneo_125m_lora --dry-run
  python run_training_pipeline.py --config gptneo_125m_lora --skip-upload
        """,
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Experiment config name from experiments/ directory (without .yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the pipeline: check S3, log actions, but don't run anything.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading training outputs to S3.",
    )

    args, overrides = parser.parse_known_args()
    args.overrides = overrides

    pipeline_start = time.time()

    logger.info("=" * 70)
    logger.info("T-MoE Unified Training Pipeline")
    logger.info("  Project Root : %s", PROJECT_ROOT)
    logger.info("  Config       : %s", args.config)
    logger.info("  Dry Run      : %s", args.dry_run)
    logger.info("=" * 70)

    try:
        # Step 1: Load configuration
        pipeline_config, experiment_config = load_configs(args)

        # Step 2: Check if dataset exists in S3
        dataset_found = check_dataset_in_s3(pipeline_config)

        if dataset_found:
            logger.info("✅ Dataset found in S3 — skipping ingestion")
        else:
            logger.info("❌ Dataset NOT found in S3 — ingestion required")

        if args.dry_run:
            logger.info("")
            logger.info("=" * 70)
            logger.info("DRY RUN — Actions that would be taken:")
            logger.info("  1. Dataset in S3: %s", "YES (skip ingestion)" if dataset_found else "NO (run ingestion)")
            logger.info("  2. Download dataset from S3 to local cache")
            logger.info("  3. Run training with config: %s", args.config)
            logger.info("  4. Upload outputs to S3: %s", "YES" if not args.skip_upload else "SKIPPED")
            logger.info("=" * 70)
            sys.exit(0)

        # Step 3: Run ingestion if needed
        if not dataset_found:
            run_data_ingestion(pipeline_config)

        # Step 4: Download dataset from S3 to local cache
        cache_dir = experiment_config.compute.aws.cache_dir
        download_dataset_from_s3(pipeline_config, cache_dir)

        # Step 5: Run training
        output_dir, final_metrics = run_training(experiment_config, cache_dir)

        # Step 6: Upload outputs to S3
        if not args.skip_upload:
            upload_outputs_to_s3(pipeline_config, output_dir)
        else:
            logger.info("Skipping S3 output upload (--skip-upload flag)")

        # Summary
        elapsed = time.time() - pipeline_start
        logger.info("")
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("  Dataset in S3    : %s", "pre-existing" if dataset_found else "newly ingested")
        logger.info("  Final loss       : %.4f", final_metrics["loss"])
        logger.info("  Best loss        : %.4f", final_metrics["best_loss"])
        logger.info("  Output dir       : %s", output_dir)
        logger.info("  Total time       : %.1f s", elapsed)
        logger.info("=" * 70)
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("Pipeline cancelled by user")
        sys.exit(130)

    except ValueError as exc:
        logger.error("")
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    except ImportError as exc:
        logger.error("")
        logger.error("Dependency error: %s", exc)
        logger.error("Install required dependencies:")
        logger.error("  pip install -r requirements.txt")
        sys.exit(1)

    except Exception as exc:
        logger.error("")
        logger.error("Pipeline execution failed: %s", exc, exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
