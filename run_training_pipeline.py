"""
T-MoE Unified Training Pipeline — AWS Batch Integration.

Orchestrates the full training workflow with three execution modes:

  LOCAL mode (--mode local):
    Runs training in-process on the current machine.
    1. Check S3 for dataset → 2. Ingest if needed → 3. Download to cache
    4. Train locally → 5. Upload outputs to S3

  BATCH mode (--mode batch):
    Submits training as an AWS Batch job, streams logs, exits with job status.
    1. Check S3 for dataset → 2. Ingest if needed
    3. Submit Batch job → 4. Poll status → 5. Stream logs → 6. Exit

  CONTAINER mode (--mode container):
    Runs inside the Docker container on AWS Batch GPU instance.
    1. Download dataset from S3 → 2. Train → 3. Upload outputs to S3

Usage:
    # Local training
    python run_training_pipeline.py --mode local --config gptneo_125m_lora

    # Submit to AWS Batch
    python run_training_pipeline.py --mode batch --config gptneo_125m_lora

    # Inside container (called by Batch, not invoked directly)
    python run_training_pipeline.py --mode container --config gptneo_125m_lora

    # Dry run (any mode)
    python run_training_pipeline.py --mode batch --config gptneo_125m_lora --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
# Configuration
# =====================================================================


def load_configs(args) -> tuple:
    """
    Load both pipeline config (for S3/ingestion) and experiment config
    (for training).

    Returns:
        (pipeline_config, experiment_config)
    """
    from infra.config.config import load_pipeline_config

    pipeline_config = load_pipeline_config()
    logger.info(
        "Pipeline config loaded: bucket=%s, dataset=%s",
        pipeline_config.raw_data_bucket,
        pipeline_config.dataset_name,
    )

    from src.utils.config_loader import load_experiment_config

    experiment_config = load_experiment_config(args.config, args.overrides or [])

    # Force execution_env to aws when using S3
    from omegaconf import OmegaConf

    OmegaConf.update(experiment_config, "execution_env", "aws", merge=True)

    logger.info("Experiment config loaded: %s", experiment_config.experiment_name)
    return pipeline_config, experiment_config


# =====================================================================
# S3 Operations (shared across modes)
# =====================================================================


def check_dataset_in_s3(pipeline_config) -> bool:
    """Check if the dataset already exists in S3."""
    from infra.s3client.client import S3Client

    s3_client = S3Client(
        region=pipeline_config.aws_region,
        max_retries=pipeline_config.max_retries,
    )
    prefix = pipeline_config.raw_data_prefix
    bucket = pipeline_config.raw_data_bucket
    logger.info("Checking for existing dataset in s3://%s/%s", bucket, prefix)
    return s3_client.dataset_exists(bucket, prefix)


def run_data_ingestion(pipeline_config) -> Dict[str, Any]:
    """Run the data ingestion pipeline to upload dataset to S3."""
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
    logger.info(
        "Data ingestion complete: %d records uploaded", result.get("total_records", 0)
    )
    return result


def download_dataset_from_s3(pipeline_config, cache_dir: str) -> None:
    """Download dataset files from S3 to the local cache directory."""
    logger.info("=" * 70)
    logger.info("STEP: Downloading dataset from S3 to local cache")
    logger.info(
        "  s3://%s/%s → %s",
        pipeline_config.raw_data_bucket,
        pipeline_config.raw_data_prefix,
        cache_dir,
    )
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
            f"{pipeline_config.raw_data_prefix}"
        )
    logger.info("Downloaded %d files to %s", len(downloaded), cache_dir)

    import tarfile
    import zipfile

    cache_path = Path(cache_dir)
    for archive in cache_path.glob("*.tar.gz"):
        logger.info("Extracting %s", archive.name)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(cache_path)
    for archive in cache_path.glob("*.tar"):
        logger.info("Extracting %s", archive.name)
        with tarfile.open(archive, "r") as tar:
            tar.extractall(cache_path)
    for archive in cache_path.glob("*.zip"):
        logger.info("Extracting %s", archive.name)
        with zipfile.ZipFile(archive, "r") as zip_ref:
            zip_ref.extractall(cache_path)


def upload_outputs_to_s3(pipeline_config, output_dir: str) -> None:
    """Upload the training output directory to S3."""
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
        logger.warning("%d files uploaded, %d failed", uploaded_count, failed_count)
        checkpoint_failures = [
            p for p in result["failed"] if "checkpoint" in str(p).lower()
        ]
        for path in result["failed"]:
            logger.warning("  FAILED: %s", path)
        if checkpoint_failures:
            raise RuntimeError(
                f"Critical checkpoint upload failed: {checkpoint_failures}"
            )
    else:
        logger.info(
            "All %d files uploaded to s3://%s/%s",
            uploaded_count,
            pipeline_config.raw_data_bucket,
            s3_prefix,
        )


# =====================================================================
# Training (for local and container modes)
# =====================================================================


def run_training(experiment_config, cache_dir: str) -> tuple:
    """
    Run model training using the shared training workflow.

    This is a thin wrapper around execute_training_workflow for backwards compatibility.
    """
    from src.utils.training_workflow import execute_training_workflow

    logger.info("=" * 70)
    logger.info("STEP: Running model training")
    logger.info("=" * 70)

    return execute_training_workflow(experiment_config, cache_dir)


# =====================================================================
# AWS Batch Submission (batch mode)
# =====================================================================


def submit_batch_job(
    config_name: str,
    pipeline_config,
    overrides: List[str],
) -> str:
    """
    Submit a training job to AWS Batch.

    Args:
        config_name: Experiment config name.
        pipeline_config: Pipeline config with AWS settings.
        overrides: Additional config overrides.

    Returns:
        Batch job ID.
    """
    import boto3

    batch_client = boto3.client("batch", region_name=pipeline_config.aws_region)

    job_queue = os.environ.get(
        "BATCH_JOB_QUEUE", f"tmoe-{os.environ.get('ENVIRONMENT', 'dev')}-training"
    )
    job_definition = os.environ.get(
        "BATCH_JOB_DEFINITION", f"tmoe-{os.environ.get('ENVIRONMENT', 'dev')}-training"
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"tmoe-training-{config_name}-{timestamp}"

    # Job command (only arguments, since entrypoint is python run_training_pipeline.py --mode container)
    command = ["--config", config_name]
    if overrides:
        command.extend(overrides)

    # Environment variables to pass to container
    environment = [
        {"name": "RAW_DATA_BUCKET", "value": pipeline_config.raw_data_bucket},
        {"name": "AWS_REGION", "value": pipeline_config.aws_region},
        {"name": "ENVIRONMENT", "value": os.environ.get("ENVIRONMENT", "dev")},
    ]

    if wandb_api_key := os.environ.get("WANDB_API_KEY"):
        environment.append({"name": "WANDB_API_KEY", "value": wandb_api_key})
    else:
        environment.append({"name": "WANDB_MODE", "value": "disabled"})
    logger.info("Submitting Batch job:")
    logger.info("  Job Name       : %s", job_name)
    logger.info("  Job Queue      : %s", job_queue)
    logger.info("  Job Definition : %s", job_definition)
    logger.info("  Command        : %s", " ".join(command))

    response = batch_client.submit_job(
        jobName=job_name,
        jobQueue=job_queue,
        jobDefinition=job_definition,
        containerOverrides={
            "command": command,
            "environment": environment,
        },
    )

    job_id = response["jobId"]
    logger.info("Job submitted! ID: %s", job_id)
    return job_id


def wait_for_batch_job(
    job_id: str,
    aws_region: str,
    poll_interval: int = 30,
    stream_logs: bool = True,
) -> str:
    """
    Wait for a Batch job to complete, optionally streaming CloudWatch logs.

    Args:
        job_id: AWS Batch job ID.
        aws_region: AWS region.
        poll_interval: Seconds between status polls.
        stream_logs: Whether to stream CloudWatch logs.

    Returns:
        Final job status: 'SUCCEEDED' or 'FAILED'.
    """
    import boto3

    batch_client = boto3.client("batch", region_name=aws_region)
    logs_client = boto3.client("logs", region_name=aws_region) if stream_logs else None

    terminal_states = {"SUCCEEDED", "FAILED"}
    last_log_token = None
    log_stream_name = None

    logger.info("Waiting for job %s...", job_id)

    while True:
        response = batch_client.describe_jobs(jobs=[job_id])
        if not response["jobs"]:
            logger.error("Job %s not found!", job_id)
            return "FAILED"

        job = response["jobs"][0]
        status = job["status"]
        status_reason = job.get("statusReason", "")

        logger.info(
            "  Status: %s %s", status, f"({status_reason})" if status_reason else ""
        )

        # Try to stream logs when job is RUNNING or finished
        if stream_logs and logs_client and status in ("RUNNING", "SUCCEEDED", "FAILED"):
            last_log_token, log_stream_name = _stream_job_logs(
                logs_client,
                job,
                log_stream_name,
                last_log_token,
                aws_region,
            )

        if status in terminal_states:
            return status

        time.sleep(poll_interval)


def _stream_job_logs(
    logs_client,
    job: dict,
    log_stream_name: Optional[str],
    last_token: Optional[str],
    aws_region: str,
) -> tuple:
    """Stream new CloudWatch log events for a Batch job."""
    try:
        # Resolve log stream name from job container details
        if log_stream_name is None:
            container = job.get("container", {})
            log_stream_name = container.get("logStreamName")
            if not log_stream_name:
                return last_token, None

        log_group = os.environ.get(
            "BATCH_LOG_GROUP",
            f"/aws/batch/tmoe-{os.environ.get('ENVIRONMENT', 'dev')}/training",
        )

        kwargs = {
            "logGroupName": log_group,
            "logStreamName": log_stream_name,
            "startFromHead": True,
        }
        if last_token:
            kwargs["nextToken"] = last_token

        response = logs_client.get_log_events(**kwargs)

        for event in response.get("events", []):
            msg = event.get("message", "").rstrip()
            if msg:
                print(f"  [BATCH] {msg}")

        new_token = response.get("nextForwardToken")
        return new_token, log_stream_name

    except Exception as e:
        logger.debug("Log streaming error (non-fatal): %s", e)
        return last_token, log_stream_name


# =====================================================================
# Mode Handlers
# =====================================================================


def run_local_mode(args, pipeline_config, experiment_config) -> None:
    """
    LOCAL mode: full pipeline runs in-process on the current machine.
    """
    dataset_found = check_dataset_in_s3(pipeline_config)
    _log_dataset_status(dataset_found)

    if args.dry_run:
        _log_dry_run(dataset_found, args)
        return

    if not dataset_found:
        run_data_ingestion(pipeline_config)

    cache_dir = experiment_config.compute.aws.cache_dir
    download_dataset_from_s3(pipeline_config, cache_dir)

    output_dir, final_metrics = run_training(experiment_config, cache_dir)

    if not args.skip_upload:
        upload_outputs_to_s3(pipeline_config, output_dir)

    _log_completion(dataset_found, final_metrics, output_dir)


def run_batch_mode(args, pipeline_config, experiment_config) -> None:
    """
    BATCH mode: ensure dataset is in S3, then submit a Batch job.
    """
    dataset_found = check_dataset_in_s3(pipeline_config)
    _log_dataset_status(dataset_found)

    if args.dry_run:
        _log_dry_run(dataset_found, args)
        return

    # Ensure dataset is in S3 before submitting (Batch container expects it)
    if not dataset_found:
        run_data_ingestion(pipeline_config)

    # Submit Batch job
    job_id = submit_batch_job(args.config, pipeline_config, args.overrides or [])

    # Wait for completion
    final_status = wait_for_batch_job(
        job_id=job_id,
        aws_region=pipeline_config.aws_region,
        poll_interval=30,
        stream_logs=True,
    )

    logger.info("")
    logger.info("=" * 70)
    if final_status == "SUCCEEDED":
        logger.info("BATCH JOB SUCCEEDED — Job ID: %s", job_id)
        sys.exit(0)
    else:
        logger.error("BATCH JOB FAILED — Job ID: %s", job_id)
        sys.exit(1)


def run_container_mode(args, pipeline_config, experiment_config) -> None:
    """
    CONTAINER mode: runs inside Docker on Batch GPU instance.
    Downloads dataset, trains, uploads outputs.
    """
    cache_dir = experiment_config.compute.aws.cache_dir
    download_dataset_from_s3(pipeline_config, cache_dir)

    output_dir, final_metrics = run_training(experiment_config, cache_dir)

    upload_outputs_to_s3(pipeline_config, output_dir)

    _log_completion(True, final_metrics, output_dir)


# =====================================================================
# Helpers
# =====================================================================


def _log_dataset_status(found: bool) -> None:
    if found:
        logger.info("✅ Dataset found in S3 — skipping ingestion")
    else:
        logger.info("❌ Dataset NOT found in S3 — ingestion required")


def _log_dry_run(dataset_found: bool, args) -> None:
    logger.info("")
    logger.info("=" * 70)
    logger.info("DRY RUN — Actions that would be taken:")
    logger.info("  Mode         : %s", args.mode)
    logger.info(
        "  Dataset in S3: %s",
        "YES (skip ingestion)" if dataset_found else "NO (run ingestion)",
    )
    if args.mode == "batch":
        logger.info("  Action       : Submit Batch job and stream logs")
    else:
        logger.info("  Action       : Train locally, upload outputs to S3")
    logger.info("=" * 70)


def _log_completion(dataset_found: bool, metrics: dict, output_dir: str) -> None:
    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(
        "  Dataset      : %s", "pre-existing" if dataset_found else "newly ingested"
    )
    logger.info("  Final loss   : %.4f", metrics["loss"])
    logger.info("  Best loss    : %.4f", metrics["best_loss"])
    logger.info("  Output dir   : %s", output_dir)
    logger.info("=" * 70)


# =====================================================================
# Main Entry Point
# =====================================================================


def main() -> None:
    """
    Main entry point — routes to the appropriate execution mode.

    Exit codes: 0=success, 1=failure/config error, 2=runtime error, 130=interrupt
    """
    parser = argparse.ArgumentParser(
        description="T-MoE Training Pipeline — Local or AWS Batch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_training_pipeline.py --mode local --config gptneo_125m_lora
  python run_training_pipeline.py --mode batch --config gptneo_125m_lora
  python run_training_pipeline.py --mode batch --config gptneo_125m_lora --dry-run
        """,
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["local", "batch", "container"],
        default="local",
        help="Execution mode: local (in-process), batch (submit to AWS Batch), "
        "container (inside Docker, called by Batch).",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Experiment config name from experiments/ directory (without .yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate: check S3, log actions, but don't run anything.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading training outputs to S3 (local mode only).",
    )

    args, overrides = parser.parse_known_args()
    args.overrides = overrides

    pipeline_start = time.time()

    logger.info("=" * 70)
    logger.info("T-MoE Training Pipeline")
    logger.info("  Mode    : %s", args.mode)
    logger.info("  Config  : %s", args.config)
    logger.info("  Dry Run : %s", args.dry_run)
    logger.info("=" * 70)

    try:
        pipeline_config, experiment_config = load_configs(args)

        mode_handlers = {
            "local": run_local_mode,
            "batch": run_batch_mode,
            "container": run_container_mode,
        }
        mode_handlers[args.mode](args, pipeline_config, experiment_config)

        elapsed = time.time() - pipeline_start
        logger.info("Total time: %.1f s", elapsed)
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("Pipeline cancelled by user")
        sys.exit(130)

    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    except ImportError as exc:
        logger.error("Dependency error: %s", exc)
        logger.error("Install: pip install -r requirements.txt")
        sys.exit(1)

    except Exception as exc:
        logger.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
