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
    python run_aws_training.py --mode local --config gptneo_125m_lora

    # Submit to AWS Batch
    python run_aws_training.py --mode batch --config gptneo_125m_lora

    # Inside container (called by Batch, not invoked directly)
    python run_aws_training.py --mode container --config gptneo_125m_lora

    # Dry run (any mode)
    python run_aws_training.py --mode batch --config gptneo_125m_lora --dry-run
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


def _dataset_s3_prefix(pipeline_config) -> str:
    """Build the dataset-specific S3 prefix: datasets/raw/{dataset_name}/."""
    safe_name = pipeline_config.dataset_name.replace("/", "_")
    base = pipeline_config.raw_data_prefix.rstrip("/") + "/"
    return f"{base}{safe_name}/"


def _find_latest_timestamp_prefix(s3_client, bucket: str, dataset_prefix: str) -> str:
    """
    Find the latest timestamp directory under a dataset prefix.

    Args:
        s3_client: S3Client instance
        bucket: S3 bucket name
        dataset_prefix: Prefix like 'datasets/raw/wikitext/'

    Returns:
        Full prefix including latest timestamp, e.g. 'datasets/raw/wikitext/20240216-150000/'

    Raises:
        RuntimeError: If no timestamp directories found
    """
    # List all objects under the dataset prefix
    objects = s3_client.list_objects(bucket, dataset_prefix)
    if not objects:
        raise RuntimeError(f"No objects found under s3://{bucket}/{dataset_prefix}")

    # Extract unique timestamp directories from keys
    # Keys look like: datasets/raw/wikitext/20240216-143022/train.jsonl
    timestamps = set()
    for obj in objects:
        key = obj["Key"]
        # Remove the dataset prefix to get relative path
        relative = key[len(dataset_prefix) :].lstrip("/")
        # Extract timestamp (first directory component)
        parts = relative.split("/")
        if parts and parts[0]:
            # Validate timestamp format: YYYYMMDD-HHMMSS
            timestamp = parts[0]
            if len(timestamp) == 15 and timestamp[8] == "-":
                timestamps.add(timestamp)

    if not timestamps:
        raise RuntimeError(
            f"No timestamp directories found under s3://{bucket}/{dataset_prefix}"
        )

    # Sort timestamps (lexicographic sort works for YYYYMMDD-HHMMSS format)
    latest_timestamp = sorted(timestamps)[-1]
    latest_prefix = f"{dataset_prefix}{latest_timestamp}/"

    logger.info(
        "Found %d timestamp(s), selecting latest: %s",
        len(timestamps),
        latest_timestamp,
    )

    return latest_prefix


def check_dataset_in_s3(pipeline_config) -> bool:
    """Check if the dataset already exists in S3 (checks latest timestamp)."""
    from infra.s3client.client import S3Client

    s3_client = S3Client(
        region=pipeline_config.aws_region,
        max_retries=pipeline_config.max_retries,
    )
    dataset_prefix = _dataset_s3_prefix(pipeline_config)

    try:
        latest_prefix = _find_latest_timestamp_prefix(
            s3_client,
            pipeline_config.raw_data_bucket,
            dataset_prefix,
        )
        logger.info(
            "Checking for existing dataset in s3://%s/%s",
            pipeline_config.raw_data_bucket,
            latest_prefix,
        )
        return s3_client.dataset_exists(pipeline_config.raw_data_bucket, latest_prefix)
    except RuntimeError:
        # No timestamps found = dataset doesn't exist
        logger.info(
            "No timestamp directories found under s3://%s/%s",
            pipeline_config.raw_data_bucket,
            dataset_prefix,
        )
        return False


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
    from infra.s3client.client import S3Client

    dataset_prefix = _dataset_s3_prefix(pipeline_config)
    s3_client = S3Client(
        region=pipeline_config.aws_region,
        max_retries=pipeline_config.max_retries,
    )

    # Find latest timestamp directory
    latest_prefix = _find_latest_timestamp_prefix(
        s3_client,
        pipeline_config.raw_data_bucket,
        dataset_prefix,
    )

    logger.info("=" * 70)
    logger.info("STEP: Downloading dataset from S3 to local cache")
    logger.info(
        "  s3://%s/%s → %s",
        pipeline_config.raw_data_bucket,
        latest_prefix,
        cache_dir,
    )
    logger.info("=" * 70)

    from infra.s3client.s3_sync import download_s3_prefix

    downloaded = download_s3_prefix(
        bucket=pipeline_config.raw_data_bucket,
        s3_prefix=latest_prefix,  # Use latest timestamp prefix
        local_dir=cache_dir,
        aws_region=pipeline_config.aws_region,
        max_retries=pipeline_config.max_retries,
    )
    if not downloaded:
        raise RuntimeError(
            f"No files downloaded from s3://{pipeline_config.raw_data_bucket}/"
            f"{latest_prefix}"
        )
    logger.info("Downloaded %d files to %s", len(downloaded), cache_dir)

    import tarfile
    import zipfile

    cache_path = Path(cache_dir)
    for archive in cache_path.glob("*.tar.gz"):
        logger.info("Extracting %s", archive.name)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(cache_path, filter="data")
    for archive in cache_path.glob("*.tar"):
        logger.info("Extracting %s", archive.name)
        with tarfile.open(archive, "r") as tar:
            tar.extractall(cache_path, filter="data")
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
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    s3_prefix = f"checkpoints/{output_path.name}/{timestamp}/"

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


def run_training(
    experiment_config, cache_dir: str, config_name: str | None = None
) -> tuple:
    """
    Run model training using the shared training workflow.

    Args:
        experiment_config: Loaded OmegaConf config.
        cache_dir: Local directory for shards and outputs.
        config_name: YAML stem used to load the config (e.g. "qwen2_1.5b_stress_v3-fineweb").
            Passed through to execute_training_workflow to avoid experiment_name/filename mismatch.
    """
    from src.utils.training_workflow import execute_training_workflow

    logger.info("=" * 70)
    logger.info("STEP: Running model training")
    logger.info("=" * 70)

    return execute_training_workflow(
        experiment_config, cache_dir, config_name=config_name
    )


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

    command = ["--mode", "container", "--config", config_name]
    if overrides:
        command.extend(overrides)

    # Environment variables to pass to container
    environment = [
        {"name": "RAW_DATA_BUCKET", "value": pipeline_config.raw_data_bucket},
        {"name": "AWS_REGION", "value": pipeline_config.aws_region},
        {"name": "ENVIRONMENT", "value": os.environ.get("ENVIRONMENT", "dev")},
        {"name": "DATASET_NAME", "value": pipeline_config.dataset_name},
    ]

    if wandb_api_key := os.environ.get("WANDB_API_KEY"):
        environment.append({"name": "WANDB_API_KEY", "value": wandb_api_key})
    else:
        environment.append({"name": "WANDB_MODE", "value": "disabled"})

    if hf_token := os.environ.get("HF_TOKEN"):
        environment.append({"name": "HF_TOKEN", "value": hf_token})
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

    from omegaconf import OmegaConf as _OC

    cache_dir = _OC.select(
        experiment_config, "compute.aws.cache_dir", default="/tmp/tmoe_data"
    )
    download_dataset_from_s3(pipeline_config, cache_dir)

    output_dir, final_metrics = run_training(
        experiment_config, cache_dir, config_name=args.config
    )

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


def run_post_training_evals(
    experiment_config,
    output_dir: str,
    cache_dir: str | None = None,
) -> None:
    """
    Run perplexity and lm_harness evals in-process after training completes.
    Results are written into <output_dir>/eval/ so they get uploaded to S3
    alongside the checkpoints.

    Args:
        experiment_config: Loaded OmegaConf config.
        output_dir: Training output directory (contains checkpoints/).
        cache_dir: Local data cache directory (contains shards/). When provided,
            SHARD_BASE_DIR is set so perplexity eval finds eval shards under
            <cache_dir>/shards/ instead of the default data/shards/ path.
    """
    from pathlib import Path as _Path
    from omegaconf import OmegaConf as _OC
    import torch

    output_path = _Path(output_dir)
    checkpoints_dir = output_path / "checkpoints"
    eval_output_dir = output_path / "eval"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Find the best checkpoint (best_model.pt preferred, else latest step checkpoint).
    best_ckpt = checkpoints_dir / "best_model.pt"
    if not best_ckpt.exists():
        step_ckpts = sorted(
            checkpoints_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: (
                int(p.stem.rsplit("_", 1)[-1])
                if p.stem.rsplit("_", 1)[-1].isdigit()
                else 0
            ),
        )
        if not step_ckpts:
            logger.warning(
                "No checkpoints found in %s — skipping post-training eval.",
                checkpoints_dir,
            )
            return
        best_ckpt = step_ckpts[-1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if device.startswith("cuda") else None

    # Read eval hyperparams from config (same keys as Modal's stage_eval).
    def _ep(key, default):
        return _OC.select(experiment_config, f"eval.{key}", default=default)

    stride = _ep("stride", 512)
    max_documents = _ep("max_documents", None)
    max_eval_length = _ep("max_eval_length", 2048)
    ppl_batch_size = int(_ep("batch_size", 32))
    lm_batch_size = _ep("lm_harness_batch_size", 1)
    limit = _ep("limit", None)

    logger.info("=" * 70)
    logger.info("STEP: Post-training evaluation")
    logger.info("  Checkpoint : %s", best_ckpt)
    logger.info("  Output     : %s", eval_output_dir)
    logger.info("=" * 70)

    # Point perplexity eval at the correct shard base directory.
    # evals/perplexity.py reads SHARD_BASE_DIR at module import time, so we must
    # set the env var BEFORE the first import of evals.perplexity. Use direct
    # assignment (not setdefault) to override any stale value from the container env.
    if cache_dir is not None:
        shard_base = str(_Path(cache_dir) / "shards")
        os.environ["SHARD_BASE_DIR"] = shard_base

    try:
        from evals.loading import load_model_for_eval

        model, checkpoint_info = load_model_for_eval(
            config=experiment_config,
            checkpoint_path=best_ckpt,
            device=device,
            dtype=autocast_dtype,
        )
    except Exception as exc:
        logger.error("Failed to load model for eval: %s", exc)
        return

    # Perplexity eval (wikitext-103 + pile-val shards).
    try:
        from evals.perplexity import run_perplexity_eval
        from evals.results_schema import log_results_to_wandb

        ppl_payload = run_perplexity_eval(
            config=experiment_config,
            checkpoint_path=best_ckpt,
            model=model,
            checkpoint_info=checkpoint_info,
            output_path=eval_output_dir / "perplexity.json",
            device=device,
            stride=stride,
            max_documents=max_documents,
            autocast_dtype=autocast_dtype,
            batch_size=ppl_batch_size,
            max_eval_length=max_eval_length,
        )
        log_results_to_wandb(ppl_payload, config=experiment_config)
        logger.info("Perplexity eval complete.")
    except Exception as exc:
        logger.error("Perplexity eval failed (non-fatal): %s", exc, exc_info=True)

    # LM harness eval.
    try:
        from evals.lm_harness_runner import run_lm_harness_eval
        from evals.results_schema import log_results_to_wandb

        lm_payload = run_lm_harness_eval(
            config=experiment_config,
            checkpoint_path=best_ckpt,
            model=model,
            checkpoint_info=checkpoint_info,
            output_path=eval_output_dir / "lm_harness.json",
            device=device,
            batch_size=lm_batch_size,
            limit=limit,
        )
        log_results_to_wandb(lm_payload, config=experiment_config)
        logger.info("LM harness eval complete.")
    except Exception as exc:
        logger.error("LM harness eval failed (non-fatal): %s", exc, exc_info=True)


def run_container_mode(args, pipeline_config, experiment_config) -> None:
    """
    CONTAINER mode: runs inside Docker on Batch GPU instance.
    Downloads dataset, trains (single- or multi-GPU via config), uploads outputs.

    Uses try/finally to upload partial checkpoints to S3 even if training
    crashes or the instance is preempted.
    """
    from omegaconf import OmegaConf

    cache_dir = OmegaConf.select(
        experiment_config, "compute.aws.cache_dir", default="/tmp/tmoe_data"
    )
    download_dataset_from_s3(pipeline_config, cache_dir)

    output_dir = None
    final_metrics = {"loss": float("inf"), "best_loss": float("inf")}
    try:
        output_dir, final_metrics = run_training(
            experiment_config, cache_dir, config_name=args.config
        )
        # Run evals in-process — results land in <output_dir>/eval/ and are
        # uploaded to S3 with the checkpoints in the finally block below.
        run_post_training_evals(experiment_config, output_dir, cache_dir=cache_dir)
    finally:
        # Always attempt S3 upload — partial checkpoints are better than none.
        if output_dir:
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
  python run_aws_training.py --mode local --config gptneo_125m_lora
  python run_aws_training.py --mode batch --config gptneo_125m_lora
  python run_aws_training.py --mode batch --config gptneo_125m_lora --dry-run
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
