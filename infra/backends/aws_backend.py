from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infra.config.config import PipelineConfig

logger = logging.getLogger(__name__)


def run_aws_training(
    config: PipelineConfig, experiment_config_name: str, dry_run: bool = False
) -> None:
    """
    Execute training on AWS Batch.

    Workflow:
        1. Ensure dataset exists in S3 (via shared utility)
        2. Submit Batch job with container mode
        3. Stream logs and wait for completion

    Args:
        config: PipelineConfig instance with AWS settings.
        experiment_config_name: Name of the experiment config (e.g., "gptneo_125m_lora")
        dry_run: If True, log actions but don't execute.

    Raises:
        RuntimeError: If Batch job fails.
    """
    from infra.dataset.ensure_dataset import ensure_dataset_in_s3

    logger.info("=" * 70)
    logger.info("AWS BATCH TRAINING MODE")
    logger.info("  Config: %s", experiment_config_name)
    logger.info("  Dry Run: %s", dry_run)
    logger.info("=" * 70)

    # Step 1: Ensure dataset in S3
    s3_path = ensure_dataset_in_s3(config)
    logger.info("Dataset ready at: %s", s3_path)

    if dry_run:
        logger.info("")
        logger.info("=" * 70)
        logger.info("DRY RUN — Would submit AWS Batch job")
        logger.info("  Dataset: %s", s3_path)
        logger.info("  Config: %s", experiment_config_name)
        logger.info("=" * 70)
        return

    # Step 2: Submit Batch job
    job_id = _submit_batch_job(config, experiment_config_name)

    # Step 3: Wait for completion and stream logs
    final_status = _wait_for_batch_job(
        job_id=job_id,
        aws_region=config.aws_region,
        poll_interval=30,
        stream_logs=True,
    )

    logger.info("")
    logger.info("=" * 70)
    if final_status == "SUCCEEDED":
        logger.info("AWS BATCH JOB SUCCEEDED")
        logger.info("  Job ID: %s", job_id)
        logger.info("=" * 70)
    else:
        logger.error("AWS BATCH JOB FAILED")
        logger.error("  Job ID: %s", job_id)
        logger.error("=" * 70)
        raise RuntimeError(f"Batch job failed: {job_id}")


def _submit_batch_job(config: PipelineConfig, experiment_config_name: str) -> str:
    """
    Submit a training job to AWS Batch.

    Args:
        config: PipelineConfig instance.
        experiment_config_name: Experiment config name.

    Returns:
        Batch job ID.
    """
    import boto3

    batch_client = boto3.client("batch", region_name=config.aws_region)

    # Determine job queue and definition from environment
    environment = config.environment
    job_queue = os.environ.get("BATCH_JOB_QUEUE", f"tmoe-{environment}-training")
    job_definition = os.environ.get(
        "BATCH_JOB_DEFINITION", f"tmoe-{environment}-training"
    )

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"tmoe-training-{experiment_config_name}-{timestamp}"

    # Container command (runs in container mode)
    command = [
        "--mode",
        "container",
        "--config",
        experiment_config_name,
    ]

    # Environment variables to pass to container
    environment_vars = [
        {"name": "RAW_DATA_BUCKET", "value": config.raw_data_bucket},
        {"name": "AWS_REGION", "value": config.aws_region},
        {"name": "ENVIRONMENT", "value": environment},
        {"name": "DATASET_NAME", "value": config.dataset_name},
    ]

    # Add WandB API key if available
    if wandb_api_key := os.environ.get("WANDB_API_KEY"):
        environment_vars.append({"name": "WANDB_API_KEY", "value": wandb_api_key})
    else:
        environment_vars.append({"name": "WANDB_MODE", "value": "disabled"})

    if hf_token := os.environ.get("HF_TOKEN"):
        environment_vars.append({"name": "HF_TOKEN", "value": hf_token})

    logger.info("Submitting AWS Batch job:")
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
            "environment": environment_vars,
        },
    )

    job_id = response["jobId"]
    logger.info("Job submitted! ID: %s", job_id)
    return job_id


def _wait_for_batch_job(
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

    logger.info("Polling job status (interval: %ds)...", poll_interval)

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

        # Stream logs when job is running or finished
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
    log_stream_name: str | None,
    last_token: str | None,
    aws_region: str,
) -> tuple[str | None, str | None]:
    """Stream CloudWatch log events for a Batch job."""
    try:
        # Resolve log stream name from job container
        if log_stream_name is None:
            container = job.get("container", {})
            log_stream_name = container.get("logStreamName")
            if not log_stream_name:
                return last_token, None

        # Determine log group from environment
        environment = os.environ.get("ENVIRONMENT", "dev")
        log_group = os.environ.get(
            "BATCH_LOG_GROUP",
            f"/aws/batch/tmoe-{environment}/training",
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
