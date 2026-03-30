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
    """Submit and monitor an AWS Batch training job.

    Ensures the dataset exists in S3, submits the job, then streams logs
    until completion. Raises RuntimeError on job failure.
    """
    from infra.dataset.ensure_dataset import ensure_dataset_in_s3

    logger.info("AWS BATCH TRAINING — config=%s dry_run=%s", experiment_config_name, dry_run)

    s3_path = ensure_dataset_in_s3(config)
    logger.info("Dataset ready at: %s", s3_path)

    if dry_run:
        logger.info("DRY RUN — would submit Batch job for dataset=%s config=%s", s3_path, experiment_config_name)
        return

    job_id = _submit_batch_job(config, experiment_config_name)
    final_status = _wait_for_batch_job(job_id=job_id, aws_region=config.aws_region, poll_interval=30, stream_logs=True)

    if final_status == "SUCCEEDED":
        logger.info("AWS BATCH JOB SUCCEEDED — job_id=%s", job_id)
    else:
        logger.error("AWS BATCH JOB FAILED — job_id=%s", job_id)
        raise RuntimeError(f"Batch job failed: {job_id}")


def _submit_batch_job(config: PipelineConfig, experiment_config_name: str) -> str:
    """Submit a training job to AWS Batch and return the job ID."""
    import boto3

    batch_client = boto3.client("batch", region_name=config.aws_region)
    environment = config.environment
    job_queue = os.environ.get("BATCH_JOB_QUEUE", f"tmoe-{environment}-training")
    job_definition = os.environ.get("BATCH_JOB_DEFINITION", f"tmoe-{environment}-training")
    job_name = f"tmoe-training-{experiment_config_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    command = ["--mode", "container", "--config", experiment_config_name]

    environment_vars = [
        {"name": "RAW_DATA_BUCKET", "value": config.raw_data_bucket},
        {"name": "AWS_REGION", "value": config.aws_region},
        {"name": "ENVIRONMENT", "value": environment},
        {"name": "DATASET_NAME", "value": config.dataset_name},
    ]

    if wandb_api_key := os.environ.get("WANDB_API_KEY"):
        environment_vars.append({"name": "WANDB_API_KEY", "value": wandb_api_key})
    else:
        environment_vars.append({"name": "WANDB_MODE", "value": "disabled"})

    if hf_token := os.environ.get("HF_TOKEN"):
        environment_vars.append({"name": "HF_TOKEN", "value": hf_token})

    logger.info("Submitting Batch job: name=%s queue=%s definition=%s", job_name, job_queue, job_definition)

    response = batch_client.submit_job(
        jobName=job_name,
        jobQueue=job_queue,
        jobDefinition=job_definition,
        containerOverrides={"command": command, "environment": environment_vars},
    )

    job_id = response["jobId"]
    logger.info("Job submitted: %s", job_id)
    return job_id


def _wait_for_batch_job(
    job_id: str,
    aws_region: str,
    poll_interval: int = 30,
    stream_logs: bool = True,
) -> str:
    """Poll a Batch job until it reaches a terminal state, optionally streaming logs.

    Returns the final status: 'SUCCEEDED' or 'FAILED'.
    """
    import boto3

    batch_client = boto3.client("batch", region_name=aws_region)
    logs_client = boto3.client("logs", region_name=aws_region) if stream_logs else None

    terminal_states = {"SUCCEEDED", "FAILED"}
    last_log_token = None
    log_stream_name = None

    logger.info("Polling job %s (interval: %ds)…", job_id, poll_interval)

    while True:
        response = batch_client.describe_jobs(jobs=[job_id])
        if not response["jobs"]:
            logger.error("Job %s not found!", job_id)
            return "FAILED"

        job = response["jobs"][0]
        status = job["status"]
        status_reason = job.get("statusReason", "")
        logger.info("  Status: %s %s", status, f"({status_reason})" if status_reason else "")

        if stream_logs and logs_client and status in ("RUNNING", "SUCCEEDED", "FAILED"):
            last_log_token, log_stream_name = _stream_job_logs(
                logs_client, job, log_stream_name, last_log_token, aws_region
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
    """Fetch and print CloudWatch log events for a running Batch job."""
    try:
        if log_stream_name is None:
            log_stream_name = job.get("container", {}).get("logStreamName")
            if not log_stream_name:
                return last_token, None

        environment = os.environ.get("ENVIRONMENT", "dev")
        log_group = os.environ.get("BATCH_LOG_GROUP", f"/aws/batch/tmoe-{environment}/training")

        kwargs = {"logGroupName": log_group, "logStreamName": log_stream_name, "startFromHead": True}
        if last_token:
            kwargs["nextToken"] = last_token

        response = logs_client.get_log_events(**kwargs)
        for event in response.get("events", []):
            msg = event.get("message", "").rstrip()
            if msg:
                print(f"  [BATCH] {msg}")

        return response.get("nextForwardToken"), log_stream_name

    except Exception as e:
        logger.debug("Log streaming error (non-fatal): %s", e)
        return last_token, log_stream_name
