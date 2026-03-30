"""
Unified dataset resolution utility for SPAR training.

Ensures datasets exist in S3 before training begins, providing a consistent
dataset handling strategy across all backends (AWS Batch, Modal).

Usage:
    from infra.dataset.ensure_dataset import ensure_dataset_in_s3

    s3_path = ensure_dataset_in_s3(pipeline_config)
    # s3_path is guaranteed to point to a valid dataset in S3
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infra.config.config import PipelineConfig

logger = logging.getLogger(__name__)


def _dataset_s3_prefix(config: PipelineConfig) -> str:
    """Build the dataset-specific S3 prefix: datasets/raw/{dataset_name}/."""
    safe_name = config.dataset_name.replace("/", "_")
    base = config.raw_data_prefix.rstrip("/") + "/"
    return f"{base}{safe_name}/"


def ensure_dataset_in_s3(config: PipelineConfig) -> str:
    from infra.s3client.client import S3Client

    bucket = config.raw_data_bucket
    prefix = _dataset_s3_prefix(config)
    s3_path = f"s3://{bucket}/{prefix}"

    logger.info("=" * 70)
    logger.info("DATASET RESOLUTION")
    logger.info("  Checking: %s", s3_path)
    logger.info("=" * 70)

    # Initialize S3 client
    s3_client = S3Client(
        region=config.aws_region,
        max_retries=config.max_retries,
    )

    # Check if dataset already exists in S3
    if s3_client.dataset_exists(bucket, prefix):
        logger.info("Dataset found in S3: %s", s3_path)
        logger.info("   Skipping ingestion.")
        return s3_path

    # Dataset not in S3 - need to ingest
    logger.info("Dataset NOT found in S3: %s", s3_path)
    logger.info("   Running data ingestion pipeline...")

    # Run ingestion to download from source and upload to S3
    _run_data_ingestion(config)

    # Verify dataset now exists
    if not s3_client.dataset_exists(bucket, prefix):
        raise RuntimeError(
            f"Dataset ingestion completed but dataset still not found in S3: {s3_path}"
        )

    logger.info("Dataset ingestion complete: %s", s3_path)
    return s3_path


def _run_data_ingestion(config: PipelineConfig) -> None:
    """
    Run the data ingestion pipeline to download and upload dataset to S3.

    Uses the fallback ingestion strategy (direct S3 upload) as it's simpler
    and more reliable than SageMaker processing jobs.

    Args:
        config: PipelineConfig instance.

    Raises:
        RuntimeError: If ingestion fails.
    """
    from infra.data_ingestion.fallback_ingestion import FallbackIngestion

    logger.info("=" * 70)
    logger.info("RUNNING DATA INGESTION")
    logger.info("  Dataset : %s", config.dataset_name)
    logger.info("  Bucket  : %s", config.raw_data_bucket)
    logger.info("  Prefix  : %s", config.raw_data_prefix)
    logger.info("  Region  : %s", config.aws_region)
    logger.info("=" * 70)

    ingestion = FallbackIngestion(
        dataset_name=config.dataset_name,
        s3_bucket=config.raw_data_bucket,
        s3_prefix=config.raw_data_prefix,
        aws_region=config.aws_region,
        dataset_config=config.dataset_config,
        dataset_splits=config.dataset_splits,
        output_format=config.output_format,
        max_retries=config.max_retries,
        log_level=config.log_level,
    )

    try:
        result = ingestion.run()
        logger.info(
            "Ingestion complete: %d records uploaded to S3",
            result.get("total_records", 0),
        )
    except Exception as exc:
        logger.error("Data ingestion failed: %s", exc, exc_info=True)
        raise RuntimeError(f"Failed to ingest dataset to S3: {exc}") from exc


def check_dataset_exists(config: PipelineConfig) -> bool:
    """
    Check if dataset exists in S3 without triggering ingestion.

    Useful for dry-run modes or pre-flight checks.

    Args:
        config: PipelineConfig instance.

    Returns:
        True if dataset exists in S3, False otherwise.
    """
    from infra.s3client.client import S3Client

    s3_client = S3Client(
        region=config.aws_region,
        max_retries=config.max_retries,
    )

    bucket = config.raw_data_bucket
    prefix = _dataset_s3_prefix(config)

    exists = s3_client.dataset_exists(bucket, prefix)

    if exists:
        logger.info("Dataset exists in s3://%s/%s", bucket, prefix)
    else:
        logger.info("Dataset NOT found in s3://%s/%s", bucket, prefix)

    return exists
