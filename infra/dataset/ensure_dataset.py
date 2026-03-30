"""Unified dataset resolution for SPAR training.

Ensures a dataset exists in S3 before training begins, triggering ingestion if needed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infra.config.config import PipelineConfig

logger = logging.getLogger(__name__)


def _dataset_s3_prefix(config: PipelineConfig) -> str:
    """Build the dataset-specific S3 prefix: {raw_data_prefix}/{safe_dataset_name}/"""
    safe_name = config.dataset_name.replace("/", "_")
    return config.raw_data_prefix.rstrip("/") + "/" + safe_name + "/"


def ensure_dataset_in_s3(config: PipelineConfig) -> str:
    """Return the S3 path to the dataset, running ingestion if it doesn't exist yet."""
    from infra.s3client.client import S3Client

    bucket = config.raw_data_bucket
    prefix = _dataset_s3_prefix(config)
    s3_path = f"s3://{bucket}/{prefix}"

    s3_client = S3Client(region=config.aws_region, max_retries=config.max_retries)

    if s3_client.dataset_exists(bucket, prefix):
        logger.info("Dataset found in S3: %s — skipping ingestion.", s3_path)
        return s3_path

    logger.info("Dataset not found in S3: %s — running ingestion.", s3_path)
    _run_data_ingestion(config)

    if not s3_client.dataset_exists(bucket, prefix):
        raise RuntimeError(f"Ingestion completed but dataset still not found: {s3_path}")

    logger.info("Dataset ingestion complete: %s", s3_path)
    return s3_path


def _run_data_ingestion(config: PipelineConfig) -> None:
    """Download dataset from HuggingFace and upload to S3 via FallbackIngestion."""
    from infra.data_ingestion.fallback_ingestion import FallbackIngestion

    logger.info(
        "Running data ingestion: dataset=%s bucket=%s prefix=%s region=%s",
        config.dataset_name, config.raw_data_bucket, config.raw_data_prefix, config.aws_region,
    )

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
        logger.info("Ingestion complete: %d records uploaded to S3", result.get("total_records", 0))
    except Exception as exc:
        logger.error("Data ingestion failed: %s", exc, exc_info=True)
        raise RuntimeError(f"Failed to ingest dataset to S3: {exc}") from exc


def check_dataset_exists(config: PipelineConfig) -> bool:
    """Check if dataset exists in S3 without triggering ingestion (useful for dry-runs)."""
    from infra.s3client.client import S3Client

    s3_client = S3Client(region=config.aws_region, max_retries=config.max_retries)
    bucket = config.raw_data_bucket
    prefix = _dataset_s3_prefix(config)
    exists = s3_client.dataset_exists(bucket, prefix)
    logger.info("Dataset %s in s3://%s/%s", "exists" if exists else "NOT found", bucket, prefix)
    return exists
