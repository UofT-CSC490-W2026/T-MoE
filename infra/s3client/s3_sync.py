"""
S3 sync utilities for uploading experiment output directories.

Recursively uploads a local directory tree to S3, preserving structure.
Uses the existing S3Client for all operations.

Usage:
    from infra.s3client.s3_sync import upload_experiment_dir
    upload_experiment_dir(
        local_dir="/outputs/experiments/run_20260216/",
        bucket="tmoe-dev-raw-data-xxx",
        s3_prefix="experiments/run_20260216/",
        aws_region="us-east-1",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def upload_experiment_dir(
    local_dir: str,
    bucket: str,
    s3_prefix: str,
    aws_region: str = "us-east-1",
    max_retries: int = 3,
) -> Dict[str, List[str]]:
    """
    Upload all files in a local directory tree to S3.

    Preserves the directory structure relative to `local_dir`.

    Args:
        local_dir: Absolute path to the local experiment output directory.
        bucket: Target S3 bucket name.
        s3_prefix: S3 key prefix under which files are uploaded.
        aws_region: AWS region.
        max_retries: Number of upload retries per file.

    Returns:
        Dict with 'uploaded' (list of S3 URIs) and 'failed' (list of local paths).

    Raises:
        FileNotFoundError: If local_dir does not exist.
        RuntimeError: If S3 bucket is inaccessible.
    """
    from infra.s3client.client import S3Client

    local_path = Path(local_dir)
    if not local_path.is_dir():
        raise FileNotFoundError(f"Output directory not found: {local_dir}")

    s3_prefix = s3_prefix.rstrip("/") + "/"

    s3_client = S3Client(region=aws_region, max_retries=max_retries)

    if not s3_client.check_bucket_exists(bucket):
        raise RuntimeError(f"S3 bucket does not exist or is inaccessible: {bucket}")

    uploaded: List[str] = []
    failed: List[str] = []

    # Collect all files recursively
    all_files = sorted(f for f in local_path.rglob("*") if f.is_file())
    logger.info(
        "Uploading %d files from %s → s3://%s/%s",
        len(all_files),
        local_dir,
        bucket,
        s3_prefix,
    )

    for file_path in all_files:
        relative = file_path.relative_to(local_path)
        s3_key = f"{s3_prefix}{relative}"

        success = s3_client.upload_file(
            local_path=file_path,
            bucket=bucket,
            key=s3_key,
            show_progress=False,
        )

        if success:
            uploaded.append(f"s3://{bucket}/{s3_key}")
        else:
            logger.error("Failed to upload: %s", file_path)
            failed.append(str(file_path))

    logger.info(
        "Upload complete: %d succeeded, %d failed",
        len(uploaded),
        len(failed),
    )
    return {"uploaded": uploaded, "failed": failed}


def download_s3_prefix(
    bucket: str,
    s3_prefix: str,
    local_dir: str,
    aws_region: str = "us-east-1",
    max_retries: int = 3,
) -> List[str]:
    """
    Download all objects under an S3 prefix to a local directory.

    Preserves the key structure relative to s3_prefix.

    Args:
        bucket: S3 bucket name.
        s3_prefix: S3 prefix to download from.
        local_dir: Local directory to download into.
        aws_region: AWS region.
        max_retries: Number of download retries per file.

    Returns:
        List of local file paths that were downloaded.

    Raises:
        RuntimeError: If S3 bucket is inaccessible.
    """
    from infra.s3client.client import S3Client

    s3_client = S3Client(region=aws_region, max_retries=max_retries)
    local_path = Path(local_dir)
    local_path.mkdir(parents=True, exist_ok=True)

    objects = s3_client.list_objects(bucket, s3_prefix)
    if not objects:
        logger.warning("No objects found under s3://%s/%s", bucket, s3_prefix)
        return []

    logger.info(
        "Downloading %d objects from s3://%s/%s → %s",
        len(objects),
        bucket,
        s3_prefix,
        local_dir,
    )

    downloaded: List[str] = []
    for obj in objects:
        s3_key = obj["Key"]
        # Compute relative path from the prefix
        relative = s3_key[len(s3_prefix) :].lstrip("/")
        if not relative:
            continue  # Skip the prefix itself if it's a "directory marker"

        dest = local_path / relative
        success = s3_client.download_file(
            bucket=bucket,
            key=s3_key,
            local_path=dest,
            show_progress=False,
        )
        if success:
            downloaded.append(str(dest))
        else:
            logger.error("Failed to download: s3://%s/%s", bucket, s3_key)

    logger.info("Downloaded %d files to %s", len(downloaded), local_dir)
    return downloaded
