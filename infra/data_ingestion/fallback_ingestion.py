"""
Fallback data ingestion module — SageMaker-independent.

Downloads HuggingFace datasets and uploads directly to S3 using boto3.
Designed to work when SageMaker is unavailable.

Usage:
    from infra.data_ingestion.fallback_ingestion import FallbackIngestion

    ingestion = FallbackIngestion(
        dataset_name="KrisMinchev/wikitext-2-raw-v1",
        s3_bucket="my-bucket",
        s3_prefix="datasets/raw/",
        aws_region="us-east-1",
    )
    result = ingestion.run()
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


from infra.data_ingestion.shared import (
    load_huggingface_dataset,
    validate_split_data,
    write_split_to_disk,
)

logger = logging.getLogger(__name__)


class FallbackIngestion:
    """Direct HuggingFace dataset ingestion to S3 without SageMaker."""

    def __init__(
        self,
        dataset_name: str,
        s3_bucket: str,
        s3_prefix: str,
        aws_region: str,
        dataset_config: Optional[str] = None,
        dataset_splits: Optional[List[str]] = None,
        output_format: str = "jsonl",
        max_retries: int = 3,
        log_level: str = "INFO",
    ) -> None:
        """
        Initialize fallback ingestion.

        Args:
            dataset_name: HuggingFace dataset identifier (e.g., "wikitext")
            s3_bucket: Target S3 bucket name
            s3_prefix: S3 key prefix (e.g., "datasets/raw/")
            aws_region: AWS region
            dataset_config: Optional dataset config/subset
            dataset_splits: Optional list of splits to ingest (None = all)
            output_format: Output format (jsonl, parquet, text)
            max_retries: Number of retries for downloads/uploads
            log_level: Logging level
        """
        self.dataset_name = dataset_name
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix.rstrip("/") + "/"
        self.aws_region = aws_region
        self.dataset_config = dataset_config
        self.dataset_splits = dataset_splits
        self.output_format = output_format
        self.max_retries = max_retries

        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_level.upper(), logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    def run(self) -> Dict[str, Any]:
        """
        Execute the fallback ingestion pipeline.

        Returns:
            Summary dict with S3 paths, file sizes, and record counts.

        Raises:
            RuntimeError: On dataset loading or S3 upload failures.
        """
        start_time = time.time()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

        logger.info("=" * 70)
        logger.info("T-MoE Fallback Data Ingestion")
        logger.info("  Dataset    : %s", self.dataset_name)
        logger.info("  Config     : %s", self.dataset_config or "default")
        logger.info("  S3 bucket  : %s", self.s3_bucket)
        logger.info("  S3 prefix  : %s", self.s3_prefix)
        logger.info("  Format     : %s", self.output_format)
        logger.info("=" * 70)

        # Create temp directory for local processing
        with tempfile.TemporaryDirectory(prefix="tmoe_ingestion_") as temp_dir:
            temp_path = Path(temp_dir)
            logger.info("Temporary directory: %s", temp_path)

            # Step 1: Load dataset from HuggingFace Hub
            dataset = load_huggingface_dataset(
                dataset_name=self.dataset_name,
                dataset_config=self.dataset_config,
                max_retries=self.max_retries,
            )

            # Step 2: Filter splits if specified
            splits_to_process = self._get_splits_to_process(dataset)

            # Step 3: Process and write each split to temp directory
            splits_info: Dict[str, Dict[str, Any]] = {}
            for split_name in splits_to_process:
                split_data = dataset[split_name]
                logger.info(
                    "Processing split: %s (%d examples)", split_name, len(split_data)
                )

                validate_split_data(split_name, split_data)
                local_file = write_split_to_disk(
                    split_name=split_name,
                    split_data=split_data,
                    output_dir=temp_path,
                    output_format=self.output_format,
                )
                splits_info[split_name] = {
                    "num_examples": len(split_data),
                    "local_path": str(local_file),
                    "file_size": local_file.stat().st_size,
                }

            # Step 4: Write metadata
            metadata = self._create_metadata(splits_info, time.time() - start_time)
            metadata_file = temp_path / "metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            logger.info("Metadata written to %s", metadata_file)

            # Step 5: Upload all files to S3
            s3_paths = self._upload_to_s3(temp_path, timestamp)

            # Step 6: Return summary
            elapsed = time.time() - start_time
            summary = {
                "status": "SUCCESS",
                "dataset": self.dataset_name,
                "splits": list(splits_info.keys()),
                "total_records": sum(s["num_examples"] for s in splits_info.values()),
                "total_bytes": sum(s["file_size"] for s in splits_info.values()),
                "elapsed_seconds": round(elapsed, 2),
                "s3_paths": s3_paths,
            }

            logger.info("=" * 70)
            logger.info("SUCCESS")
            logger.info("  Total records : %d", summary["total_records"])
            logger.info(
                "  Total size    : %.2f MB", summary["total_bytes"] / 1024 / 1024
            )
            logger.info("  Elapsed time  : %.1f s", elapsed)
            logger.info(
                "  S3 location   : s3://%s/%s%s/",
                self.s3_bucket,
                self.s3_prefix,
                timestamp,
            )
            logger.info("=" * 70)

            return summary

    def _get_splits_to_process(self, dataset: Dict[str, Any]) -> List[str]:
        """Determine which splits to process."""
        available_splits = list(dataset.keys())

        if self.dataset_splits:
            # Validate requested splits exist
            missing = set(self.dataset_splits) - set(available_splits)
            if missing:
                raise ValueError(
                    f"Requested splits {missing} not found in dataset. "
                    f"Available: {available_splits}"
                )
            return self.dataset_splits

        return available_splits

    def _create_metadata(
        self,
        splits_info: Dict[str, Dict[str, Any]],
        processing_time: float,
    ) -> Dict[str, Any]:
        """Create metadata dict matching SageMaker output format."""
        return {
            "dataset_name": self.dataset_name,
            "dataset_config": self.dataset_config,
            "processing_timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_seconds": round(processing_time, 2),
            "output_format": self.output_format,
            "splits": {
                name: {
                    "num_examples": info["num_examples"],
                    "file_size": info["file_size"],
                }
                for name, info in splits_info.items()
            },
            "total_records": sum(s["num_examples"] for s in splits_info.values()),
            "total_bytes": sum(s["file_size"] for s in splits_info.values()),
            "ingestion_mode": "fallback",
        }

    def _upload_to_s3(self, local_dir: Path, timestamp: str) -> Dict[str, str]:
        """Upload all files in local_dir to S3 with retry logic."""
        # Import S3Client here to avoid circular imports
        import sys
        from pathlib import Path as PathLib

        # Add project root to path
        project_root = PathLib(__file__).resolve().parent.parent.parent
        sys.path.insert(0, str(project_root))

        from infra.s3client.client import S3Client

        s3_client = S3Client(region=self.aws_region, max_retries=self.max_retries)

        # Check bucket exists
        if not s3_client.check_bucket_exists(self.s3_bucket):
            raise RuntimeError(
                f"S3 bucket does not exist or is not accessible: {self.s3_bucket}"
            )

        s3_paths: Dict[str, str] = {}
        files_to_upload = list(local_dir.iterdir())

        logger.info("Uploading %d files to S3…", len(files_to_upload))

        for local_file in files_to_upload:
            if not local_file.is_file():
                continue

            s3_key = f"{self.s3_prefix}{timestamp}/{local_file.name}"

            # Check if file already exists (idempotency)
            existing_objects = s3_client.list_objects(
                bucket=self.s3_bucket,
                prefix=s3_key,
                max_keys=1,
            )

            if existing_objects:
                logger.warning("File already exists in S3, skipping: %s", s3_key)
                s3_paths[local_file.name] = f"s3://{self.s3_bucket}/{s3_key}"
                continue

            # Upload file
            success = s3_client.upload_file(
                local_path=local_file,
                bucket=self.s3_bucket,
                key=s3_key,
                show_progress=True,
            )

            if not success:
                raise RuntimeError(f"Failed to upload {local_file.name} to S3")

            s3_paths[local_file.name] = f"s3://{self.s3_bucket}/{s3_key}"

        logger.info("Successfully uploaded %d files to S3", len(s3_paths))
        return s3_paths
