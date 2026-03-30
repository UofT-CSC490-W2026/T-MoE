"""S3 client for SPAR data operations.

Provides upload, download, list, delete, presigned URL, and dataset existence checks
with retry logic and progress tracking. Authenticates via IAM roles — no hardcoded credentials.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class S3Client:
    """Thread-safe S3 client with retry logic and progress tracking."""

    def __init__(
        self,
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,
        max_retries: int = 3,
    ) -> None:
        self.region = region
        self.endpoint_url = endpoint_url

        boto_config = BotoConfig(
            region_name=region,
            retries={"max_attempts": max_retries, "mode": "adaptive"},
            connect_timeout=10,
            read_timeout=30,
        )

        kwargs: dict[str, Any] = {"config": boto_config}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url

        session = boto3.Session(region_name=region)
        self._client = session.client("s3", **kwargs)
        self._transfer_config = TransferConfig(
            multipart_threshold=100 * 1024 * 1024,
            max_concurrency=10,
            multipart_chunksize=50 * 1024 * 1024,
        )

        try:
            sts = session.client("sts", config=boto_config)
            identity = sts.get_caller_identity()
            logger.info("S3Client initialised: region=%s account=%s", region, identity["Account"])
        except (NoCredentialsError, ClientError) as exc:
            raise RuntimeError(
                "AWS credentials not found. Configure via IAM role, "
                f"environment variables, or `aws configure`. Error: {exc}"
            ) from exc

    def upload_file(
        self,
        local_path: Union[str, Path],
        bucket: str,
        key: str,
        metadata: Optional[Dict[str, str]] = None,
        content_type: Optional[str] = None,
        show_progress: bool = True,
    ) -> bool:
        """Upload a local file to S3 with server-side encryption (AES256)."""
        local_path = Path(local_path)
        if not local_path.is_file():
            logger.error("File not found: %s", local_path)
            return False

        file_size = local_path.stat().st_size
        extra_args: Dict[str, Any] = {"ServerSideEncryption": "AES256"}
        if metadata:
            extra_args["Metadata"] = metadata
        if content_type:
            extra_args["ContentType"] = content_type

        callback = self._progress_callback(file_size, key) if show_progress else None

        try:
            self._client.upload_file(
                Filename=str(local_path),
                Bucket=bucket,
                Key=key,
                ExtraArgs=extra_args,
                Callback=callback,
                Config=self._transfer_config,
            )
            logger.info("Uploaded %s → s3://%s/%s (%s bytes)", local_path.name, bucket, key, file_size)
            return True
        except ClientError as exc:
            logger.error(
                "Upload failed [%s]: s3://%s/%s — %s",
                exc.response["Error"]["Code"], bucket, key, exc.response["Error"]["Message"],
            )
            return False

    def download_file(
        self,
        bucket: str,
        key: str,
        local_path: Union[str, Path],
        show_progress: bool = True,
    ) -> bool:
        """Download an S3 object to the local filesystem, verifying size on completion."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            head = self._client.head_object(Bucket=bucket, Key=key)
            remote_size = head["ContentLength"]
        except ClientError as exc:
            logger.error("Cannot stat s3://%s/%s — %s", bucket, key, exc.response["Error"]["Code"])
            return False

        callback = self._progress_callback(remote_size, key) if show_progress else None

        try:
            self._client.download_file(
                Bucket=bucket, Key=key, Filename=str(local_path),
                Callback=callback, Config=self._transfer_config,
            )
        except ClientError as exc:
            logger.error(
                "Download failed [%s]: s3://%s/%s — %s",
                exc.response["Error"]["Code"], bucket, key, exc.response["Error"]["Message"],
            )
            return False

        local_size = local_path.stat().st_size
        if local_size != remote_size:
            logger.warning("Size mismatch for %s: remote=%d local=%d", key, remote_size, local_size)
            return False

        logger.info("Downloaded s3://%s/%s → %s (%d bytes)", bucket, key, local_path, local_size)
        return True

    def list_objects(self, bucket: str, prefix: str = "", max_keys: int = 1000) -> List[Dict[str, Any]]:
        """List objects under a prefix with automatic pagination."""
        results: List[Dict[str, Any]] = []
        paginator = self._client.get_paginator("list_objects_v2")

        try:
            pages = paginator.paginate(
                Bucket=bucket, Prefix=prefix,
                PaginationConfig={"MaxItems": max_keys},
            )
            for page in pages:
                for obj in page.get("Contents", []):
                    results.append({
                        "Key": obj["Key"],
                        "Size": obj["Size"],
                        "LastModified": obj["LastModified"].isoformat(),
                        "ETag": obj["ETag"],
                    })
        except ClientError as exc:
            logger.error("list_objects failed for s3://%s/%s — %s", bucket, prefix, exc.response["Error"]["Code"])
            return []

        logger.info("Listed %d objects in s3://%s/%s", len(results), bucket, prefix)
        return results

    def delete_objects(self, bucket: str, keys: List[str]) -> Dict[str, Any]:
        """Batch-delete objects in groups of 1000 (S3 API limit)."""
        if not keys:
            return {"deleted": [], "errors": []}

        deleted: List[str] = []
        errors: List[Dict[str, str]] = []

        for i in range(0, len(keys), 1000):
            batch = keys[i : i + 1000]
            delete_request = {"Objects": [{"Key": k} for k in batch], "Quiet": True}
            try:
                resp = self._client.delete_objects(Bucket=bucket, Delete=delete_request)
                errors.extend(resp.get("Errors", []))
                deleted.extend(batch[: len(batch) - len(resp.get("Errors", []))])
            except ClientError as exc:
                logger.error("delete_objects failed: %s", exc.response["Error"]["Code"])
                errors.extend([{"Key": k, "Code": "ClientError", "Message": str(exc)} for k in batch])

        logger.info("Deleted %d objects, %d errors", len(deleted), len(errors))
        return {"deleted": deleted, "errors": errors}

    def check_bucket_exists(self, bucket: str) -> bool:
        """Return True if the bucket exists and is accessible."""
        try:
            self._client.head_bucket(Bucket=bucket)
            logger.debug("Bucket exists: %s", bucket)
            return True
        except ClientError as exc:
            raw_code = exc.response["Error"]["Code"]
            try:
                code = int(raw_code)
            except (ValueError, TypeError):
                code = raw_code

            if code in (404, "404", "NoSuchBucket"):
                logger.info("Bucket does not exist: %s", bucket)
            elif code in (403, "403", "AccessDenied"):
                logger.warning("Bucket exists but access denied: %s", bucket)
            else:
                logger.error("head_bucket error %s for %s", code, bucket)
            return False

    def generate_presigned_url(self, bucket: str, key: str, expiration: int = 3600) -> Optional[str]:
        """Generate a presigned GET URL valid for `expiration` seconds."""
        try:
            url = self._client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=expiration,
            )
            logger.info("Presigned URL generated for s3://%s/%s (expires in %ds)", bucket, key, expiration)
            return url  # type: ignore[return-value]
        except ClientError as exc:
            logger.error("presigned_url failed: %s", exc.response["Error"]["Code"])
            return None

    def dataset_exists(self, bucket: str, prefix: str) -> bool:
        """Return True if at least one .jsonl or .parquet file exists under the prefix."""
        data_extensions = (".jsonl", ".parquet")
        objects = self.list_objects(bucket, prefix, max_keys=20)
        found = any(obj["Key"].endswith(data_extensions) for obj in objects)
        logger.info(
            "%s dataset in s3://%s/%s (%d objects checked)",
            "Found" if found else "No", bucket, prefix, len(objects),
        )
        return found

    @staticmethod
    def _progress_callback(total_bytes: int, label: str):
        """Return a callback that logs upload/download progress at 25% intervals."""
        transferred = {"bytes": 0, "last_pct": 0}

        def _callback(chunk: int) -> None:
            transferred["bytes"] += chunk
            if total_bytes <= 0:
                return
            pct = int(transferred["bytes"] * 100 / total_bytes)
            if pct >= transferred["last_pct"] + 25:
                transferred["last_pct"] = pct - (pct % 25)
                logger.info("Progress %s: %d%%", label, min(pct, 100))

        return _callback
