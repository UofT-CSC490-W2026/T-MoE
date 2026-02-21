"""
SageMaker Processing Script — runs inside the HuggingFace container.

Executed by ``HuggingFaceProcessor`` on a managed ML instance.
Downloads a HuggingFace dataset, validates every split, writes raw data
to ``/opt/ml/processing/output``, and emits ``metadata.json``.
SageMaker automatically syncs that directory to S3 when the job finishes.

Environment variables consumed:
    DATASET_NAME   — HuggingFace dataset identifier  (default: KrisMinchev/wikitext-2-raw-v1)
    OUTPUT_BASE_DIR — local output directory          (default: /opt/ml/processing/output)
    OUTPUT_FORMAT  — jsonl | parquet | text            (default: jsonl)
    LOG_LEVEL      — DEBUG | INFO | WARNING | …       (default: INFO)
    MAX_RETRIES    — download retry count              (default: 3)
    RETRY_DELAY    — base retry delay in seconds       (default: 5.0)
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from shared import (
    load_huggingface_dataset,
    validate_split_data,
    write_split_to_disk,
)

# ---------------------------------------------------------------------------
# Configuration (all driven by environment variables)
# ---------------------------------------------------------------------------
DATASET_NAME: str = os.environ.get("DATASET_NAME", "KrisMinchev/wikitext-2-raw-v1")
OUTPUT_BASE_DIR: str = os.environ.get("OUTPUT_BASE_DIR", "/opt/ml/processing/output")
OUTPUT_FORMAT: str = os.environ.get("OUTPUT_FORMAT", "jsonl")
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
MAX_RETRIES: int = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_DELAY: float = float(os.environ.get("RETRY_DELAY", "5.0"))

# ---------------------------------------------------------------------------
# Logging — plain format (CloudWatch auto-captures stdout/stderr)
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("tmoe.processing")


# ============================================================================
# 1. Environment Validation
# ============================================================================
def validate_environment() -> None:
    """Ensure the runtime environment is sane before heavy work begins.

    Raises:
        EnvironmentError: on any validation failure.
    """
    logger.info("Validating environment …")

    # Output directory
    output_dir = Path(OUTPUT_BASE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write test
    probe = output_dir / ".write_test"
    try:
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        raise EnvironmentError(
            f"Output dir {output_dir} is not writable: {exc}"
        ) from exc

    # Disk space (warn if < 1 GB)
    usage = shutil.disk_usage(str(output_dir))
    free_gb = usage.free / (1024**3)
    logger.info("Disk free: %.1f GB", free_gb)
    if free_gb < 1.0:
        raise EnvironmentError(f"Insufficient disk space: {free_gb:.1f} GB free")

    # Dataset name
    if not DATASET_NAME.strip():
        raise EnvironmentError("DATASET_NAME is empty")

    # Output format
    if OUTPUT_FORMAT not in ("jsonl", "parquet", "text"):
        raise EnvironmentError(
            f"OUTPUT_FORMAT must be jsonl/parquet/text — got {OUTPUT_FORMAT!r}"
        )

    # Network connectivity (best-effort)
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=10)
        logger.info("Network: huggingface.co reachable")
    except Exception:
        logger.warning("Cannot reach huggingface.co — dataset download may fail")

    logger.info("Python %s on %s", sys.version, platform.platform())
    logger.info("Environment validation passed")


# ============================================================================
# 5. Metadata
# ============================================================================
def write_metadata(
    dataset_name: str,
    splits_info: Dict[str, Dict[str, Any]],
    output_dir: Path,
    processing_time_seconds: float,
) -> Path:
    """Write ``metadata.json`` with provenance and statistics."""
    metadata = {
        "dataset_name": dataset_name,
        "processing_timestamp": datetime.now(timezone.utc).isoformat(),
        "processing_time_seconds": round(processing_time_seconds, 2),
        "output_format": OUTPUT_FORMAT,
        "splits": splits_info,
        "total_records": sum(s.get("num_examples", 0) for s in splits_info.values()),
        "total_bytes": sum(s.get("file_size", 0) for s in splits_info.values()),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "sagemaker_output_dir": OUTPUT_BASE_DIR,
        },
    }

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    logger.info("Metadata written to %s", meta_path)
    return meta_path


# ============================================================================
# 6. Main
# ============================================================================
def main() -> None:
    """Orchestrate the full processing pipeline."""
    start = time.time()
    output_dir = Path(OUTPUT_BASE_DIR)

    # 1 — environment
    validate_environment()

    # 2 — load
    dataset = load_huggingface_dataset(DATASET_NAME, None, MAX_RETRIES, RETRY_DELAY)
    logger.info("Loaded %d splits", len(dataset))

    # 3+4 — validate & write each split
    splits_info: Dict[str, Dict[str, Any]] = {}
    for split_name, split_data in dataset.items():
        logger.info("Processing split: %s", split_name)
        validate_split_data(split_name, split_data)
        out_path = write_split_to_disk(
            split_name, split_data, output_dir, OUTPUT_FORMAT
        )
        splits_info[split_name] = {
            "num_examples": len(split_data),
            "file_path": str(out_path),
            "file_size": out_path.stat().st_size,
        }

    # 5 — metadata
    elapsed = time.time() - start
    write_metadata(DATASET_NAME, splits_info, output_dir, elapsed)

    # 6 — final validation
    files = list(output_dir.iterdir())
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    logger.info(
        "Output directory: %d files, %.2f MB total",
        len(files),
        total_size / 1024 / 1024,
    )

    completion = {
        "status": "SUCCESS",
        "dataset": DATASET_NAME,
        "splits": list(splits_info.keys()),
        "total_records": sum(s["num_examples"] for s in splits_info.values()),
        "total_bytes": total_size,
        "elapsed_seconds": round(elapsed, 2),
    }
    logger.info("COMPLETION: %s", json.dumps(completion))


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    try:
        logger.info("=" * 70)
        logger.info("T-MoE SageMaker Processing Job")
        logger.info("  Dataset : %s", DATASET_NAME)
        logger.info("  Output  : %s", OUTPUT_BASE_DIR)
        logger.info("  Format  : %s", OUTPUT_FORMAT)
        logger.info("=" * 70)
        main()
        logger.info("Processing job completed successfully")
    except Exception as exc:
        logger.error("Processing job FAILED: %s", exc, exc_info=True)
        # Write error metadata so downstream can detect failure
        try:
            error_meta = Path(OUTPUT_BASE_DIR) / "metadata.json"
            error_meta.parent.mkdir(parents=True, exist_ok=True)
            error_meta.write_text(
                json.dumps(
                    {
                        "status": "FAILED",
                        "error": str(exc),
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                )
            )
        except Exception:
            pass
        sys.exit(1)
