"""SageMaker Processing Script — runs inside the HuggingFace container.

Downloads a HuggingFace dataset, validates splits, writes to /opt/ml/processing/output,
and emits metadata.json. SageMaker syncs that directory to S3 on completion.

Env vars: DATASET_NAME, OUTPUT_BASE_DIR, OUTPUT_FORMAT, LOG_LEVEL, MAX_RETRIES, RETRY_DELAY
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

DATASET_NAME: str = os.environ.get("DATASET_NAME", "EleutherAI/wikitext-2")
OUTPUT_BASE_DIR: str = os.environ.get("OUTPUT_BASE_DIR", "/opt/ml/processing/output")
OUTPUT_FORMAT: str = os.environ.get("OUTPUT_FORMAT", "jsonl")
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
MAX_RETRIES: int = int(os.environ.get("MAX_RETRIES", "3"))
RETRY_DELAY: float = float(os.environ.get("RETRY_DELAY", "5.0"))

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("tmoe.processing")


def validate_environment() -> None:
    """Validate output dir writability, disk space, dataset name, format, and network."""
    logger.info("Validating environment …")

    output_dir = Path(OUTPUT_BASE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    probe = output_dir / ".write_test"
    try:
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        raise EnvironmentError(
            f"Output dir {output_dir} is not writable: {exc}"
        ) from exc

    free_gb = shutil.disk_usage(str(output_dir)).free / (1024**3)
    logger.info("Disk free: %.1f GB", free_gb)
    if free_gb < 1.0:
        raise EnvironmentError(f"Insufficient disk space: {free_gb:.1f} GB free")

    if not DATASET_NAME.strip():
        raise EnvironmentError("DATASET_NAME is empty")

    if OUTPUT_FORMAT not in ("jsonl", "parquet", "text"):
        raise EnvironmentError(
            f"OUTPUT_FORMAT must be jsonl/parquet/text — got {OUTPUT_FORMAT!r}"
        )

    try:
        urllib.request.urlopen("https://huggingface.co", timeout=10)
        logger.info("Network: huggingface.co reachable")
    except Exception:
        logger.warning("Cannot reach huggingface.co — dataset download may fail")

    logger.info("Python %s on %s", sys.version, platform.platform())
    logger.info("Environment validation passed")


def load_huggingface_dataset(dataset_name: str) -> Dict[str, Any]:
    """Download dataset from HuggingFace Hub with exponential-backoff retry."""
    from datasets import load_dataset  # type: ignore[import-untyped]

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "Loading dataset %s (attempt %d/%d)", dataset_name, attempt, MAX_RETRIES
            )
            ds = load_dataset(dataset_name)
            break
        except (ConnectionError, TimeoutError, OSError) as exc:
            last_error = exc
            wait = RETRY_DELAY * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d failed (%s). Retrying in %.1fs …", attempt, exc, wait
            )
            time.sleep(wait)
    else:
        raise RuntimeError(
            f"Failed to load {dataset_name} after {MAX_RETRIES} attempts: {last_error}"
        )

    if ds is None or len(ds) == 0:  # type: ignore[arg-type]
        raise RuntimeError(f"Dataset {dataset_name} is empty or None")

    for name in ds:
        split = ds[name]
        logger.info(
            "  split %-12s — %7d rows, columns=%s", name, len(split), split.column_names
        )

    return dict(ds)  # type: ignore[arg-type]


def validate_split_data(split_name: str, split_data: Any) -> None:
    """Assert a split is non-empty, has a 'text' column, and contains non-blank rows."""
    if split_data is None:
        raise ValueError(f"Split '{split_name}' is None")
    if len(split_data) == 0:
        raise ValueError(f"Split '{split_name}' is empty")

    columns = split_data.column_names
    if "text" not in columns:
        raise ValueError(
            f"Split '{split_name}' is missing 'text' column. Found: {columns}"
        )

    sample = split_data.select(range(min(5, len(split_data))))
    if not any(row.get("text") and row["text"].strip() for row in sample):
        raise ValueError(f"Split '{split_name}' first 5 rows are all empty/None")

    logger.info(
        "Validated split '%s': %d examples, columns=%s",
        split_name,
        len(split_data),
        columns,
    )


_EXT_MAP = {"jsonl": ".jsonl", "parquet": ".parquet", "text": ".txt"}


def write_split_to_disk(
    split_name: str, split_data: Any, output_format: str = "jsonl"
) -> Path:
    """Write one split to disk in the requested format. Returns the output path."""
    if output_format not in _EXT_MAP:
        raise ValueError(f"Unsupported format {output_format!r}")

    output_path = Path(OUTPUT_BASE_DIR) / f"{split_name}{_EXT_MAP[output_format]}"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    empty_records = 0

    if output_format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as fh:
            for row in split_data:
                text = row.get("text")
                if text is None or not text.strip():
                    empty_records += 1
                    continue
                fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                total_records += 1

    elif output_format == "parquet":
        split_data.to_parquet(str(output_path))
        total_records = len(split_data)

    elif output_format == "text":
        with open(output_path, "w", encoding="utf-8") as fh:
            for row in split_data:
                text = row.get("text")
                if text is None or not text.strip():
                    empty_records += 1
                    continue
                fh.write(text.strip() + "\n")
                total_records += 1

    file_size = output_path.stat().st_size
    logger.info(
        "Written split '%s': %d records (%d empty skipped), %.2f MB → %s",
        split_name,
        total_records,
        empty_records,
        file_size / 1024 / 1024,
        output_path,
    )

    if file_size == 0:
        raise IOError(f"Output file is empty: {output_path}")

    return output_path


def write_metadata(
    dataset_name: str,
    splits_info: Dict[str, Dict[str, Any]],
    output_dir: Path,
    processing_time_seconds: float,
) -> Path:
    """Write metadata.json with provenance and statistics."""
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


def main() -> None:
    """Orchestrate the full processing pipeline."""
    start = time.time()
    output_dir = Path(OUTPUT_BASE_DIR)

    validate_environment()

    dataset = load_huggingface_dataset(DATASET_NAME)
    logger.info("Loaded %d splits", len(dataset))

    splits_info: Dict[str, Dict[str, Any]] = {}
    for split_name, split_data in dataset.items():
        logger.info("Processing split: %s", split_name)
        validate_split_data(split_name, split_data)
        out_path = write_split_to_disk(split_name, split_data, OUTPUT_FORMAT)
        splits_info[split_name] = {
            "num_examples": len(split_data),
            "file_path": str(out_path),
            "file_size": out_path.stat().st_size,
        }

    elapsed = time.time() - start
    write_metadata(DATASET_NAME, splits_info, output_dir, elapsed)

    files = list(output_dir.iterdir())
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    logger.info(
        "Output directory: %d files, %.2f MB total",
        len(files),
        total_size / 1024 / 1024,
    )
    logger.info(
        "COMPLETION: %s",
        json.dumps(
            {
                "status": "SUCCESS",
                "dataset": DATASET_NAME,
                "splits": list(splits_info.keys()),
                "total_records": sum(s["num_examples"] for s in splits_info.values()),
                "total_bytes": total_size,
                "elapsed_seconds": round(elapsed, 2),
            }
        ),
    )


if __name__ == "__main__":
    try:
        logger.info(
            "SPAR SageMaker Processing Job — dataset=%s format=%s",
            DATASET_NAME,
            OUTPUT_FORMAT,
        )
        main()
        logger.info("Processing job completed successfully")
    except Exception as exc:
        logger.error("Processing job FAILED: %s", exc, exc_info=True)
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
