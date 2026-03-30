"""Shared data serialization utilities for SPAR ingestion."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

_EXT_MAP = {"jsonl": ".jsonl", "parquet": ".parquet", "text": ".txt"}


def load_huggingface_dataset(
    dataset_name: str,
    dataset_config: str | None,
    max_retries: int,
    retry_delay: float = 5.0,
) -> Dict[str, Any]:
    """Download a dataset from HuggingFace Hub with exponential-backoff retry."""
    from datasets import load_dataset  # type: ignore[import-untyped]

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Loading dataset %s (attempt %d/%d)", dataset_name, attempt, max_retries)
            ds = load_dataset(dataset_name, dataset_config) if dataset_config else load_dataset(dataset_name)
            break
        except (ConnectionError, TimeoutError, OSError) as exc:
            last_error = exc
            wait = retry_delay * (2 ** (attempt - 1))
            logger.warning("Attempt %d failed (%s). Retrying in %.1fs ...", attempt, exc, wait)
            time.sleep(wait)
    else:
        raise RuntimeError(f"Failed to load {dataset_name} after {max_retries} attempts: {last_error}")

    if ds is None or len(ds) == 0:  # type: ignore[arg-type]
        raise RuntimeError(f"Dataset {dataset_name} is empty or None")

    for name in ds:
        split = ds[name]
        logger.info("  split %-12s — %7d rows, columns=%s", name, len(split), split.column_names)

    return dict(ds)  # type: ignore[arg-type]


def validate_split_data(split_name: str, split_data: Any) -> None:
    """Assert a split is non-empty, has a 'text' column, and contains non-blank rows."""
    if split_data is None:
        raise ValueError(f"Split '{split_name}' is None")
    if len(split_data) == 0:
        raise ValueError(f"Split '{split_name}' is empty")

    columns = split_data.column_names
    if "text" not in columns:
        raise ValueError(f"Split '{split_name}' is missing 'text' column. Found: {columns}")

    sample = split_data.select(range(min(5, len(split_data))))
    if not any(row.get("text") and row["text"].strip() for row in sample):
        raise ValueError(f"Split '{split_name}' first 5 rows are all empty/None")

    logger.info("Validated split '%s': %d examples, columns=%s", split_name, len(split_data), columns)


def write_split_to_disk(
    split_name: str,
    split_data: Any,
    output_dir: Path,
    output_format: str = "jsonl",
) -> Path:
    """Write one split to disk in the requested format. Returns the output path."""
    if output_format not in _EXT_MAP:
        raise ValueError(f"Unsupported format {output_format!r}")

    output_path = output_dir / f"{split_name}{_EXT_MAP[output_format]}"
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
        "Written split '%s': %d records (%d empty skipped), %.2f MB",
        split_name, total_records, empty_records, file_size / 1024 / 1024,
    )

    if file_size == 0:
        raise IOError(f"Output file is empty: {output_path}")

    return output_path
