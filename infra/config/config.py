"""Centralized configuration for the SPAR data ingestion pipeline.

Merges configuration from three sources in priority order:
  1. Environment variables (highest)
  2. config.yaml (data_ingestion, dataset, compute sections)
  3. Hardcoded defaults (lowest)
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_YAML_PATH = PROJECT_ROOT / "config.yaml"

_DEFAULTS = {
    "aws_region": "us-east-1",
    "instance_type": "ml.m5.xlarge",
    "instance_count": 1,
    "max_runtime_seconds": 3600,
    "dataset_name": "wikitext",
    "output_format": "jsonl",
    "raw_data_prefix": "datasets/raw/",
    "environment": "dev",
    "log_level": "INFO",
    "transformers_version": "4.26",
    "pytorch_version": "2.0",
    "python_version": "py310",
    "max_retries": 3,
    "use_sagemaker": False,
    "dataset_config": None,
    "dataset_splits": None,
}


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable, validated pipeline configuration."""

    aws_region: str
    raw_data_bucket: str
    dataset_name: str
    output_format: str
    raw_data_prefix: str
    environment: str
    log_level: str
    max_retries: int
    use_sagemaker: bool
    dataset_config: Optional[str] = None
    dataset_splits: Optional[list] = None

    # SageMaker fields (required only when use_sagemaker=True)
    sagemaker_role_arn: Optional[str] = None
    instance_type: Optional[str] = None
    instance_count: Optional[int] = None
    max_runtime_seconds: Optional[int] = None
    transformers_version: Optional[str] = None
    pytorch_version: Optional[str] = None
    python_version: Optional[str] = None

    compute_backend: str = "aws"  # "aws" | "modal"

    # Modal fields
    modal_gpu: Optional[str] = None
    modal_cpu: Optional[int] = None
    modal_timeout: Optional[int] = None
    modal_volume_name: Optional[str] = None


def _load_yaml_section() -> dict:
    """Return the data_ingestion section from config.yaml, or {}."""
    try:
        from omegaconf import OmegaConf  # type: ignore[import-untyped]
    except ImportError:
        return {}

    if not CONFIG_YAML_PATH.is_file():
        return {}

    try:
        cfg = OmegaConf.load(CONFIG_YAML_PATH)
        di = cfg.get("data_ingestion")
        return OmegaConf.to_container(di, resolve=True) if di else {}  # type: ignore[return-value]
    except Exception:
        logger.warning(
            "Failed to parse config.yaml data_ingestion section", exc_info=True
        )
        return {}


def _load_yaml_dataset_section() -> dict:
    """Return the top-level dataset section from config.yaml, or {}."""
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return {}

    if not CONFIG_YAML_PATH.is_file():
        return {}

    try:
        cfg = OmegaConf.load(CONFIG_YAML_PATH)
        ds = cfg.get("dataset")
        return OmegaConf.to_container(ds, resolve=True) if ds else {}  # type: ignore[return-value]
    except Exception:
        logger.warning("Failed to parse config.yaml dataset section", exc_info=True)
        return {}


def _flatten_yaml(yaml: dict) -> dict:
    """Flatten combined YAML sections into a single config dict."""
    flat: dict = {}

    ds_block = yaml.get("_dataset_block", {})
    custom_name = ds_block.get("custom_dataset_name")

    if custom_name:
        flat["dataset_name"] = custom_name
        flat["dataset_config"] = ds_block.get("custom_dataset_config")
    elif ds_block.get("dataset_key"):
        logger.warning(
            "dataset_key '%s' specified but catalog lookup is not supported; use custom_dataset_name instead",
            ds_block.get("dataset_key"),
        )

    # Legacy override
    if yaml.get("source_dataset"):
        flat["dataset_name"] = yaml["source_dataset"]
        flat["dataset_config"] = yaml.get("dataset_config")

    flat["use_sagemaker"] = yaml.get("use_sagemaker")
    flat["dataset_splits"] = yaml.get("dataset_splits")

    sm = yaml.get("sagemaker", {})
    flat.update(
        {
            "instance_type": sm.get("instance_type"),
            "instance_count": sm.get("instance_count"),
            "max_runtime_seconds": sm.get("max_runtime_seconds"),
            "transformers_version": sm.get("transformers_version"),
            "pytorch_version": sm.get("pytorch_version"),
            "python_version": sm.get("python_version"),
        }
    )

    s3 = yaml.get("s3", {})
    flat["raw_data_bucket"] = s3.get("raw_data_bucket")
    flat["raw_data_prefix"] = s3.get("raw_data_prefix")

    proc = yaml.get("processing", {})
    flat.update(
        {
            "output_format": proc.get("output_format"),
            "max_retries": proc.get("max_retries"),
            "log_level": proc.get("log_level"),
        }
    )

    return {k: v for k, v in flat.items() if v is not None}


def _load_compute_config() -> dict:
    """Load compute backend and Modal settings from config.yaml."""
    try:
        from omegaconf import OmegaConf
    except ImportError:
        return {}

    if not CONFIG_YAML_PATH.is_file():
        return {}

    try:
        cfg = OmegaConf.load(CONFIG_YAML_PATH)
        compute = cfg.get("compute")
        if compute is None:
            return {}

        flat: dict = {"compute_backend": compute.get("backend", "aws")}
        modal = compute.get("modal", {})
        if modal:
            flat.update(
                {
                    "modal_gpu": modal.get("gpu"),
                    "modal_cpu": modal.get("cpu"),
                    "modal_timeout": modal.get("timeout"),
                    "modal_volume_name": modal.get("volume_name"),
                }
            )

        return {k: v for k, v in flat.items() if v is not None}
    except Exception:
        logger.debug("Failed to parse compute config from config.yaml", exc_info=True)
        return {}


def _try_load_dotenv() -> None:
    """Load .env from project root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv  # type: ignore[import-untyped]

        env_path = PROJECT_ROOT / ".env"
        if env_path.is_file():
            load_dotenv(env_path)
            logger.debug("Loaded .env from %s", env_path)
    except ImportError:
        pass


_ENV_MAP = {
    "AWS_REGION": "aws_region",
    "SAGEMAKER_ROLE_ARN": "sagemaker_role_arn",
    "RAW_DATA_BUCKET": "raw_data_bucket",
    "INSTANCE_TYPE": "instance_type",
    "OUTPUT_FORMAT": "output_format",
    "ENVIRONMENT": "environment",
    "LOG_LEVEL": "log_level",
    "USE_SAGEMAKER": "use_sagemaker",
    "DATASET_CONFIG": "dataset_config",
    "COMPUTE_BACKEND": "compute_backend",
    "MODAL_GPU": "modal_gpu",
    "MODAL_CPU": "modal_cpu",
    "MODAL_TIMEOUT": "modal_timeout",
    "MODAL_VOLUME_NAME": "modal_volume_name",
}


def _load_env_overrides() -> dict:
    """Read environment variables and map them to config keys."""
    overrides: dict = {}
    for env_key, config_key in _ENV_MAP.items():
        val = os.environ.get(env_key)
        if not val:
            continue
        if config_key == "use_sagemaker":
            overrides[config_key] = val.lower() in ("true", "1", "yes")
        elif config_key in ("modal_cpu", "modal_timeout"):
            try:
                overrides[config_key] = int(val)
            except ValueError:
                logger.warning("Invalid integer value for %s: %s", env_key, val)
        else:
            overrides[config_key] = val
    return overrides


def _validate(merged: dict) -> PipelineConfig:
    """Validate merged config and return a frozen PipelineConfig."""
    use_sagemaker = merged.get("use_sagemaker", False)
    required = ["aws_region", "raw_data_bucket"]
    if use_sagemaker:
        required.append("sagemaker_role_arn")

    missing = [k for k in required if not merged.get(k)]
    if missing:
        raise ValueError(
            f"Missing required configuration: {missing}. "
            "Set via environment variables, .env file, or config.yaml. "
            "Run `cd infra/terraform && terraform output env_configuration` after `terraform apply`."
        )

    if use_sagemaker:
        role = merged["sagemaker_role_arn"]
        if not role.startswith("arn:aws:iam::"):
            raise ValueError(
                f"SAGEMAKER_ROLE_ARN must start with 'arn:aws:iam::' — got: {role!r}"
            )

    env = merged.get("environment", "dev")
    if env not in ("dev", "staging", "prod"):
        raise ValueError(f"ENVIRONMENT must be dev/staging/prod — got: {env!r}")

    log_level = merged.get("log_level", "INFO").upper()
    if log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        raise ValueError(
            f"LOG_LEVEL must be DEBUG/INFO/WARNING/ERROR/CRITICAL — got: {log_level!r}"
        )
    merged["log_level"] = log_level

    if merged.get("output_format", "jsonl") not in ("jsonl", "parquet", "text"):
        raise ValueError(
            f"OUTPUT_FORMAT must be jsonl/parquet/text — got: {merged.get('output_format')!r}"
        )

    # Coerce types
    use_sm = merged.get("use_sagemaker", False)
    merged["use_sagemaker"] = (
        use_sm.lower() in ("true", "1", "yes")
        if isinstance(use_sm, str)
        else bool(use_sm)
    )
    merged["max_retries"] = int(merged.get("max_retries", 3))

    for field in ("instance_count", "max_runtime_seconds"):
        if use_sagemaker and field in merged:
            try:
                merged[field] = int(merged[field])
            except (ValueError, TypeError):
                pass

    for field in ("modal_cpu", "modal_timeout"):
        if merged.get(field) is not None:
            try:
                merged[field] = int(merged[field])
            except (ValueError, TypeError):
                pass

    backend = merged.get("compute_backend", "aws")
    if backend not in ("aws", "modal"):
        raise ValueError(f"compute_backend must be 'aws' or 'modal' — got: {backend!r}")
    merged["compute_backend"] = backend

    known_fields = {
        "aws_region",
        "raw_data_bucket",
        "dataset_name",
        "output_format",
        "raw_data_prefix",
        "environment",
        "log_level",
        "max_retries",
        "use_sagemaker",
        "dataset_config",
        "dataset_splits",
        "sagemaker_role_arn",
        "instance_type",
        "instance_count",
        "max_runtime_seconds",
        "transformers_version",
        "pytorch_version",
        "python_version",
        "compute_backend",
        "modal_gpu",
        "modal_cpu",
        "modal_timeout",
        "modal_volume_name",
    }
    return PipelineConfig(**{k: v for k, v in merged.items() if k in known_fields})


def load_pipeline_config() -> PipelineConfig:
    """Load, merge, and validate pipeline configuration.

    Priority: env vars > YAML config > defaults.

    Returns:
        Validated, immutable PipelineConfig.

    Raises:
        ValueError: If required configuration is missing or invalid.
    """
    _try_load_dotenv()

    merged: dict = dict(_DEFAULTS)

    yaml_section = _load_yaml_section()
    dataset_block = _load_yaml_dataset_section()
    if dataset_block:
        yaml_section["_dataset_block"] = dataset_block
    merged.update(_flatten_yaml(yaml_section))

    merged.update(_load_compute_config())
    merged.update(_load_env_overrides())

    config = _validate(merged)
    logger.info(
        "Pipeline config loaded: region=%s bucket=%s dataset=%s backend=%s",
        config.aws_region,
        config.raw_data_bucket,
        config.dataset_name,
        config.compute_backend,
    )
    return config
