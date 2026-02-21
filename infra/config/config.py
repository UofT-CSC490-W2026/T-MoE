"""
Centralized configuration for T-MoE data ingestion pipeline.

Loads and validates configuration from three sources (in priority order):
  1. Environment variables (highest priority)
  2. config.yaml data_ingestion section
  3. Hardcoded defaults (lowest priority)

Usage:
    from infra.config.config import load_pipeline_config, PipelineConfig
    config = load_pipeline_config()
    print(config.aws_region, config.raw_data_bucket)
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
TERRAFORM_DIR = PROJECT_ROOT / "infra" / "terraform"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {
    "aws_region": "us-east-1",
    "instance_type": "ml.m5.xlarge",
    "instance_count": 1,
    "max_runtime_seconds": 3600,
    "dataset_name": "EleutherAI/wikitext-2",
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


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
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

    # SageMaker-specific fields (only required when use_sagemaker=True)
    sagemaker_role_arn: Optional[str] = None
    instance_type: Optional[str] = None
    instance_count: Optional[int] = None
    max_runtime_seconds: Optional[int] = None
    transformers_version: Optional[str] = None
    pytorch_version: Optional[str] = None
    python_version: Optional[str] = None

    # Backend selection (NEW)
    compute_backend: str = "aws"  # Options: "aws", "modal"

    # Modal-specific fields (NEW)
    modal_gpu: Optional[str] = None
    modal_cpu: Optional[int] = None
    modal_timeout: Optional[int] = None
    modal_volume_name: Optional[str] = None


# ---------------------------------------------------------------------------
# YAML loader (best-effort)
# ---------------------------------------------------------------------------
def _load_yaml_section() -> dict:
    """Return the data_ingestion section from config.yaml, or {}."""
    try:
        from omegaconf import OmegaConf  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("omegaconf not installed — skipping YAML config")
        return {}

    if not CONFIG_YAML_PATH.is_file():
        logger.debug("config.yaml not found at %s", CONFIG_YAML_PATH)
        return {}

    try:
        cfg = OmegaConf.load(CONFIG_YAML_PATH)
        di = cfg.get("data_ingestion")
        if di is None:
            return {}
        return OmegaConf.to_container(di, resolve=True)  # type: ignore[return-value]
    except Exception:
        logger.warning(
            "Failed to parse config.yaml data_ingestion section", exc_info=True
        )
        return {}


def _flatten_yaml(yaml: dict) -> dict:
    """Flatten the combined YAML sections into a flat dict."""
    flat: dict = {}

    # Resolve Dataset
    ds_block = yaml.get("_dataset_block", {})
    custom_name = ds_block.get("custom_dataset_name")

    if custom_name:
        flat["dataset_name"] = custom_name
        flat["dataset_config"] = ds_block.get("custom_dataset_config")
    elif ds_block.get("dataset_key"):
        dataset_key = ds_block.get("dataset_key")
        try:
            import sys

            # Ensure catalog is importable
            project_root = Path(__file__).resolve().parent.parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))

            from catalog.dataset_catalog import get_dataset_info

            cat_info = get_dataset_info(dataset_key)
            flat["dataset_name"] = cat_info["name"]
            flat["dataset_config"] = cat_info["config"]
        except Exception as exc:
            logger.warning(
                "Failed to load dataset key '%s' from catalog: %s", dataset_key, exc
            )

    # In case data_ingestion still manually defines source_dataset (legacy override)
    if "source_dataset" in yaml and yaml["source_dataset"]:
        flat["dataset_name"] = yaml.get("source_dataset")
        flat["dataset_config"] = yaml.get("dataset_config")

    flat["use_sagemaker"] = yaml.get("use_sagemaker")
    flat["dataset_splits"] = yaml.get("dataset_splits")

    sm = yaml.get("sagemaker", {})
    flat["instance_type"] = sm.get("instance_type")
    flat["instance_count"] = sm.get("instance_count")
    flat["max_runtime_seconds"] = sm.get("max_runtime_seconds")
    flat["transformers_version"] = sm.get("transformers_version")
    flat["pytorch_version"] = sm.get("pytorch_version")
    flat["python_version"] = sm.get("python_version")

    s3 = yaml.get("s3", {})
    flat["raw_data_bucket"] = s3.get("raw_data_bucket")
    flat["raw_data_prefix"] = s3.get("raw_data_prefix")

    proc = yaml.get("processing", {})
    flat["output_format"] = proc.get("output_format")
    flat["max_retries"] = proc.get("max_retries")
    flat["log_level"] = proc.get("log_level")

    # Remove None values so they don't override defaults
    return {k: v for k, v in flat.items() if v is not None}


def _load_compute_config() -> dict:
    """Load compute configuration from config.yaml for backend selection and Modal settings."""
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

        flat: dict = {}

        # Backend selection
        flat["compute_backend"] = compute.get("backend", "aws")

        # Modal-specific configuration
        modal = compute.get("modal", {})
        if modal:
            flat["modal_gpu"] = modal.get("gpu")
            flat["modal_cpu"] = modal.get("cpu")
            flat["modal_timeout"] = modal.get("timeout")
            flat["modal_volume_name"] = modal.get("volume_name")

        return {k: v for k, v in flat.items() if v is not None}
    except Exception:
        logger.debug("Failed to parse compute config from config.yaml", exc_info=True)
        return {}


# ---------------------------------------------------------------------------
# .env loader (best-effort)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Environment variable mapping
# ---------------------------------------------------------------------------
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
    """Read environment variables and map to config keys."""
    overrides: dict = {}
    for env_key, config_key in _ENV_MAP.items():
        val = os.environ.get(env_key)
        if val:
            # Convert boolean strings
            if config_key == "use_sagemaker":
                overrides[config_key] = val.lower() in ("true", "1", "yes")
            # Convert integer strings for Modal config
            elif config_key in ("modal_cpu", "modal_timeout"):
                try:
                    overrides[config_key] = int(val)
                except ValueError:
                    logger.warning("Invalid integer value for %s: %s", env_key, val)
            else:
                overrides[config_key] = val
    return overrides


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate(merged: dict) -> PipelineConfig:
    """Validate merged config and return a frozen PipelineConfig."""
    # Always required fields
    required = ["aws_region", "raw_data_bucket"]

    use_sagemaker = merged.get("use_sagemaker", False)

    # SageMaker-specific required fields
    if use_sagemaker:
        required.append("sagemaker_role_arn")

    missing = [k for k in required if not merged.get(k)]
    if missing:
        raise ValueError(
            f"Missing required configuration: {missing}. "
            "Set via environment variables, .env file, or config.yaml. "
            "Run `cd infra/terraform && terraform output env_configuration` "
            "to get the values after `terraform apply`."
        )

    # Validate SageMaker role ARN if provided
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
    valid_levels = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    if log_level not in valid_levels:
        raise ValueError(
            f"LOG_LEVEL must be one of {valid_levels} — got: {log_level!r}"
        )
    merged["log_level"] = log_level

    fmt = merged.get("output_format", "jsonl")
    if fmt not in ("jsonl", "parquet", "text"):
        raise ValueError(f"OUTPUT_FORMAT must be jsonl/parquet/text — got: {fmt!r}")

    # Coerce boolean field
    use_sagemaker_val = merged.get("use_sagemaker", False)
    if isinstance(use_sagemaker_val, str):
        merged["use_sagemaker"] = use_sagemaker_val.lower() in ("true", "1", "yes")
    else:
        merged["use_sagemaker"] = bool(use_sagemaker_val)

    # Coerce numeric fields
    merged["max_retries"] = int(merged.get("max_retries", 3))
    if use_sagemaker and "instance_count" in merged:
        try:
            merged["instance_count"] = int(merged["instance_count"])
        except (ValueError, TypeError):
            pass
    if use_sagemaker and "max_runtime_seconds" in merged:
        try:
            merged["max_runtime_seconds"] = int(merged["max_runtime_seconds"])
        except (ValueError, TypeError):
            pass

    # Coerce Modal numeric fields
    if "modal_cpu" in merged and merged["modal_cpu"] is not None:
        try:
            merged["modal_cpu"] = int(merged["modal_cpu"])
        except (ValueError, TypeError):
            pass
    if "modal_timeout" in merged and merged["modal_timeout"] is not None:
        try:
            merged["modal_timeout"] = int(merged["modal_timeout"])
        except (ValueError, TypeError):
            pass

    # Validate backend selection
    backend = merged.get("compute_backend", "aws")
    if backend not in ("aws", "modal"):
        raise ValueError(f"compute_backend must be 'aws' or 'modal' — got: {backend!r}")
    merged["compute_backend"] = backend

    # Build config with only the fields that exist in the dataclass
    config_dict = {}
    for field_name in [
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
    ]:
        if field_name in merged:
            config_dict[field_name] = merged[field_name]

    return PipelineConfig(**config_dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_pipeline_config() -> PipelineConfig:
    """
    Load, merge, and validate pipeline configuration.

    Priority: env vars > YAML config > defaults.

    Returns:
        Validated, immutable PipelineConfig instance.

    Raises:
        ValueError: If required configuration is missing or invalid.
    """
    _try_load_dotenv()

    # Start with defaults
    merged: dict = dict(_DEFAULTS)

    # Layer 2a: YAML data_ingestion and dataset overrides
    yaml_section = _load_yaml_sections()
    yaml_flat = _flatten_yaml(yaml_section)
    merged.update(yaml_flat)

    # Layer 2b: YAML compute overrides (for backend selection and Modal config)
    compute_config = _load_compute_config()
    merged.update(compute_config)

    # Layer 3: env var overrides (highest priority)
    env_overrides = _load_env_overrides()
    merged.update(env_overrides)

    config = _validate(merged)
    logger.info(
        "Pipeline config loaded: region=%s bucket=%s dataset=%s backend=%s",
        config.aws_region,
        config.raw_data_bucket,
        config.dataset_name,
        config.compute_backend,
    )
    return config
