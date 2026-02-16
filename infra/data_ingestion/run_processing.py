"""
Launch a SageMaker HuggingFace Processing Job.

Runs locally (or in CI/CD) to submit ``processing_script.py`` to a
managed SageMaker HuggingFace container.  Configuration is resolved
via ``infra.config.config.load_pipeline_config()`` which merges
env vars > config.yaml > defaults.

Usage:
    python infra/data_ingestion/run_processing.py
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("tmoe.run_processing")

# ---------------------------------------------------------------------------
# Late import — keeps the top-level import fast and gives a clear error
# when ``sagemaker`` is missing.
# ---------------------------------------------------------------------------
SAGEMAKER_AVAILABLE = True
try:
    import boto3
    import sagemaker
    from sagemaker.huggingface import HuggingFaceProcessor
    from sagemaker.processing import ProcessingOutput
except ImportError:
    SAGEMAKER_AVAILABLE = False


# ============================================================================
# Terraform output loader
# ============================================================================
def load_terraform_outputs(
    terraform_dir: Optional[str] = None,
) -> Dict[str, str]:
    """Run ``terraform output -json`` and return a flat {key: value} dict.

    Returns an empty dict on any failure (Terraform not installed, state
    not initialised, etc.) so that the caller can fall back to env vars.
    """
    tf_dir = Path(terraform_dir) if terraform_dir else PROJECT_ROOT / "infra" / "terraform"
    if not tf_dir.is_dir():
        logger.debug("Terraform directory not found: %s", tf_dir)
        return {}

    state_file = tf_dir / "terraform.tfstate"
    dot_tf = tf_dir / ".terraform"
    if not state_file.exists() and not dot_tf.exists():
        logger.info("Terraform not initialised — skipping terraform outputs")
        return {}

    try:
        result = subprocess.run(
            ["terraform", "output", "-json"],
            cwd=str(tf_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("terraform output failed: %s", result.stderr.strip())
            return {}
    except FileNotFoundError:
        logger.debug("terraform binary not found on PATH")
        return {}
    except subprocess.TimeoutExpired:
        logger.warning("terraform output timed out")
        return {}

    try:
        raw = json.loads(result.stdout)
    except json.JSONDecodeError:
        logger.warning("terraform output returned invalid JSON")
        return {}

    # ``terraform output -json`` wraps each output as {"value": …, "type": …}
    outputs: Dict[str, str] = {}
    for key, obj in raw.items():
        if isinstance(obj, dict) and "value" in obj:
            outputs[key] = str(obj["value"])
        else:
            outputs[key] = str(obj)

    logger.info("Loaded %d terraform outputs", len(outputs))
    return outputs


# ============================================================================
# Merge Terraform outputs into environment (so config.load picks them up)
# ============================================================================
_TF_TO_ENV = {
    "raw_data_bucket_name": "RAW_DATA_BUCKET",
    "sagemaker_execution_role_arn": "SAGEMAKER_ROLE_ARN",
    "aws_region": "AWS_REGION",
}


def apply_terraform_outputs(tf_outputs: Dict[str, str]) -> None:
    """Push Terraform outputs into the process environment (does not
    overwrite values that are already set)."""
    import os

    for tf_key, env_key in _TF_TO_ENV.items():
        val = tf_outputs.get(tf_key)
        if val and not os.environ.get(env_key):
            os.environ[env_key] = val
            logger.info("Set %s from terraform output", env_key)


# ============================================================================
# Processor factory
# ============================================================================
def create_processor(config: Any) -> "HuggingFaceProcessor":
    """Build a ``HuggingFaceProcessor`` from a ``PipelineConfig``.

    Raises:
        ImportError:  if ``sagemaker`` is not installed.
        ValueError:   if the role ARN looks malformed.
    """
    if not SAGEMAKER_AVAILABLE:
        raise ImportError(
            "The 'sagemaker' package is required. Install it with:\n"
            "  pip install -r infra/data_ingestion/requirements.txt"
        )

    role = config.sagemaker_role_arn
    if ":role/" not in role:
        raise ValueError(f"SAGEMAKER_ROLE_ARN looks malformed: {role!r}")

    boto_session = boto3.Session(region_name=config.aws_region)
    sm_session = sagemaker.Session(boto_session=boto_session)

    processor = HuggingFaceProcessor(
        role=role,
        instance_type=config.instance_type,
        instance_count=config.instance_count,
        transformers_version=config.transformers_version,
        pytorch_version=config.pytorch_version,
        py_version=config.python_version,
        sagemaker_session=sm_session,
        base_job_name="tmoe-data-ingestion",
        max_runtime_in_seconds=config.max_runtime_seconds,
    )

    logger.info(
        "HuggingFaceProcessor created: instance=%s role=…%s",
        config.instance_type,
        role[-20:],
    )
    return processor


# ============================================================================
# Job execution
# ============================================================================
def run_processing_job(processor: "HuggingFaceProcessor", config: Any) -> str:
    """Submit and monitor the processing job.  Returns the job ARN."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"tmoe-ingestion-{timestamp}"

    s3_output = f"s3://{config.raw_data_bucket}/{config.raw_data_prefix}{timestamp}/"

    output = ProcessingOutput(
        output_name="raw_data",
        source="/opt/ml/processing/output",
        destination=s3_output,
    )

    env_vars: Dict[str, str] = {
        "DATASET_NAME": config.dataset_name,
        "OUTPUT_FORMAT": config.output_format,
        "LOG_LEVEL": config.log_level,
        "MAX_RETRIES": str(config.max_retries),
    }

    script_path = str(Path(__file__).parent / "processing_script.py")
    source_dir = str(Path(__file__).parent)

    logger.info("Launching SageMaker processing job")
    logger.info("  Job name   : %s", job_name)
    logger.info("  S3 output  : %s", s3_output)
    logger.info("  Dataset    : %s", config.dataset_name)
    logger.info("  Instance   : %s x %d", config.instance_type, config.instance_count)

    start = time.time()

    processor.run(
        code=script_path,
        source_dir=source_dir,
        outputs=[output],
        job_name=job_name,
        wait=True,
        logs=True,
        environment=env_vars,
    )

    elapsed = time.time() - start

    job_arn = ""
    try:
        job_arn = processor.latest_processing_job.describe()["ProcessingJobArn"]
    except Exception:
        pass

    logger.info("Job completed in %.1f s", elapsed)
    logger.info("  S3 output : %s", s3_output)
    logger.info("  Job ARN   : %s", job_arn)
    return job_arn


# ============================================================================
# Main
# ============================================================================
def main() -> None:
    """End-to-end orchestration: config → processor → job → report."""

    # 1. Terraform outputs (best-effort)
    tf_outputs = load_terraform_outputs()
    if tf_outputs:
        apply_terraform_outputs(tf_outputs)

    # 2. Load validated config (env > yaml > defaults)
    from infra.config.config import load_pipeline_config

    config = load_pipeline_config()

    # 3. Create processor
    processor = create_processor(config)

    # 4. Run
    job_arn = run_processing_job(processor, config)

    # 5. Summary
    logger.info("=" * 70)
    logger.info("SUCCESS")
    logger.info("  Job ARN   : %s", job_arn)
    logger.info("  S3 bucket : %s", config.raw_data_bucket)
    logger.info("  Dataset   : %s", config.dataset_name)
    logger.info("=" * 70)


# ============================================================================
# Entry
# ============================================================================
if __name__ == "__main__":
    try:
        logger.info("=" * 70)
        logger.info("T-MoE SageMaker Data Ingestion Pipeline")
        logger.info("  Project Root : %s", PROJECT_ROOT)
        logger.info("  Timestamp    : %s", datetime.now().isoformat())
        logger.info("=" * 70)
        main()
    except KeyboardInterrupt:
        logger.warning("Cancelled by user")
        sys.exit(130)
    except ImportError as exc:
        logger.error(
            "%s\n  Install dependencies:\n  pip install -r infra/data_ingestion/requirements.txt",
            exc,
        )
        sys.exit(1)
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed: %s", exc, exc_info=True)
        sys.exit(1)
