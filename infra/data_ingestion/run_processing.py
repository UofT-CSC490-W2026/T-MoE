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

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

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

    s3_output = (
        f"s3://{config.raw_data_bucket}/{config.raw_data_prefix}"
        f"{config.dataset_name.replace('/', '_')}/{timestamp}/"
    )

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

    # 1. Load validated config (env > yaml > defaults)
    from infra.config.config import load_pipeline_config

    config = load_pipeline_config()

    # 2. Create processor
    processor = create_processor(config)

    # 3. Run
    job_arn = run_processing_job(processor, config)

    # 4. Summary
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
