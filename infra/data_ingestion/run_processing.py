"""Submit a SageMaker HuggingFace Processing Job for dataset ingestion.

Runs locally or in CI/CD. Configuration is resolved via load_pipeline_config()
which merges env vars > config.yaml > defaults.
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("tmoe.run_processing")

SAGEMAKER_AVAILABLE = True
try:
    import boto3
    import sagemaker
    from sagemaker.huggingface import HuggingFaceProcessor
    from sagemaker.processing import ProcessingOutput
except ImportError:
    SAGEMAKER_AVAILABLE = False


def create_processor(config: Any) -> "HuggingFaceProcessor":
    """Build a HuggingFaceProcessor from a PipelineConfig.

    Raises:
        ImportError: if sagemaker is not installed.
        ValueError: if the role ARN looks malformed.
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


def run_processing_job(processor: "HuggingFaceProcessor", config: Any) -> str:
    """Submit and wait for the processing job. Returns the job ARN."""
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

    logger.info(
        "Launching SageMaker job: %s → %s (%s x %d)",
        job_name,
        s3_output,
        config.instance_type,
        config.instance_count,
    )

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

    logger.info(
        "Job completed in %.1f s — ARN: %s — output: %s", elapsed, job_arn, s3_output
    )
    return job_arn


def main() -> None:
    from infra.config.config import load_pipeline_config

    config = load_pipeline_config()
    processor = create_processor(config)
    job_arn = run_processing_job(processor, config)
    logger.info(
        "SUCCESS — job_arn=%s bucket=%s dataset=%s",
        job_arn,
        config.raw_data_bucket,
        config.dataset_name,
    )


if __name__ == "__main__":
    try:
        logger.info("SPAR SageMaker Data Ingestion Pipeline — root=%s", PROJECT_ROOT)
        main()
    except KeyboardInterrupt:
        logger.warning("Cancelled by user")
        sys.exit(130)
    except ImportError as exc:
        logger.error("%s\n  pip install -r infra/data_ingestion/requirements.txt", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed: %s", exc, exc_info=True)
        sys.exit(1)
