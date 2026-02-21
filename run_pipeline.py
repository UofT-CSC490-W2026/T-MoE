"""
T-MoE Data Ingestion Pipeline — Consolidated Entry Point.

Routes execution to either SageMaker-based or fallback ingestion based on
configuration. Single command to run data ingestion regardless of mode.

Usage:
    python run_pipeline.py

Configuration:
    Set use_sagemaker in config.yaml or via USE_SAGEMAKER environment variable.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("tmoe.run_pipeline")


def load_configuration() -> Any:
    """
    Load and validate pipeline configuration.

    Returns:
        PipelineConfig instance.

    Raises:
        ValueError: If configuration is invalid or missing required fields.
        ImportError: If required dependencies are missing.
    """
    try:
        from infra.config.config import load_pipeline_config
    except ImportError as exc:
        logger.error("Failed to import config module: %s", exc)
        logger.error("Ensure you're running from the project root directory.")
        raise

    try:
        config = load_pipeline_config()
        logger.info("Configuration loaded successfully")
        logger.info(
            "  Execution mode : %s", "SageMaker" if config.use_sagemaker else "Fallback"
        )
        logger.info("  Dataset        : %s", config.dataset_name)
        logger.info("  S3 bucket      : %s", config.raw_data_bucket)
        logger.info("  AWS region     : %s", config.aws_region)
        return config
    except ValueError as exc:
        logger.error("Configuration validation failed: %s", exc)
        logger.error("")
        logger.error("Quick fix:")
        logger.error(
            "  1. Run: cd infra/terraform && terraform output env_configuration"
        )
        logger.error("  2. Copy the output to a .env file in the project root")
        logger.error("  3. Or set environment variables directly")
        raise


def run_sagemaker_pipeline(config: Any) -> Dict[str, Any]:
    """
    Run the SageMaker-based ingestion pipeline.

    Args:
        config: PipelineConfig instance.

    Returns:
        Summary dict with job ARN and S3 paths.

    Raises:
        ImportError: If sagemaker package is not installed.
        RuntimeError: If SageMaker job fails.
    """
    logger.info("=" * 70)
    logger.info("Running SageMaker-based ingestion pipeline")
    logger.info("=" * 70)

    try:
        # Import SageMaker modules
        from infra.data_ingestion.run_processing import (
            create_processor,
            run_processing_job,
            load_terraform_outputs,
            apply_terraform_outputs,
        )
    except ImportError as exc:
        logger.error("Failed to import SageMaker modules: %s", exc)
        logger.error("")
        logger.error("Install SageMaker dependencies:")
        logger.error("  pip install sagemaker>=2.150.0")
        raise

    # Load Terraform outputs (best-effort)
    tf_outputs = load_terraform_outputs()
    if tf_outputs:
        apply_terraform_outputs(tf_outputs)

    # Create processor and run job
    processor = create_processor(config)
    job_arn = run_processing_job(processor, config)

    return {
        "mode": "sagemaker",
        "job_arn": job_arn,
        "dataset": config.dataset_name,
        "s3_bucket": config.raw_data_bucket,
    }


def run_fallback_pipeline(config: Any) -> Dict[str, Any]:
    """
    Run the fallback (direct S3 upload) ingestion pipeline.

    Args:
        config: PipelineConfig instance.

    Returns:
        Summary dict with S3 paths and statistics.

    Raises:
        RuntimeError: If ingestion fails.
    """
    logger.info("=" * 70)
    logger.info("Running fallback ingestion pipeline (direct S3 upload)")
    logger.info("=" * 70)

    try:
        from infra.data_ingestion.fallback_ingestion import FallbackIngestion
    except ImportError as exc:
        logger.error("Failed to import fallback ingestion module: %s", exc)
        raise

    # Create and run fallback ingestion
    ingestion = FallbackIngestion(
        dataset_name=config.dataset_name,
        s3_bucket=config.raw_data_bucket,
        s3_prefix=config.raw_data_prefix,
        aws_region=config.aws_region,
        dataset_config=config.dataset_config,
        dataset_splits=config.dataset_splits,
        output_format=config.output_format,
        max_retries=config.max_retries,
        log_level=config.log_level,
    )

    result = ingestion.run()
    result["mode"] = "fallback"
    return result


def main() -> None:
    """
    Main entry point — orchestrates configuration loading and pipeline execution.

    Exit codes:
        0: Success
        1: Configuration error
        2: Pipeline execution error
        130: Keyboard interrupt
    """
    logger.info("=" * 70)
    logger.info("T-MoE Data Ingestion Pipeline")
    logger.info("  Project Root : %s", PROJECT_ROOT)
    logger.info("=" * 70)

    try:
        # Step 1: Load configuration
        config = load_configuration()

        # Step 2: Route to appropriate pipeline
        if config.use_sagemaker:
            result = run_sagemaker_pipeline(config)
        else:
            result = run_fallback_pipeline(config)

        # Step 3: Log summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("  Mode    : %s", result["mode"])
        logger.info("  Dataset : %s", result.get("dataset", config.dataset_name))
        if "s3_bucket" in result:
            logger.info("  Bucket  : %s", result["s3_bucket"])
        if "job_arn" in result:
            logger.info("  Job ARN : %s", result["job_arn"])
        if "total_records" in result:
            logger.info("  Records : %d", result["total_records"])
        if "elapsed_seconds" in result:
            logger.info("  Time    : %.1f s", result["elapsed_seconds"])
        if "s3_paths" in result:
            logger.info("  S3 Files: %d uploaded", len(result["s3_paths"]))
        logger.info("=" * 70)

        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("Pipeline cancelled by user")
        sys.exit(130)

    except ValueError as exc:
        logger.error("")
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    except ImportError as exc:
        logger.error("")
        logger.error("Dependency error: %s", exc)
        logger.error("Install required dependencies:")
        logger.error("  pip install -r infra/data_ingestion/requirements.txt")
        sys.exit(1)

    except Exception as exc:
        logger.error("")
        logger.error("Pipeline execution failed: %s", exc, exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
