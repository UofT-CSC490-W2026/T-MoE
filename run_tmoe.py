"""
T-MoE Unified Training Entry Point
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("tmoe.run_tmoe")


def main() -> None:
    """
    Exit codes:
        0: Success
        1: Configuration error
        2: Runtime error
        130: Keyboard interrupt
    """
    parser = argparse.ArgumentParser(
        description="T-MoE Unified Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with backend from config.yaml
  python run_tmoe.py --config gptneo_125m_metabolic
  
  # Override backend
  python run_tmoe.py --config gptneo_125m_metabolic --backend modal
  
  # Dry run (check dataset, log actions, no execution)
  python run_tmoe.py --config gptneo_125m_metabolic --dry-run
        """,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Experiment config name from experiments/ directory (without .yaml).",
    )
    parser.add_argument(
        "-b",
        "--backend",
        type=str,
        choices=["aws", "modal"],
        help="Override backend selection (default: from config.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate: check dataset, log actions, but don't run training.",
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("T-MoE Unified Training Pipeline")
    logger.info("  Project Root: %s", PROJECT_ROOT)
    logger.info("  Config      : %s", args.config)
    logger.info("  Dry Run     : %s", args.dry_run)
    logger.info("=" * 70)

    try:
        # Step 1: Load pipeline configuration
        from infra.config.config import load_pipeline_config

        logger.info("\nLoading pipeline configuration...")
        config = load_pipeline_config()

        # Determine backend (CLI flag overrides config.yaml)
        backend = args.backend or config.compute_backend

        logger.info("\nConfiguration loaded:")
        logger.info("  Backend     : %s", backend)
        logger.info("  Dataset     : %s", config.dataset_name)
        logger.info("  S3 Bucket   : %s", config.raw_data_bucket)
        logger.info("  AWS Region  : %s", config.aws_region)
        logger.info("  Environment : %s", config.environment)

        if backend not in ("aws", "modal"):
            raise ValueError(
                f"Invalid backend: {backend}. Must be 'aws' or 'modal'. "
                f"Set in config.yaml (compute.backend) or use --backend flag."
            )

        # Step 2: Route to appropriate backend
        logger.info("\nRouting to %s backend...", backend.upper())

        if backend == "aws":
            from infra.backends.aws_backend import run_aws_training

            run_aws_training(config, args.config, dry_run=args.dry_run)

        elif backend == "modal":
            from infra.backends.modal_backend import run_modal_training

            run_modal_training(config, args.config, dry_run=args.dry_run)

        logger.info("\n%s", "=" * 70)
        logger.info("SUCCESS — Training pipeline completed")
        logger.info("=" * 70)
        sys.exit(0)

    except KeyboardInterrupt:
        logger.warning("\n\nPipeline cancelled by user")
        sys.exit(130)

    except ValueError as exc:
        logger.error("\n\nConfiguration error: %s", exc)
        logger.error("\nQuick fix:")
        logger.error(
            "  1. Check config.yaml has compute.backend set to 'aws' or 'modal'"
        )
        logger.error("  2. Ensure .env file has required variables (see .env.example)")
        logger.error(
            "  3. Run: cd infra/terraform && terraform output env_configuration"
        )
        sys.exit(1)

    except ImportError as exc:
        logger.error("\n\nDependency error: %s", exc)
        logger.error("\nInstall required dependencies:")
        logger.error("  pip install -r requirements.txt")
        if "modal" in str(exc).lower():
            logger.error("  pip install modal  # For Modal backend")
        sys.exit(1)

    except Exception as exc:
        logger.error("\n\nPipeline execution failed: %s", exc, exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
