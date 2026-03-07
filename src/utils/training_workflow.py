"""
src/utils/training_workflow.py — Thin training wrapper for AWS Batch container mode.

AWS Batch's run_aws_training.py (container mode) imports execute_training_workflow
from here. This module bridges the AWS orchestration layer to scripts/train.main().

It is intentionally minimal: configure the output directory argument, then delegate
to the same training script used everywhere else. No duplicated training logic here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple


def execute_training_workflow(
    experiment_config,
    cache_dir: str,
) -> Tuple[str, dict]:
    """
    Run T-MoE training from within an AWS Batch container.

    Reads distributed.num_gpus from the config to decide whether to launch
    with torchrun (multi-GPU) or python (single-GPU). Uses the same
    scripts/train.py entrypoint as Modal and local runs.

    Args:
        experiment_config: OmegaConf config loaded by run_aws_training.py
        cache_dir: Local directory where dataset shards were downloaded

    Returns:
        (output_dir, metrics) — output directory path and final training metrics
    """
    import subprocess
    from omegaconf import OmegaConf

    # Resolve the config file path (re-serialize so scripts/train.py can load it)
    from src.project_types import EXPERIMENTS_DIR

    config_name = experiment_config.get("experiment_name", "experiment")
    config_path = EXPERIMENTS_DIR / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find config at {config_path}. "
            f"Ensure experiment_name matches the YAML filename."
        )

    # Output directory inside container
    output_dir = Path("/tmp/tmoe_outputs") / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the training command
    num_gpus = OmegaConf.select(experiment_config, "distributed.num_gpus", default=1)
    if num_gpus > 1:
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={num_gpus}",
            "-m",
            "scripts.train",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "scripts.train",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]

    subprocess.run(cmd, check=True)

    # Read the final checkpoint metrics if available
    metrics = _read_last_checkpoint_metrics(output_dir)
    return str(output_dir), metrics


def _read_last_checkpoint_metrics(output_dir: Path) -> dict:
    """Read metrics from the most recent checkpoint JSON, if available."""
    import json

    checkpoints = sorted(output_dir.rglob("checkpoint_step_*.json"))
    if not checkpoints:
        return {"loss": float("inf"), "best_loss": float("inf")}
    with open(checkpoints[-1]) as f:
        data = json.load(f)
    metrics = data.get("metrics", {})
    return {
        "loss": metrics.get("loss", float("inf")),
        "best_loss": metrics.get("loss", float("inf")),
    }
