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
    """Run T-MoE training from within an AWS Batch container. Returns (output_dir, metrics)."""
    import subprocess
    from omegaconf import OmegaConf
    from src.project_types import EXPERIMENTS_DIR
    from src.configs.dataset import get_shard_dir

    config_name = experiment_config.get("experiment_name", "experiment")
    config_path = EXPERIMENTS_DIR / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find config at {config_path}. "
            f"Ensure experiment_name matches the YAML filename."
        )

    dataset_key = OmegaConf.select(
        experiment_config, "dataset.dataset_key", default="wikitext-2"
    )
    model_key = OmegaConf.select(
        experiment_config, "model.model_key", default="gpt-neo-125m"
    )
    shard_dir = get_shard_dir(dataset_key, model_key, base="/tmp/tmoe_shards")

    # Output directory inside container
    output_dir = Path("/tmp/tmoe_outputs") / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Step 1: Prepare shards (idempotent — skips if already done)
    # -----------------------------------------------------------------
    if next(shard_dir.glob("train_shard_*.bin"), None) is None:
        prep_cmd = [
            sys.executable,
            "-m",
            "scripts.prepare_data",
            "--config",
            str(config_path),
            "--out-dir",
            str(shard_dir),
        ]
        subprocess.run(prep_cmd, check=True)

    # -----------------------------------------------------------------
    # Step 2: Train
    # -----------------------------------------------------------------
    import torch

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    base_cmd = (
        ["torchrun", "--standalone", f"--nproc_per_node={num_gpus}"]
        if num_gpus > 1
        else [sys.executable]
    )
    train_cmd = base_cmd + [
        "-m",
        "scripts.train",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--shard-dir",
        str(shard_dir),
    ]

    subprocess.run(train_cmd, check=True)

    # Read the final checkpoint metrics if available
    metrics = _read_last_checkpoint_metrics(output_dir)
    return str(output_dir), metrics


def _read_last_checkpoint_metrics(output_dir: Path) -> dict:
    """Read metrics from the most recent checkpoint JSON, if available."""
    import json

    checkpoints = sorted((output_dir / "checkpoints").glob("checkpoint_step_*.json"))
    if not checkpoints:
        return {"loss": float("inf"), "best_loss": float("inf")}
    with open(checkpoints[-1]) as f:
        data = json.load(f)
    metrics = data.get("metrics", {})
    return {
        "loss": metrics.get("loss", float("inf")),
        "best_loss": metrics.get("loss", float("inf")),
    }
