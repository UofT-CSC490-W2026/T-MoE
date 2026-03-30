from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

from omegaconf import OmegaConf
from src.project_types import EXPERIMENTS_DIR
from src.configs.dataset import get_shard_dir


def execute_training_workflow(
    experiment_config,
    cache_dir: str,
    config_name: str | None = None,
) -> Tuple[str, dict]:
    """Run SPAR training from within an AWS Batch container. Returns (output_dir, metrics).

    Args:
        experiment_config: Loaded OmegaConf config.
        cache_dir: Local directory for shards and outputs.
        config_name: The YAML stem used to load the config (e.g. "qwen2_1.5b_stress_v3-fineweb").
            When provided this is used directly to locate the config file, avoiding the mismatch
            between experiment_name (e.g. "qwen2_1.5b_stress_v3_fineweb") and the filename which
            uses hyphens. Falls back to experiment_name when omitted.
    """
    import subprocess

    # Resolve config path: prefer the explicit config_name (YAML stem with hyphens)
    # over experiment_name (which may use underscores that don't match the filename).
    if config_name:
        config_path = EXPERIMENTS_DIR / f"{config_name}.yaml"
        if not config_path.exists():
            # Caller may have passed a full path
            config_path = Path(config_name)
    else:
        # Fallback: try experiment_name directly
        exp_name = experiment_config.get("experiment_name", "experiment")
        config_path = EXPERIMENTS_DIR / f"{exp_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find config at {config_path}. "
            f"Pass config_name matching the YAML filename stem."
        )

    dataset_key = OmegaConf.select(
        experiment_config, "dataset.dataset_key", default="wikitext-2"
    )
    model_key = OmegaConf.select(
        experiment_config, "model.model_key", default="gpt-neo-125m"
    )
    shard_dir = get_shard_dir(
        dataset_key, model_key, base=str(Path(cache_dir) / "shards")
    )

    exp_name = experiment_config.get("experiment_name", config_path.stem)
    output_dir = Path(cache_dir) / "outputs" / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Forward HF_TOKEN + set HF cache dirs so tokenizer/dataset downloads work
    # inside VPC-isolated Batch containers and are cached across retries.
    hf_cache = str(Path(cache_dir) / "hf_cache")
    subprocess_env = {
        **os.environ,
        "HF_DATASETS_CACHE": hf_cache,
        "HF_HOME": hf_cache,
    }

    # Step 1: Prepare training shards (idempotent — skips if already done)
    if next(shard_dir.glob("train_shard_*.bin"), None) is None:
        prep_cmd = [
            sys.executable,
            "-m",
            "scripts.prepare_data",
            "--config",
            str(config_path),
            "--out-dir",
            str(shard_dir),
            "--cache-dir",
            str(cache_dir),
        ]
        subprocess.run(prep_cmd, check=True, env=subprocess_env)

    # Step 1b: Prepare eval shards (wikitext-103 + pile-val) for perplexity eval.
    # Modal does this in stage_eval_data; AWS must do it here before training so
    # run_post_training_evals can find val_shard_*.bin files.
    _EVAL_DATASETS = ("wikitext-103", "pile-val")
    for eval_dataset_key in _EVAL_DATASETS:
        eval_shard_dir = get_shard_dir(
            eval_dataset_key, model_key, base=str(Path(cache_dir) / "shards")
        )
        eval_shard_dir.mkdir(parents=True, exist_ok=True)
        if next(eval_shard_dir.glob("val_shard_*.bin"), None) is None:
            eval_prep_cmd = [
                sys.executable,
                "-m",
                "scripts.prepare_data",
                "--config",
                str(config_path),
                "--dataset",
                eval_dataset_key,
                "--out-dir",
                str(eval_shard_dir),
                "--cache-dir",
                str(cache_dir),
            ]
            subprocess.run(eval_prep_cmd, check=True, env=subprocess_env)

    # Step 2: Train
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

    # Mirror Modal's TORCHELASTIC_ERROR_FILE so per-rank crash messages are
    # printed on multi-GPU failures instead of a silent CalledProcessError.
    error_file = "/tmp/torchelastic_error.json"
    try:
        subprocess.run(
            train_cmd,
            check=True,
            env={**subprocess_env, "TORCHELASTIC_ERROR_FILE": error_file},
        )
    except subprocess.CalledProcessError:
        import json as _json

        if Path(error_file).exists():
            with open(error_file) as _f:
                for rank, msg in _json.load(_f).get("message", {}).items():
                    print(f"\n--- Rank {rank} ---\n{msg.get('message', '')}")
        raise

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
        "loss": metrics.get("loss", metrics.get("train_loss", float("inf"))),
        "best_loss": metrics.get("loss", metrics.get("train_loss", float("inf"))),
    }
