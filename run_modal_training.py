"""
run_modal_training.py — Modal Cloud Orchestrator for T-MoE

To switch experiments: change CONFIG at the top of this file.
GPU spec and count are read automatically from compute.modal.gpu in that YAML.

Usage:
    modal run run_modal_training.py                         # full pipeline
    modal run run_modal_training.py --skip-data             # train only
    modal run run_modal_training.py --eval-tasks all        # train, then run every eval task
    modal run run_modal_training.py::stage_data             # data prep only
    modal run run_modal_training.py::stage_train            # training only
    modal run run_modal_training.py::stage_eval \
        --task perplexity                                   # eval latest checkpoint
    modal run run_modal_training.py::stage_train \
        --overrides "training.lr=3e-4,training.steps=3000" # hyperparameter sweep
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from modal import App, Image, Secret, Volume
from omegaconf import OmegaConf

# =============================================================================
# CONFIGURATION — change this one line to switch experiments
# =============================================================================

CONFIG = "experiments/gptneo_125m_standard_v2.yaml"

# GPU spec is read from compute.modal.gpu in the active config.
# Must be resolved at import time for Modal's @app.function(gpu=...) decorator.
try:
    _cfg = OmegaConf.load(CONFIG)
    GPU = str(OmegaConf.select(_cfg, "compute.modal.gpu", default="A100:4"))
except Exception:
    GPU = "A100:4"

_N_GPUS = int(GPU.split(":")[1]) if ":" in GPU else 1
SUPPORTED_EVAL_TASKS = ("perplexity", "lm_harness", "efficiency")

# =============================================================================

# ---------------------------------------------------------------------------
# Volume / image / app
# ---------------------------------------------------------------------------
VOLUME_NAME = "tmoe-data"
VOLUME_MOUNT = "/vol"
SHARDS_DIR = f"{VOLUME_MOUNT}/data"
OUTPUTS_DIR = f"{VOLUME_MOUNT}/outputs"
SECRET_NAME = "tmoe-secrets"

volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .env({"PYTHONPATH": "/app"})
    .add_local_dir(
        ".",
        remote_path="/app",
        ignore=[
            ".idea",
            ".git",
            "__pycache__",
            ".pytest_cache",
            "outputs",
            "cache",
            "*.pyc",
            ".venv",
            ".env",
        ],
    )
)

app = App(name="tmoe", image=image, secrets=[Secret.from_name(SECRET_NAME)])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _config_path(config: str) -> str:
    """Resolve config to absolute path inside the container (/app/...)."""
    if config.startswith("experiments/") or config.startswith("/"):
        return f"/app/{config}" if not config.startswith("/") else config
    return f"/app/experiments/{config}"


def _load_cfg(config_path: str, overrides: str):
    cfg = OmegaConf.load(config_path)
    if overrides:
        parts = [o.strip() for o in overrides.split(",") if o.strip()]
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(parts))
    return cfg


def _override_list(overrides: str) -> list[str]:
    if not overrides:
        return []
    return [o.strip() for o in overrides.split(",") if o.strip()]


def _experiment_output_dir(cfg) -> str:
    return f"{OUTPUTS_DIR}/{cfg.experiment_name}"


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.startswith("checkpoint_step_"):
        try:
            return (int(stem.rsplit("_", 1)[-1]), path.name)
        except ValueError:
            pass
    return (10**18, path.name)


def _latest_checkpoint_path(checkpoints_dir: Path) -> Path:
    checkpoint_paths = sorted(
        checkpoints_dir.glob("checkpoint_step_*.pt"),
        key=_checkpoint_sort_key,
    )
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"No checkpoint_step_*.pt files found in '{checkpoints_dir}'"
        )
    return checkpoint_paths[-1]


def _resolve_runtime_path(path: str) -> str:
    if not path:
        return path
    if path.startswith("/"):
        return path
    if path.startswith("outputs/"):
        return f"{VOLUME_MOUNT}/{path}"
    return f"/app/{path}"


def _resolve_eval_tasks(eval_tasks: str) -> list[str]:
    if not eval_tasks.strip():
        return []

    raw_tasks = [task.strip() for task in eval_tasks.split(",") if task.strip()]
    if len(raw_tasks) == 1 and raw_tasks[0] == "all":
        return list(SUPPORTED_EVAL_TASKS)

    invalid = [task for task in raw_tasks if task not in SUPPORTED_EVAL_TASKS]
    if invalid:
        raise ValueError(
            f"Unsupported eval task(s): {', '.join(invalid)}. "
            f"Choose from: {', '.join(SUPPORTED_EVAL_TASKS)} or 'all'."
        )
    return raw_tasks


def _resolve_eval_checkpoint(cfg, checkpoint: str, all_checkpoints: bool) -> str:
    checkpoints_dir = Path(_experiment_output_dir(cfg)) / "checkpoints"
    if checkpoint:
        if checkpoint == "latest":
            return str(_latest_checkpoint_path(checkpoints_dir))
        if checkpoint == "best":
            return str(checkpoints_dir / "best_model.pt")
        return _resolve_runtime_path(checkpoint)
    if all_checkpoints:
        return str(checkpoints_dir)
    return str(_latest_checkpoint_path(checkpoints_dir))


# ---------------------------------------------------------------------------
# Stage 1: Data Preparation (CPU — cheap, run once per dataset)
# ---------------------------------------------------------------------------


@app.function(
    volumes={VOLUME_MOUNT: volume},
    cpu=8,
    memory=32768,
    timeout=18000,  # 5h: ~10 min download + ~30 min tokenization + buffer
)
def stage_data(config: str = CONFIG, force: bool = False):  # noqa: B008
    """
    Tokenize and pack dataset into binary shards on the Modal Volume.
    Idempotent: skips if training shards already exist (--force to redo).
    """
    import glob

    cfg_path = _config_path(config)
    cfg = OmegaConf.load(cfg_path)
    dataset_key = cfg.dataset.dataset_key

    if glob.glob(f"{SHARDS_DIR}/{dataset_key}/**/train_shard_*.bin") and not force:
        n = len(glob.glob(f"{SHARDS_DIR}/{dataset_key}/**/train_shard_*.bin"))
        print(
            f"[stage_data] {n} shard(s) already in {SHARDS_DIR}/{dataset_key}/. Skipping (--force to redo)."
        )
        volume.commit()
        return

    # Compute tokenizer-aware shard dir: /vol/data/<dataset>/<vocab_size>/
    # This lets different tokenizer families share the same volume without collision.
    from src.configs.dataset import get_shard_dir

    out_dir = str(get_shard_dir(dataset_key, cfg.model.model_key, base=SHARDS_DIR))

    print(f"[stage_data] Preparing '{dataset_key}' → {out_dir}")
    # Route HF cache to the Volume so dataset downloads persist across runs.
    hf_cache = f"{VOLUME_MOUNT}/hf_cache"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.prepare_data",
            "--config",
            cfg_path,
            "--out-dir",
            out_dir,
            "--num-proc",
            "8",
        ],
        cwd="/app",
        check=True,
        env={**os.environ, "HF_DATASETS_CACHE": hf_cache, "HF_HOME": hf_cache},
    )
    volume.commit()
    print(f"[stage_data] Done → {out_dir}")


# ---------------------------------------------------------------------------
# Stage 2: Training (GPU)
# ---------------------------------------------------------------------------


@app.function(
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU,  # provisioned from the constant above
    memory=32768,
    timeout=60 * 60 * 12,
    retries=0,
)
def stage_train(config: str = CONFIG, overrides: str = ""):  # noqa: B008
    """
    Train T-MoE. GPU count is always _N_GPUS (derived from GPU at top of file).

    Args:
        config:    Experiment YAML (defaults to CONFIG at top of file).
        overrides: Comma-separated OmegaConf overrides, e.g. "training.lr=3e-4".
    """
    cfg_path = _config_path(config)
    cfg = _load_cfg(cfg_path, overrides)
    out_dir = _experiment_output_dir(cfg)

    # Symlink /app/data/shards → /vol/data so train.py finds shards at
    # data/shards/{dataset_key}/ as expected.
    os.makedirs("/app/data", exist_ok=True)
    local_shards = "/app/data/shards"
    if os.path.lexists(local_shards):
        if os.path.islink(local_shards):
            os.unlink(local_shards)
        else:
            import shutil

            shutil.rmtree(local_shards)
    os.symlink(SHARDS_DIR, local_shards)

    # Use actual GPU count from hardware — avoids stale _N_GPUS from cached YAML in container.
    import torch

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[stage_train] Experiment : {cfg.experiment_name}")
    print(f"[stage_train] GPU        : {gpu_name} × {n_gpus}")
    print(f"[stage_train] Output     : {out_dir}")

    cmd = (
        (
            [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={n_gpus}",
                "-m",
                "scripts.train",
            ]
            if n_gpus > 1
            else [sys.executable, "-m", "scripts.train"]
        )
        + ["--config", cfg_path, "--output-dir", out_dir]
        + _override_list(overrides)
    )

    error_file = "/tmp/torchelastic_error.json"
    try:
        subprocess.run(
            cmd,
            cwd="/app",
            check=True,
            env={**os.environ, "TORCHELASTIC_ERROR_FILE": error_file},
        )
    except subprocess.CalledProcessError:
        if os.path.exists(error_file):
            with open(error_file) as f:
                for rank, msg in json.load(f).get("message", {}).items():
                    print(f"\n--- Rank {rank} ---\n{msg.get('message', '')}")
        raise

    volume.commit()
    print(f"[stage_train] Done → {out_dir}")


# ---------------------------------------------------------------------------
# Stage 3: Post-training evaluation (GPU, single checkpoint or sweep)
# ---------------------------------------------------------------------------


@app.function(
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU,
    memory=32768,
    timeout=60 * 60 * 12,
    retries=1,
)
def stage_eval(
    task: str = "perplexity",
    config: str = CONFIG,  # noqa: B008
    overrides: str = "",
    checkpoint: str = "",
    all_checkpoints: bool = False,
    device: str = "cuda",
    stride: int = 512,
    max_documents: int | None = None,
    batch_size: str = "1",
    limit: float | None = None,
    seq_len: int = 1024,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
    reference_checkpoint: str = "",
    reference_config: str = "",
):
    """
    Run post-training evaluation against the latest saved checkpoint by default.

    Args:
        task:                One of perplexity, lm_harness, efficiency.
        config:              Experiment YAML (defaults to CONFIG at top of file).
        overrides:           Comma-separated OmegaConf overrides.
        checkpoint:          Optional checkpoint path, or 'latest'/'best'. Defaults to latest step checkpoint.
        all_checkpoints:     Sweep every checkpoint_step_*.pt in the experiment's checkpoints dir.
        device:              Eval device, e.g. cuda, cuda:0, or cpu.
        stride:              Sliding-window stride for perplexity evaluation.
        max_documents:       Optional smoke-test cap for perplexity.
        batch_size:          lm-eval batch size.
        limit:               Optional lm-eval example cap.
        seq_len:             Sequence length for efficiency profiling.
        warmup_iters:        Warmup iterations for efficiency profiling.
        benchmark_iters:     Timed iterations for efficiency profiling.
        reference_checkpoint: Optional reference checkpoint for router overhead ratio.
        reference_config:    Optional config for the reference checkpoint.
    """
    if task not in SUPPORTED_EVAL_TASKS:
        raise ValueError(
            f"Unsupported eval task '{task}'. Choose from: {', '.join(SUPPORTED_EVAL_TASKS)}."
        )

    cfg_path = _config_path(config)
    cfg = _load_cfg(cfg_path, overrides)
    checkpoint_path = _resolve_eval_checkpoint(cfg, checkpoint, all_checkpoints)
    output_dir = f"{_experiment_output_dir(cfg)}/eval"

    print(f"[stage_eval] Experiment : {cfg.experiment_name}")
    print(f"[stage_eval] Task       : {task}")
    print(f"[stage_eval] Checkpoint : {checkpoint_path}")
    print(f"[stage_eval] Output     : {output_dir}")

    cmd = [
        sys.executable,
        "-m",
        "scripts.eval",
        "--task",
        task,
        "--checkpoint",
        checkpoint_path,
        "--config",
        cfg_path,
        "--output-dir",
        output_dir,
        "--device",
        device,
        "--stride",
        str(stride),
        "--batch-size",
        str(batch_size),
        "--seq-len",
        str(seq_len),
        "--warmup-iters",
        str(warmup_iters),
        "--benchmark-iters",
        str(benchmark_iters),
    ]
    if all_checkpoints:
        cmd.append("--all-checkpoints")
    if max_documents is not None:
        cmd.extend(["--max-documents", str(max_documents)])
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if reference_checkpoint:
        cmd.extend(
            ["--reference-checkpoint", _resolve_runtime_path(reference_checkpoint)]
        )
    if reference_config:
        cmd.extend(["--reference-config", _config_path(reference_config)])
    cmd.extend(_override_list(overrides))

    subprocess.run(cmd, cwd="/app", check=True)
    volume.commit()
    print(f"[stage_eval] Done → {output_dir}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    config: str = CONFIG,
    skip_data: bool = False,
    overrides: str = "",
    eval_tasks: str = "",
    checkpoint: str = "",
    all_checkpoints: bool = False,
    device: str = "cuda",
    stride: int = 512,
    max_documents: int | None = None,
    batch_size: str = "1",
    limit: float | None = None,
    seq_len: int = 1024,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
    reference_checkpoint: str = "",
    reference_config: str = "",
):  # noqa: B008
    """
    Run Stage 1 (data) then Stage 2 (train), with optional post-training evals.

    Set eval_tasks to a comma-separated list like "perplexity,lm_harness" or "all"
    to chain evals after training completes.
    """
    if not skip_data:
        stage_data.remote(config=config)
    stage_train.remote(config=config, overrides=overrides)
    for task in _resolve_eval_tasks(eval_tasks):
        stage_eval.remote(
            task=task,
            config=config,
            overrides=overrides,
            checkpoint=checkpoint,
            all_checkpoints=all_checkpoints,
            device=device,
            stride=stride,
            max_documents=max_documents,
            batch_size=batch_size,
            limit=limit,
            seq_len=seq_len,
            warmup_iters=warmup_iters,
            benchmark_iters=benchmark_iters,
            reference_checkpoint=reference_checkpoint,
            reference_config=reference_config,
        )
