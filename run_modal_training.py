"""
run_modal_training.py — Modal Cloud Orchestrator for SPAR

To switch experiments: change CONFIG at the top of this file.
GPU spec and count are read automatically from compute.modal.gpu in that YAML.

Usage:
    modal run run_modal_training.py                         # full pipeline
    modal run run_modal_training.py --skip-data             # train only
    modal run run_modal_training.py --eval-tasks all        # train, then run every eval task
    modal run run_modal_training.py::stage_data             # data prep only
    modal run run_modal_training.py::stage_train            # training only
    modal run run_modal_training.py::stage_eval             # eval all tasks (best checkpoint)
    modal run run_modal_training.py::stage_eval \
        --task perplexity                                   # eval specific task
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

CONFIG = "experiments/qwen2_1.5b_stress_v3-fineweb.yaml"

# GPU spec is read from compute.modal.gpu in the active config.
# Must be resolved at import time for Modal's @app.function(gpu=...) decorator.
try:
    _cfg = OmegaConf.load(CONFIG)
    GPU = str(OmegaConf.select(_cfg, "compute.modal.gpu", default="A100:4"))
    # Eval uses same GPU family as training but with a configurable count (default 4).
    _eval_gpu_base = GPU.split(":")[0]
    _N_EVAL_GPUS = int(GPU.split(":")[1]) if ":" in GPU else 1
    EVAL_GPU = f"{_eval_gpu_base}:{_N_EVAL_GPUS}"
except Exception:  # noqa: BLE001
    GPU = "A100:4"
    EVAL_GPU = "A100:4"
    _N_EVAL_GPUS = 4

_N_GPUS = int(GPU.split(":")[1]) if ":" in GPU else 1
SUPPORTED_EVAL_TASKS = ("perplexity", "lm_harness", "efficiency", "routing_analysis")

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

_FLASH_ATTN_WHEEL = (
    "https://huggingface.co/strangertoolshf/flash_attention_2_wheelhouse/resolve/main/"
    "wheelhouse-flash_attn-2.8.3/linux_x86_64/torch2.8/cu12/abiFALSE/cp311/"
    "flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)

image = (
    Image.from_registry(
        "nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11"
    )
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    .pip_install_from_requirements("requirements.txt")
    .pip_install(_FLASH_ATTN_WHEEL)
    .env(
        {
            "PYTHONPATH": "/app",
            "OMP_NUM_THREADS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
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
    if all_checkpoints:
        return str(checkpoints_dir)
    if not checkpoint or checkpoint == "best":
        best_path = checkpoints_dir / "best_model.pt"
        if best_path.exists():
            return str(best_path)
        # Fall back to latest if best_model.pt was not saved (e.g. no val run)
        return str(_latest_checkpoint_path(checkpoints_dir))
    if checkpoint == "latest":
        return str(_latest_checkpoint_path(checkpoints_dir))
    return _resolve_runtime_path(checkpoint)


def _hf_env(base_env: dict) -> dict:
    """Return base_env with HF_TOKEN injected if available in os.environ."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return {**base_env, "HF_TOKEN": token}
    return base_env


# ---------------------------------------------------------------------------
# Stage 1: Data Preparation (CPU — cheap, run once per dataset)
# ---------------------------------------------------------------------------


@app.function(
    volumes={VOLUME_MOUNT: volume},
    cpu=16,
    memory=49152,
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

    from src.configs.dataset import get_shard_dir

    out_dir = str(get_shard_dir(dataset_key, cfg.model.model_key, base=SHARDS_DIR))

    if glob.glob(f"{out_dir}/train_shard_*.bin") and not force:
        n = len(glob.glob(f"{out_dir}/train_shard_*.bin"))
        print(
            f"[stage_data] {n} shard(s) already in {out_dir}/. Skipping (--force to redo)."
        )
        volume.commit()
        return

    print(f"[stage_data] Preparing '{dataset_key}' → {out_dir}")
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
            "16",
        ],
        cwd="/app",
        check=True,
        env=_hf_env({**os.environ, "HF_DATASETS_CACHE": hf_cache, "HF_HOME": hf_cache}),
    )
    volume.commit()
    print(f"[stage_data] Done → {out_dir}")


# ---------------------------------------------------------------------------
# Stage 1b: Eval Data Preparation (CPU — small datasets, runs fast)
# ---------------------------------------------------------------------------

# Datasets required for shard-based perplexity eval.
# wikitext-103 test split: ~240k tokens (~0.5 MB). pile-val: ~5k docs.
_EVAL_DATASETS = ("wikitext-103", "pile-val")


@app.function(
    volumes={VOLUME_MOUNT: volume},
    cpu=4,
    memory=16384,
    timeout=3600,
)
def stage_eval_data(config: str = CONFIG, force: bool = False):  # noqa: B008
    """
    Tokenize eval datasets (wikitext-103, pile-val) into binary shards.
    Idempotent: skips any dataset whose val shards already exist.
    """
    import glob as _glob

    cfg_path = _config_path(config)
    cfg = OmegaConf.load(cfg_path)
    model_key = cfg.model.model_key
    hf_cache = f"{VOLUME_MOUNT}/hf_cache"

    from src.configs.dataset import get_shard_dir

    for dataset_key in _EVAL_DATASETS:
        out_dir = str(get_shard_dir(dataset_key, model_key, base=SHARDS_DIR))
        existing = _glob.glob(f"{out_dir}/val_shard_*.bin")
        if existing and not force:
            print(
                f"[stage_eval_data] {len(existing)} val shard(s) already exist for "
                f"'{dataset_key}' in {out_dir}. Skipping (--force to redo)."
            )
            continue

        print(f"[stage_eval_data] Preparing '{dataset_key}' → {out_dir}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.prepare_data",
                "--config",
                cfg_path,
                "--dataset",
                dataset_key,
                "--out-dir",
                out_dir,
                "--num-proc",
                "4",
            ],
            cwd="/app",
            check=True,
            env=_hf_env(
                {**os.environ, "HF_DATASETS_CACHE": hf_cache, "HF_HOME": hf_cache}
            ),
        )

    volume.commit()
    print("[stage_eval_data] Done.")


# ---------------------------------------------------------------------------
# Stage 2: Training (GPU)
# ---------------------------------------------------------------------------


@app.function(
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU,
    memory=32768,
    timeout=60 * 60 * 12,
    retries=0,
)
def stage_train(config: str = CONFIG, overrides: str = "", resume: str = ""):  # noqa: B008
    """
    Train SPAR. GPU count is always _N_GPUS (derived from GPU at top of file).

    Args:
        config:    Experiment YAML (defaults to CONFIG at top of file).
        overrides: Comma-separated OmegaConf overrides, e.g. "training.lr=3e-4".
        resume:    Path to checkpoint to resume from, e.g. "outputs/exp/checkpoints/checkpoint_step_13000.pt".
    """
    cfg_path = _config_path(config)
    cfg = _load_cfg(cfg_path, overrides)
    out_dir = _experiment_output_dir(cfg)

    os.makedirs("/app/data", exist_ok=True)
    local_shards = "/app/data/shards"
    if os.path.lexists(local_shards):
        if os.path.islink(local_shards):
            os.unlink(local_shards)
        else:
            import shutil

            shutil.rmtree(local_shards)
    os.symlink(SHARDS_DIR, local_shards)

    import torch

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[stage_train] Experiment : {cfg.experiment_name}")
    print(f"[stage_train] GPU        : {gpu_name} × {n_gpus}")
    print(f"[stage_train] Output     : {out_dir}")
    if resume:
        print(f"[stage_train] Resuming   : {_resolve_runtime_path(resume)}")

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
        + (["--resume", _resolve_runtime_path(resume)] if resume else [])
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
    finally:
        # Commit even on crash so checkpoints written before the failure are
        # persisted to Modal storage and not lost when the container exits.
        volume.commit()

    print(f"[stage_train] Done → {out_dir}")


# ---------------------------------------------------------------------------
# Stage 3: Post-training evaluation (GPU, single checkpoint or sweep)
# ---------------------------------------------------------------------------


@app.function(
    volumes={VOLUME_MOUNT: volume},
    gpu=EVAL_GPU,
    memory=32768,
    timeout=60 * 60 * 12,
    retries=1,
)
def stage_eval(
    task: str = "all",
    config: str = CONFIG,  # noqa: B008
    overrides: str = "",
    checkpoint: str = "",
    all_checkpoints: bool = False,
    device: str = "cuda",
    reference_checkpoint: str = "",
    reference_config: str = "",
):
    """
    Run post-training evaluation against the best saved checkpoint by default.
    All eval hyperparameters (stride, max_documents, batch_size, etc.) are read
    from the `eval:` section of the experiment YAML. Override any of them via
    the `overrides` arg, e.g. overrides="eval.stride=2048,eval.max_documents=null".

    Args:
        task:                Comma-separated eval tasks, or 'all'. Options: perplexity, lm_harness, efficiency, routing_analysis.
        config:              Experiment YAML (defaults to CONFIG at top of file).
        overrides:           Comma-separated OmegaConf overrides.
        checkpoint:          Optional checkpoint path, or 'latest'/'best'. Defaults to best_model.pt.
        all_checkpoints:     Sweep every checkpoint_step_*.pt in the experiment's checkpoints dir.
        device:              Eval device, e.g. cuda, cuda:0, or cpu.
        reference_checkpoint: Optional reference checkpoint for router overhead ratio.
        reference_config:    Optional config for the reference checkpoint.
    """
    tasks = _resolve_eval_tasks(task)

    cfg_path = _config_path(config)
    cfg = _load_cfg(cfg_path, overrides)
    checkpoint_path = _resolve_eval_checkpoint(cfg, checkpoint, all_checkpoints)
    output_dir = f"{_experiment_output_dir(cfg)}/eval"
    hf_cache = f"{VOLUME_MOUNT}/hf_cache"

    # Read eval params from YAML (with fallback defaults matching scripts/eval.py)
    from omegaconf import OmegaConf as _OC

    def _ep(key, default):
        return _OC.select(cfg, f"eval.{key}", default=default)

    stride = _ep("stride", 512)
    max_documents = _ep("max_documents", None)
    max_eval_length = _ep("max_eval_length", 2048)
    batch_size = str(_ep("batch_size", 32))
    lm_batch_size = _ep("lm_harness_batch_size", 1)
    limit = _ep("limit", None)
    seq_len = _ep("seq_len", 1024)
    warmup_iters = _ep("warmup_iters", 10)
    benchmark_iters = _ep("benchmark_iters", 100)
    n_samples = _ep("n_samples", 200)
    max_length = _ep("max_length", 512)
    top_n_tokens = _ep("top_n_tokens", 50)

    print(f"[stage_eval] Experiment : {cfg.experiment_name}")
    print(f"[stage_eval] Tasks      : {', '.join(tasks)}")
    print(f"[stage_eval] Checkpoint : {checkpoint_path}")
    print(f"[stage_eval] Output     : {output_dir}")
    print(
        f"[stage_eval] Eval cfg   : stride={stride}, max_documents={max_documents}, max_eval_length={max_eval_length}, batch_size={batch_size}"
    )

    # Ensure eval shards exist before running perplexity.
    if "perplexity" in tasks:
        import glob as _glob
        from src.configs.dataset import get_shard_dir

        model_key = cfg.model.model_key
        for dataset_key in _EVAL_DATASETS:
            out_dir = str(get_shard_dir(dataset_key, model_key, base=SHARDS_DIR))
            if not _glob.glob(f"{out_dir}/val_shard_*.bin"):
                print(
                    f"[stage_eval] Eval shards missing for '{dataset_key}', preparing now..."
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "scripts.prepare_data",
                        "--config",
                        cfg_path,
                        "--dataset",
                        dataset_key,
                        "--out-dir",
                        out_dir,
                        "--num-proc",
                        "4",
                    ],
                    cwd="/app",
                    check=True,
                    env=_hf_env(
                        {
                            **os.environ,
                            "HF_DATASETS_CACHE": hf_cache,
                            "HF_HOME": hf_cache,
                        }
                    ),
                )
                volume.commit()
                print(f"[stage_eval] Eval shards ready: {out_dir}")

    # ---------------------------------------------------------------------------
    # Perplexity: torchrun subprocess (distributed across all GPUs)
    # ---------------------------------------------------------------------------
    if "perplexity" in tasks:
        print("[stage_eval] --- Running task: perplexity ---")
        use_torchrun = _N_EVAL_GPUS > 1
        cmd = (
            [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={_N_EVAL_GPUS}",
                "-m",
                "scripts.eval",
            ]
            if use_torchrun
            else [sys.executable, "-m", "scripts.eval"]
        )
        cmd += [
            "--task",
            "perplexity",
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
            "--max-eval-length",
            str(max_eval_length),
        ]
        if all_checkpoints:
            cmd.append("--all-checkpoints")
        if max_documents is not None:
            cmd.extend(["--max-documents", str(max_documents)])
        cmd.extend(_override_list(overrides))
        subprocess.run(
            cmd,
            cwd="/app",
            check=True,
            env=_hf_env(
                {
                    **os.environ,
                    "HF_DATASETS_CACHE": hf_cache,
                    "HF_HOME": hf_cache,
                    "SHARD_BASE_DIR": SHARDS_DIR,
                }
            ),
        )

    # ---------------------------------------------------------------------------
    # lm_harness + efficiency: in-process, model loaded once and shared
    # ---------------------------------------------------------------------------
    inprocess_tasks = [
        t for t in tasks if t in ("lm_harness", "efficiency", "routing_analysis")
    ]
    if inprocess_tasks:
        import torch
        from pathlib import Path as _Path
        from evals.loading import load_model_for_eval
        from evals.results_schema import log_results_to_wandb, infer_checkpoint_step

        checkpoint_p = _Path(checkpoint_path)
        if checkpoint_p.is_dir() or all_checkpoints:
            checkpoint_dir = (
                checkpoint_p if checkpoint_p.is_dir() else checkpoint_p.parent
            )
            checkpoint_paths = sorted(
                checkpoint_dir.glob("checkpoint_step_*.pt"),
                key=lambda p: (infer_checkpoint_step(p) or 10**18, p.name),
            )
            if not checkpoint_paths:
                raise FileNotFoundError(
                    f"No checkpoint_step_*.pt in '{checkpoint_dir}'"
                )
        else:
            checkpoint_paths = [checkpoint_p]

        multiple = len(checkpoint_paths) > 1
        autocast_dtype = torch.bfloat16 if device.startswith("cuda") else None

        for ckpt in checkpoint_paths:
            output_base = _Path(output_dir)

            model, checkpoint_info = load_model_for_eval(
                config=cfg,
                checkpoint_path=ckpt,
                device=device,
                dtype=autocast_dtype,
            )

            if "lm_harness" in inprocess_tasks:
                print("[stage_eval] --- Running task: lm_harness ---")
                from evals.lm_harness_runner import run_lm_harness_eval

                out_path = (
                    output_base / "history" / ckpt.stem / "lm_harness.json"
                    if multiple
                    else output_base / "lm_harness.json"
                )
                payload = run_lm_harness_eval(
                    config=cfg,
                    checkpoint_path=ckpt,
                    model=model,
                    checkpoint_info=checkpoint_info,
                    output_path=out_path,
                    device=device,
                    batch_size=lm_batch_size,
                    limit=limit,
                )
                log_results_to_wandb(payload, config=cfg)

            if "efficiency" in inprocess_tasks:
                print("[stage_eval] --- Running task: efficiency ---")
                from evals.efficiency import run_efficiency_eval

                out_path = (
                    output_base / "history" / ckpt.stem / "efficiency.json"
                    if multiple
                    else output_base / "efficiency.json"
                )
                ref_ckpt = (
                    _resolve_runtime_path(reference_checkpoint)
                    if reference_checkpoint
                    else None
                )
                ref_cfg = (
                    _load_cfg(_config_path(reference_config), "")
                    if reference_config
                    else None
                )
                payload = run_efficiency_eval(
                    config=cfg,
                    checkpoint_path=ckpt,
                    model=model,
                    checkpoint_info=checkpoint_info,
                    output_path=out_path,
                    device=device,
                    seq_len=seq_len,
                    warmup_iters=warmup_iters,
                    benchmark_iters=benchmark_iters,
                    reference_checkpoint_path=ref_ckpt,
                    reference_config=ref_cfg,
                )
                log_results_to_wandb(payload, config=cfg)

            if "routing_analysis" in inprocess_tasks:
                print("[stage_eval] --- Running task: routing_analysis ---")
                from evals.routing_analysis import run_routing_analysis

                out_path = (
                    output_base / "history" / ckpt.stem / "routing_analysis.json"
                    if multiple
                    else output_base / "routing_analysis.json"
                )
                payload = run_routing_analysis(
                    config=cfg,
                    checkpoint_path=ckpt,
                    model=model,
                    checkpoint_info=checkpoint_info,
                    output_path=out_path,
                    device=device,
                    n_samples=n_samples,
                    max_length=max_length,
                    top_n_tokens=top_n_tokens,
                )
                log_results_to_wandb(payload, config=cfg)

    volume.commit()
    print(f"[stage_eval] Done → {output_dir}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    config: str = CONFIG,  # noqa: B008
    skip_data: bool = False,
    overrides: str = "",
    eval_tasks: str = "all",
    checkpoint: str = "",
    all_checkpoints: bool = False,
    device: str = "cuda",
    reference_checkpoint: str = "",
    reference_config: str = "",
):
    """
    Run Stage 1 (data) then Stage 2 (train), with optional post-training evals.
    All eval hyperparameters are read from the `eval:` section of the experiment YAML.
    Override any of them via overrides, e.g. --overrides "eval.stride=2048,eval.max_documents=null".

    Set eval_tasks to a comma-separated list like "perplexity,lm_harness" or "all"
    to chain evals after training completes.
    """
    if not skip_data:
        stage_data.remote(config=config)
        stage_eval_data.remote(config=config)
    stage_train.remote(config=config, overrides=overrides)
    for task in _resolve_eval_tasks(eval_tasks):
        stage_eval.remote(
            task=task,
            config=config,
            overrides=overrides,
            checkpoint=checkpoint,
            all_checkpoints=all_checkpoints,
            device=device,
            reference_checkpoint=reference_checkpoint,
            reference_config=reference_config,
        )
