"""
run_modal_training.py — Modal Cloud Orchestrator for T-MoE

Usage (individual stages):
    # Stage 1: Tokenize & pack dataset → Modal Volume (runs on cheap CPU)
    modal run run_modal_training.py::stage_data --config gptneo_125m_metabolic.yaml

    # Stage 2: Train on GPU, reads shards directly from Volume
    modal run run_modal_training.py::stage_train --config gptneo_125m_metabolic.yaml

    # With CLI overrides (same as local):
    modal run run_modal_training.py::stage_train --config gptneo_125m_metabolic.yaml \\
        --overrides "training.lr=1e-4" "training.batch_size=8"

Prerequisites:
    1. modal setup  (first time only)
    2. Create a Modal Workspace Secret named "tmoe-secrets" with:
       - HF_TOKEN            = your HuggingFace access token
       - WANDB_API_KEY        = your WandB API key
       - WANDB_PROJECT        = your WandB project (e.g. tmoe)
       - WANDB_ENTITY         = your WandB entity / team  (optional)
       - AWS_ACCESS_KEY_ID    = your AWS key  (only needed if using S3)
       - AWS_SECRET_ACCESS_KEY = your AWS secret  (only needed if using S3)
"""

from __future__ import annotations

import os
import subprocess

from modal import App, Image, Volume, Secret
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# GPU Configuration
# Read GPU type AND count from the config at module load time so Modal's
# @app.function(gpu=...) decorator picks them up before any function runs.
#
# Modal uses "A100:4" notation to provision 4× A100s for a single function.
# The default config is modal_test.yaml; change this to your active experiment
# if you're running a different config as the primary Modal job.
#
# To override without editing this file, set TMOE_GPU env var:
#   TMOE_GPU="H100:2" modal run run_modal_training.py::stage_train --config ...
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = "experiments/modal_test.yaml"
try:
    _cfg = OmegaConf.load(_DEFAULT_CONFIG)
    _gpu_type = OmegaConf.select(_cfg, "compute.modal.gpu", default="A10G")
    _num_gpus = OmegaConf.select(_cfg, "distributed.num_gpus", default=1)
    # Modal format: "A100:4" provisions 4 GPUs; "A10G" provisions 1.
    GPU_TRAIN = f"{_gpu_type}:{_num_gpus}" if _num_gpus > 1 else _gpu_type
except Exception:
    GPU_TRAIN = os.environ.get("TMOE_GPU", "A10G")  # fallback

# Volume name — shared between all stages so shards persist across runs.
VOLUME_NAME = "tmoe-data"

# Where the volume is mounted inside the container.
VOLUME_MOUNT = "/vol"

# Subdirectories inside the volume.
SHARDS_DIR = f"{VOLUME_MOUNT}/data"  # pre-tokenized .bin shards per dataset
OUTPUTS_DIR = f"{VOLUME_MOUNT}/outputs"  # checkpoints and logs

# Local path where experiment YAML files live in the mounted repo image.
EXPERIMENTS_DIR = "/app/experiments"

# Modal Secret name — create this once in the Modal web dashboard.
SECRET_NAME = "tmoe-secrets"

# ---------------------------------------------------------------------------
# Modal App Setup
# ---------------------------------------------------------------------------

# Shared volume: persistent across stages and container restarts.
volume = Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Directories/patterns to exclude from the Modal image build.
# .idea/ is critical — JetBrains IDEs mutate workspace.xml continuously,
# which triggers Modal's "was modified during build process" error.
_MODAL_IGNORE = [
    ".idea",
    ".git",
    "__pycache__",
    ".pytest_cache",
    "outputs",
    "cache",
    "*.pyc",
    ".venv",
    ".env",
]

# Container image: install requirements, then map the rest of the code.
image = (
    Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .env({"PYTHONPATH": "/app"})
    .add_local_dir(".", remote_path="/app", ignore=_MODAL_IGNORE)
)

app = App(
    name="tmoe",
    image=image,
    secrets=[Secret.from_name(SECRET_NAME)],
)


# ---------------------------------------------------------------------------
# Stage 1: Data Preparation (CPU — cheap)
# ---------------------------------------------------------------------------


@app.function(
    volumes={VOLUME_MOUNT: volume},
    cpu=4,
    memory=8192,
    timeout=60 * 60,  # 1 hour max for large datasets
)
def stage_data(config: str = "gptneo_125m_metabolic.yaml", overrides: str = ""):
    """
    Stage 1: Download dataset from HuggingFace, tokenize, and pack into
    binary shards. Saves shards to the persistent Modal Volume.

    This runs on a cheap CPU container — no GPU/hour cost here.
    The data only needs to be prepared ONCE per dataset.

    Args:
        config:    Filename of the experiment YAML (e.g. gptneo_125m_metabolic.yaml)
        overrides: Optional list of OmegaConf overrides e.g. ["dataset.dataset_key=c4"]
    """
    import sys

    sys.path.insert(0, "/app")

    from omegaconf import OmegaConf
    from pathlib import Path

    # Load config
    config_path = f"/app/experiments/{config}"
    cfg = OmegaConf.load(config_path)
    if overrides:
        override_list = [o.strip() for o in overrides.split(",")]
        override_cfg = OmegaConf.from_dotlist(override_list)
        cfg = OmegaConf.merge(cfg, override_cfg)

    dataset_key = cfg.dataset.dataset_key
    out_dir = Path(SHARDS_DIR) / dataset_key

    # Check if shards already exist (idempotent — skip if already done)
    existing = list(out_dir.glob("train_shard_*.bin")) if out_dir.exists() else []
    if existing:
        print(f"Shards already exist in {out_dir} ({len(existing)} found). Skipping.")
        volume.commit()
        return

    print(f"Preparing dataset '{dataset_key}' → {out_dir}")

    # Run prepare_data.py inside the container
    cmd = [
        sys.executable,
        "-m",
        "scripts.prepare_data",
        "--config",
        config_path,
        "--out-dir",
        str(out_dir),
    ]
    if overrides:
        override_list = [o.strip() for o in overrides.split(",")]
        cmd += override_list  # Extra CLI overrides pass through

    subprocess.run(cmd, cwd="/app", check=True, capture_output=False)

    # Commit to volume so the next stage sees the shards
    volume.commit()
    print(f"Stage 1 complete. Shards saved to: {out_dir}")


# ---------------------------------------------------------------------------
# Stage 2: Training (GPU)
# ---------------------------------------------------------------------------


@app.function(
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_TRAIN,
    memory=32768,
    timeout=60 * 60 * 12,  # 12 hours max run time
)
def stage_train(config: str = "gptneo_125m_metabolic.yaml", overrides: str = ""):
    """
    Stage 2: Train the T-MoE model. Reads shards directly from the persistent
    Modal Volume — no S3 downloads, no tokenization lag. Saves ckpt.pt back
    to the Volume.

    Args:
        config:    Filename of the experiment YAML (e.g. gptneo_125m_metabolic.yaml)
        overrides: Optional OmegaConf overrides e.g. ["training.lr=1e-4"]
    """
    import sys

    sys.path.insert(0, "/app")

    config_path = f"/app/experiments/{config}"

    # Symlink the volume shard dir into the expected local path
    from pathlib import Path

    local_data_dir = Path("/app/data")
    volume_data_dir = Path(SHARDS_DIR)

    local_data_dir.parent.mkdir(parents=True, exist_ok=True)
    if local_data_dir.exists() or local_data_dir.is_symlink():
        import shutil

        if local_data_dir.is_dir() and not local_data_dir.is_symlink():
            shutil.rmtree(local_data_dir)
        else:
            local_data_dir.unlink()

    local_data_dir.symlink_to(volume_data_dir)
    print(f"Symlinked {local_data_dir} → {volume_data_dir}")

    # Output dir inside the volume
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    if overrides:
        override_list = [o.strip() for o in overrides.split(",")]
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(override_list))

    out_dir = Path(OUTPUTS_DIR) / cfg.experiment_name

    print(f"\n{'=' * 60}")
    print(f"Experiment : {cfg.experiment_name}")
    print(f"Config     : {config}")
    if overrides:
        print(f"Overrides  : {overrides}")
    print(f"Output Dir : {out_dir}")
    print(f"{'=' * 60}\n")

    # Build and run train command
    # Read num_gpus from config to decide between python and torchrun.
    num_gpus = OmegaConf.select(cfg, "distributed.num_gpus", default=1)

    if num_gpus > 1:
        cmd = [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={num_gpus}",
            "-m",
            "scripts.train",
            "--config",
            config_path,
            "--output-dir",
            str(out_dir),
        ]
    else:
        cmd = [
            sys.executable,
            "-m",
            "scripts.train",
            "--config",
            config_path,
            "--output-dir",
            str(out_dir),
        ]

    if overrides:
        override_list = [o.strip() for o in overrides.split(",")]
        cmd += override_list

    import json
    import os

    # Set error file so we can read the actual traceback if torchrun fails
    error_file = "/tmp/torchelastic_error.json"
    env = os.environ.copy()
    env["TORCHELASTIC_ERROR_FILE"] = error_file

    try:
        subprocess.run(cmd, cwd="/app", check=True, env=env)
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print("TORCHRUN FAILED. Attempting to extract root cause traceback...")
        if os.path.exists(error_file):
            try:
                with open(error_file, "r") as f:
                    err_data = json.load(f)
                    print("\nROOT CAUSE TRACEBACK:")
                    for idx, err in err_data.get("message", {}).items():
                        print(f"--- Rank {idx} ---")
                        print(err.get("message", "No message"))
                        print("---" * 20)
            except Exception as read_e:
                print(f"Could not read error file: {read_e}")
        else:
            print("Error file not found.")
        print("=" * 80 + "\n")
        raise e
    # Commit checkpoint to volume
    volume.commit()
    print(f"Stage 2 complete. Checkpoint saved to: {out_dir}")


# ---------------------------------------------------------------------------
# Convenience: Run all stages sequentially
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    config: str = "gptneo_125m_metabolic.yaml",
    skip_data: bool = False,
    overrides: str = "",
):
    """
    Run the full pipeline sequentially: Stage 1 (data) → Stage 2 (train).

    Usage:
        modal run run_modal_training.py --config gptneo_125m_metabolic.yaml
        modal run run_modal_training.py --config gptneo_125m_metabolic.yaml --skip-data
    """
    if not skip_data:
        print("=== Stage 1: Data Preparation ===")
        stage_data.remote(config=config, overrides=overrides)

    print("=== Stage 2: Training ===")
    stage_train.remote(config=config, overrides=overrides)
    print("Pipeline complete.")
