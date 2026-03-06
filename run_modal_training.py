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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# GPU for training. Options: "A10G", "A100", "H100", "A100:8", "H100:8"
# Single GPU is fine for experiments; upgrade for large-scale runs.
GPU_TRAIN = os.environ.get("TMOE_GPU", "A10G")

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

# Container image: install requirements, then map the rest of the code.
image = (
    Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .env({"PYTHONPATH": "/app"})
    .add_local_dir(".", remote_path="/app")
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

    # Build and run train command, explicitly passing output dir to Volume
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

    subprocess.run(cmd, cwd="/app", check=True)

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
