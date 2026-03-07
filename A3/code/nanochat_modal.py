"""
Usage
-----
Full speedrun (mirrors `bash runs/speedrun.sh`):
    modal run nanochat_modal.py

Individual stages (if you want to re-run one step):
    modal run nanochat_modal.py::stage_data
    modal run nanochat_modal.py::stage_tokenizer
    modal run nanochat_modal.py::stage_pretrain
    modal run nanochat_modal.py::stage_post_pretrain_eval
    modal run nanochat_modal.py::stage_sft
    modal run nanochat_modal.py::stage_rl          # optional
    modal run nanochat_modal.py::stage_chat_sample

Part 3 (context extension) helper stages:
    modal run nanochat_modal.py::part3_phase1_512
    modal run nanochat_modal.py::part3_phase2_2048 --resume-from-step <STEP>
    modal run nanochat_modal.py::part3_eval_needle --phase1-step <STEP1> --phase2-step <STEP2>

Cost reference (8×H100 at ~$31/hr for the node)
------------------------------------------------
    quick_test  d12, 8 shards    : ~15 min
    speedrun    d24, 240 shards  : ~3 hours

Notes
-----
- Modal Volumes persist data between runs, so downloaded shards and
  checkpoints survive container restarts. Stages are idempotent where
  possible (they skip work already done).
- The nanochat repo is regularly updated. If a flag name changes, check the
  matching speedrun.sh in your cloned repo and update the args
"""

import os
import subprocess
import modal
from modal import App, Image as ModalImage, Volume, Secret

# =============================================================================
# CONFIGURATION
# =============================================================================

# ── Model depth ──────────────────────────────────────────────────────────────
#   d12  ~125M params   5 min on 8xH100    good for iterating on code changes
#   d20  ~560M params   1.5 hr on 8xH100   budget speedrun (~$36)
#   d24  ~768M params   3 hr on 8xH100     
#   d26  ~1B params     6 hr on 8xH100 
#   d32  ~1.9B params   41 hr on 8xH100
DEPTH = 12

# ── Data shards ───────────────────────────────────────────────────────────────
# FineWeb-EDU is split into 1822 parquet shards, each ~250M chars / ~100MB.
# 240 shards is enough for d24. Use 450 for d26 and 800 for d32.
NUM_SHARDS = 20

# ── GPU configuration ─────────────────────────────────────────────────────────
# "H100:8" = 8 H100s, the reference configuration for the speedrun leaderboard.
# "H100:4" = 4 H100s, half the speed, same cost per GPU-hour.
# "A100:8" = 8 A100 80GBs, ~10-20% slower than H100s but sometimes cheaper.
# Single GPU works too — code auto-compensates with gradient accumulation.
GPU_PRETRAIN = "H100:4"
GPU_FINETUNE = "H100:4"   # SFT and RL don't need all 8 GPUs

# ── Device batch size ─────────────────────────────────────────────────────────
# Sequences per GPU per forward pass. Reduce if you hit OOM.
# The training script automatically adjusts gradient accumulation to compensate
# so the effective total batch size (524,288 tokens default) stays the same.
#
#   H100 80GB: 32 fits for d24, 16 for d26, 8 for d32
#   A100 80GB: same as H100
#   A100 40GB: use 16 for d24
DEVICE_BATCH_SIZE = 16

# ── WandB ─────────────────────────────────────────────────────────────────────
# Set to "dummy" to disable WandB logging
WANDB_RUN = "dummy"

# ── Pretrain controls (explicit defaults from scripts.base_train) ────────────
PRETRAIN_MAX_SEQ_LEN = 2048
PRETRAIN_WINDOW_PATTERN = "SSSL"

# ── Part 3 defaults (context extension) ───────────────────────────────────────
P3_MODEL_TAG = "p3-pico"
P3_WINDOW_PATTERN = "SSSL"
P3_PHASE1_SEQ_LEN = 512
P3_PHASE2_SEQ_LEN = 2048
P3_PHASE1_NUM_ITERS = 2000
P3_PHASE2_NUM_ITERS = 4000

# ── Volume mount path ──────────────────────────────────────────────────────────
# All cached data (shards, tokenizer, checkpoints, eval bundle) lives here
# inside the Modal Volume. nanochat defaults to ~/.cache/nanochat; symlink
# the path to here so the code finds everything without modification.
VOLUME_MOUNT = "/vol"
NANOCHAT_CACHE = f"{VOLUME_MOUNT}/nanochat_cache"  # mirrors $NANOCHAT_BASE_DIR
BASE_DIR = "/data/.cache/nanochat" 

# ── Timeout ───────────────────────────────────────────────────────────────────
# Modal kills a container after this many seconds of wall-clock time.
# The pretrain timeout must be longer than your expected training time.
PRETRAIN_TIMEOUT_SEC  = 60 * 60 * 6    # 6 hours
FINETUNE_TIMEOUT_SEC  = 60 * 60 * 2    # 2 hours (SFT and RL are much shorter)
DOWNLOAD_TIMEOUT_SEC  = 60 * 90        # 90 min for shard download

# ── Derived: GPU count ────────────────────────────────────────────────────────
# Extract the integer from "H100:8" -> 8.  Used to pass --nproc_per_node.
_N_PRETRAIN_GPUS  = int(GPU_PRETRAIN.split(":")[1]) if ":" in GPU_PRETRAIN else 1
_N_FINETUNE_GPUS  = int(GPU_FINETUNE.split(":")[1]) if ":" in GPU_FINETUNE else 1

# Eval bundle URL (fixed, hosted by Karpathy)
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

# Identity conversations for SFT personality layer
IDENTITY_JSONL_URL = (
    "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
)

# =============================================================================
# MODAL PRIMITIVES — App, Volume, Secret, Image
# =============================================================================

app = modal.App("nanochat-speedrun")

# Persistent network volume: survives container shutdowns.
# Stores downloaded shards (~24GB), tokenizer, checkpoints, eval bundle.
# First time you run, Modal creates this automatically.
volume = Volume.from_name("nanochat-vol", create_if_missing=True)

# Secret: injects WANDB_API_KEY and HF_TOKEN as env vars inside containers.
# Create once with:
#   modal secret create nanochat-secrets WANDB_API_KEY=... HF_TOKEN=hf_...
secret = Secret.from_name("nanochat-secrets")

# Container image -- built once, cached by Modal until you change it.
# Mirrors the environment setup block at the top of speedrun.sh:
#   command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
#   uv sync
#   maturin develop --release --manifest-path rustbpe/Cargo.toml
image = (
    # NVIDIA CUDA 12.8 with Python 3.11
    ModalImage.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")

    # System dependencies
    .apt_install("git", "build-essential", "curl", "wget", "unzip")

    # Copy nanochat repo into the image
    .add_local_dir(
        local_path="./nanochat",
        remote_path="/root/nanochat",
        copy=True,
        ignore=[
            "**/.git/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/.venv/**",
            "**/venv/**",
            "**/.pytest_cache/**",
            "**/.mypy_cache/**",
            "**/wandb/**",
        ],
    )
    .workdir("/root/nanochat")

    # Install Rust and uv
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        "echo 'source $HOME/.cargo/env' >> $HOME/.bashrc",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "echo 'export PATH=\"$HOME/.cargo/bin:$PATH\"' >> $HOME/.bashrc",
        "bash -c 'source $HOME/.cargo/env'",
    )
    .pip_install("uv")
    # Environment variables
    .env({
        "OMP_NUM_THREADS": "1",
        "NANOCHAT_BASE_DIR": "/data/.cache/nanochat",
        "HF_HOME": "/data/.cache/huggingface",
    })
    .run_commands("ls /root/nanochat/.venv/bin/python || echo 'VENV NOT FOUND'")
    .run_commands(
        "cd /root/nanochat && uv sync --extra gpu --no-install-project",
    )
)

# =============================================================================
# HELPERS
# =============================================================================

def _python(module: str, args: list | None = None, *, cwd: str = "/root/nanochat") -> None:
    """Run `python -m {module} [args]` -- for non-distributed scripts."""
    args = args or []
    cmd = f"cd {cwd} && uv run python -m {module} {' '.join(args)}"
    _run(cmd)


def _torchrun(module: str, args: list | None = None, *, nproc: int) -> None:
    """
    Run a nanochat training script under torchrun for multi-GPU distributed execution.

    Mirrors the pattern used throughout speedrun.sh:
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m {module} -- {args}

    torchrun spawns `nproc` processes (one per GPU), assigns each a local rank,
    and sets up NCCL for gradient synchronisation across GPUs.
    --standalone means single-node (no multi-machine rendezvous server needed).
    The -- separates torchrun's own flags from the script's argument parser.
    """
    args = args or []
    args_str = (" -- " + " ".join(args)) if args else ""
    cmd = (
        f"cd /root/nanochat && "
        f"uv run torchrun --standalone --nproc_per_node={nproc} -m {module}{args_str}"
    )
    print(cmd)
    _run(cmd)


def _run(cmd: str) -> None:
    """Shell out to bash, stream stdout/stderr, and raise on failure."""
    print(f"\n>>>  {cmd}\n")
    result = subprocess.run(["bash", "-c", cmd], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command exited with code {result.returncode}:\n  {cmd}")


# def _setup_base_dir():
#     os.makedirs(BASE_DIR, exist_ok=True)
#     os.makedirs(f"{BASE_DIR}/base_data", exist_ok=True)
#     os.makedirs(f"{BASE_DIR}/tokenizer", exist_ok=True)
#     os.makedirs(f"{BASE_DIR}/checkpoints", exist_ok=True)
#     os.makedirs(f"{BASE_DIR}/eval_bundle", exist_ok=True)
#     os.makedirs(f"{BASE_DIR}/report", exist_ok=True)

def _setup_cache() -> None:
    """
    Create cache directories and symlink ~/.cache/nanochat -> the volume.

    nanochat hardcodes $NANOCHAT_BASE_DIR (defaulting to ~/.cache/nanochat) as
    the root for all its output: data shards, the tokenizer, checkpoints,
    the eval bundle, and the markdown report.  By symlinking that path to
    our persistent Modal Volume, everything survives across container restarts.

    speedrun.sh:
        export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
        mkdir -p $NANOCHAT_BASE_DIR
    """
    # _setup_base_dir()
    os.makedirs(NANOCHAT_CACHE, exist_ok=True)

    if not os.path.lexists(BASE_DIR):
        os.makedirs("/data/.cache/", exist_ok=True)
        os.symlink(NANOCHAT_CACHE, BASE_DIR)
        print(f"Symlinked {BASE_DIR} -> {NANOCHAT_CACHE}")
    else:
        print(f"Cache symlink already exists: {BASE_DIR}")


def _curl(url: str, dest: str) -> None:
    """Download a file with curl, skipping if already present."""
    if os.path.exists(dest):
        print(f"Already cached, skipping: {dest}")
        return
    _run(f"curl -L -o {dest} {url}")


# =============================================================================
# STAGE 0: DATA DOWNLOAD
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    cpu=8,
    memory=16384,
    timeout=DOWNLOAD_TIMEOUT_SEC,
)
def stage_data(num_shards: int = NUM_SHARDS) -> None:
    """
    Download FineWeb-EDU dataset shards (CPU-only, run once).

    speedrun.sh:
        python -m nanochat.dataset -n 240

    Each shard is one parquet file of ~250M chars / ~100MB of high-quality
    educational web text, re-packaged by Karpathy from HuggingFace.
    nanochat.dataset parallelises the download internally and skips shards
    that are already present on disk -- this stage is idempotent.

    240 shards = ~24GB = enough data for a d24 model at the default
    tokens:params ratio (~10x Chinchilla-optimal).
    """
    _setup_cache()
    print(f"Downloading {num_shards} FineWeb-EDU shards...")
    _python("nanochat.dataset", [f"-n {num_shards}"])
    volume.commit()
    print(f"Done: {num_shards} shards downloaded.")


# =============================================================================
# STAGE 1: TOKENIZER TRAINING
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu="H100:1",
    timeout=60 * 30,
)
def stage_tokenizer() -> None:
    """
    Train a custom BPE tokenizer on 2B characters of FineWeb-EDU.

    speedrun.sh:
        python -m scripts.tok_train --max-chars=2000000000
        python -m scripts.tok_eval

    The tokenizer is implemented in Rust (rustbpe/) for speed and wrapped in
    a Python API in nanochat/tokenizer.py. It uses the same algorithm as GPT-4:
    regex pre-splitting followed by byte-level BPE. The default vocab size is
    2^16 = 65,536 tokens (9 are reserved as special chat tokens like
    <|user_start|>, <|assistant_start|>, etc.).

    tok_eval prints the compression ratio (should be ~4.8 chars/token, beating
    GPT-2's ~3.9 chars/token).

    This stage takes ~1-2 minutes and only needs to run once.
    """
    _setup_cache()

    tokenizer_path = os.path.join(NANOCHAT_CACHE, "tokenizer.model")
    if os.path.exists(tokenizer_path):
        print("Tokenizer already trained. Skipping tok_train.")
    else:
        print("Training tokenizer on 2B characters...")
        # speedrun.sh: python -m scripts.tok_train --max-chars=2000000000
        _python("scripts.tok_train", ["--max-chars=2000000000"])
        volume.commit()

    # speedrun.sh: python -m scripts.tok_eval
    print("Evaluating tokenizer compression ratio...")
    _python("scripts.tok_eval")
    print("Tokenizer ready.")


# =============================================================================
# STAGE 2: BASE MODEL PRETRAINING
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_PRETRAIN,
    timeout=PRETRAIN_TIMEOUT_SEC,
)
def stage_pretrain(
    depth: int = DEPTH,
    device_batch_size: int = DEVICE_BATCH_SIZE,
    wandb_run: str = WANDB_RUN,
    max_seq_len: int = PRETRAIN_MAX_SEQ_LEN,
    window_pattern: str = PRETRAIN_WINDOW_PATTERN,
    model_tag: str = "",
    resume_from_step: int = -1,
    num_iterations: int = -1,
) -> None:
    _run_pretrain_stage(
        depth=depth,
        device_batch_size=device_batch_size,
        wandb_run=wandb_run,
        max_seq_len=max_seq_len,
        window_pattern=window_pattern,
        model_tag=model_tag,
        resume_from_step=resume_from_step,
        num_iterations=num_iterations,
    )


def _run_pretrain_stage(
    depth: int,
    device_batch_size: int,
    wandb_run: str,
    max_seq_len: int,
    window_pattern: str,
    model_tag: str,
    resume_from_step: int,
    num_iterations: int,
) -> None:
    """
    Pretrain the base GPT model on FineWeb-EDU from random initialization.

    speedrun.sh:
        python -m nanochat.report reset
        torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \\
            --depth=20 \\
            --device-batch-size=16 \\
            --run=$WANDB_RUN

    This is the most compute-intensive stage. The training loop in
    scripts/base_train.py implements:
        - Chinchilla-optimal token budget derived from depth
        - Muon optimizer for weight matrices, AdamW for embeddings
        - BOS-aligned BestFit-Crop data packing (no midtraining)
        - Cosine LR warmup + linear warmdown (50% of training)
        - Gradient accumulation if device_batch_size * n_gpus < target batch

    Flags:
        --depth               Transformer depth; controls all other hparams
        --device-batch-size   Sequences per GPU per step (reduce if OOM)
        --run                 WandB run name ("dummy" to disable logging)
        --save-every          Checkpoint every N steps (resume-friendly)
    """
    _setup_cache()

    # speedrun.sh: python -m nanochat.report reset
    # Resets the markdown report file and writes system info + run timestamp.
    print("Resetting training report...")
    _python("nanochat.report", ["reset"])

    print(
        f"Starting pretraining: depth={depth}, "
        f"max_seq_len={max_seq_len}, "
        f"window_pattern={window_pattern}, "
        f"device_batch_size={device_batch_size}, "
        f"nproc={_N_PRETRAIN_GPUS}, run={wandb_run}, "
        f"model_tag={model_tag or f'd{depth}'}, resume_from_step={resume_from_step}, "
        f"num_iterations={num_iterations}"
    )

    # speedrun.sh: torchrun --standalone --nproc_per_node=$NPROC_PER_NODE
    #              -m scripts.base_train -- --depth=24 --device-batch-size=16 --run=...
    train_args = [
        f"--depth={depth}",
        f"--max-seq-len={max_seq_len}",
        f"--window-pattern={window_pattern}",
        f"--device-batch-size={device_batch_size}",
        f"--run={wandb_run}",
        "--save-every=1000",    # checkpoint every 1k steps for resilience
    ]
    if model_tag:
        train_args.append(f"--model-tag={model_tag}")
    if resume_from_step != -1:
        train_args.append(f"--resume-from-step={resume_from_step}")
    if num_iterations != -1:
        train_args.append(f"--num-iterations={num_iterations}")

    _torchrun("scripts.base_train", train_args, nproc=_N_PRETRAIN_GPUS)

    volume.commit()
    print("Pretraining complete.")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_PRETRAIN,
    timeout=PRETRAIN_TIMEOUT_SEC,
)
def part3_phase1_512(
    depth: int = DEPTH,
    device_batch_size: int = DEVICE_BATCH_SIZE,
    wandb_run: str = "a3-p3-512",
) -> None:
    """Part 3 phase 1: train at short context (default 512)."""
    _run_pretrain_stage(
        depth=depth,
        device_batch_size=device_batch_size,
        wandb_run=wandb_run,
        max_seq_len=P3_PHASE1_SEQ_LEN,
        window_pattern=P3_WINDOW_PATTERN,
        model_tag=P3_MODEL_TAG,
        resume_from_step=-1,
        num_iterations=P3_PHASE1_NUM_ITERS,
    )


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_PRETRAIN,
    timeout=PRETRAIN_TIMEOUT_SEC,
)
def part3_phase2_2048(
    resume_from_step: int,
    depth: int = DEPTH,
    device_batch_size: int = DEVICE_BATCH_SIZE,
    wandb_run: str = "a3-p3-2048",
) -> None:
    """Part 3 phase 2: resume at long context (default 2048)."""
    _run_pretrain_stage(
        depth=depth,
        device_batch_size=device_batch_size,
        wandb_run=wandb_run,
        max_seq_len=P3_PHASE2_SEQ_LEN,
        window_pattern=P3_WINDOW_PATTERN,
        model_tag=P3_MODEL_TAG,
        resume_from_step=resume_from_step,
        num_iterations=P3_PHASE2_NUM_ITERS,
    )


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_PRETRAIN,
    timeout=60 * 60 * 2,
)
def part3_eval_needle(
    phase1_tag: str = P3_MODEL_TAG,
    phase1_step: int = P3_PHASE1_NUM_ITERS,
    phase2_tag: str = P3_MODEL_TAG,
    phase2_step: int = P3_PHASE2_NUM_ITERS,
    context_lengths: str = "256,512,1024,1536,2048",
    positions: str = "0.1,0.5,0.9",
    trials: int = 10,
    num_decoys: int = 4,
) -> None:
    """Run Part 3 needle retrieval eval comparing phase-1 vs phase-2 checkpoints."""
    _setup_cache()
    _python(
        "scripts.part3_eval",
        [
            f"--phase1-tag={phase1_tag}",
            f"--phase1-step={phase1_step}",
            f"--phase2-tag={phase2_tag}",
            f"--phase2-step={phase2_step}",
            f"--context-lengths={context_lengths}",
            f"--positions={positions}",
            f"--trials={trials}",
            f"--num-decoys={num_decoys}",
            "--output-json=/tmp/part3_eval_results.json",
        ],
    )
    _run("cp /tmp/part3_eval_results.json /vol/part3_eval_results.json")
    volume.commit()
    print("Saved needle eval JSON to /vol/part3_eval_results.json")


@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_PRETRAIN,
    timeout=60 * 60 * 2,
)
def part3_eval_2of3(
    phase1_tag: str = P3_MODEL_TAG,
    phase1_step: int = P3_PHASE1_NUM_ITERS,
    phase2_tag: str = P3_MODEL_TAG,
    phase2_step: int = P3_PHASE2_NUM_ITERS,
    context_lengths: str = "256,512,1024,1536,2048",
    trials: int = 10,
) -> None:
    """Run Part 3 2-of-3 docs multiple-choice eval for phase-1 vs phase-2."""
    _setup_cache()
    _python(
        "scripts.part3_eval_2of3",
        [
            f"--phase1-tag={phase1_tag}",
            f"--phase1-step={phase1_step}",
            f"--phase2-tag={phase2_tag}",
            f"--phase2-step={phase2_step}",
            f"--context-lengths={context_lengths}",
            f"--trials={trials}",
            "--output-json=/tmp/part3_eval_2of3_results.json",
        ],
    )
    _run("cp /tmp/part3_eval_2of3_results.json /vol/part3_eval_2of3_results.json")
    volume.commit()
    print("Saved 2-of-3 eval JSON to /vol/part3_eval_2of3_results.json")


# =============================================================================
# STAGE 3: POST-PRETRAIN EVALUATION
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_PRETRAIN,
    timeout=60 * 60 * 2,
)
def stage_post_pretrain_eval() -> None:
    """
    Evaluate the base model immediately after pretraining.

    speedrun.sh:
        torchrun ... -m scripts.base_eval

    scripts.base_eval  -- runs the CORE metric: zero-shot evaluation across
        22 diverse benchmarks from the DCLM paper (HellaSwag, ARC, BoolQ,
        LAMBADA, TriviaQA, ...). The target is 0.256525 (GPT-2's score).
        A successful d24 speedrun hits ~0.258-0.260. Takes ~20-40 min.

    The eval bundle (benchmark data files, ~1GB) is downloaded on first run
    and cached in the volume for subsequent runs.
    """
    _setup_cache()

    # speedrun.sh:
    #   if [ ! -d "$NANOCHAT_BASE_DIR/eval_bundle" ]; then
    #       curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL && unzip -q ...
    eval_bundle_dir = os.path.join(NANOCHAT_CACHE, "eval_bundle")
    if not os.path.isdir(eval_bundle_dir):
        print("Downloading eval bundle (~1GB)...")
        zip_path = "/tmp/eval_bundle.zip"
        _curl(EVAL_BUNDLE_URL, zip_path)
        _run(f"unzip -q {zip_path} -d {NANOCHAT_CACHE} && rm {zip_path}")
        volume.commit()

    # speedrun.sh: torchrun ... -m scripts.base_eval
    print("Running CORE evaluation (22 benchmarks, ~20-40 min)...")
    _torchrun("scripts.base_eval", nproc=_N_PRETRAIN_GPUS)

    volume.commit()
    print("Post-pretrain eval complete.")


# =============================================================================
# STAGE 4: SUPERVISED FINE-TUNING (SFT)
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_FINETUNE,
    timeout=FINETUNE_TIMEOUT_SEC,
)
def stage_sft(wandb_run: str = WANDB_RUN) -> None:
    """
    Supervised fine-tuning: teach the model to follow chat instructions.

    speedrun.sh:
        curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl $IDENTITY_URL
        torchrun ... -m scripts.chat_sft -- --run=$WANDB_RUN
        torchrun ... -m scripts.chat_eval -- -i sft

    chat_sft trains on a curated mixture of conversation data with loss masked
    to assistant-only tokens. This is the key structural difference from
    pretraining: the model sees the full context (user + assistant turns) but
    only gets gradient signal from its own tokens. User prompt tokens have
    their targets set to -1, which F.cross_entropy ignores.

    Data mixture includes:
        - SmolTalk:   ~460K general conversations (dominant)
        - MMLU:       ~100K multiple-choice knowledge questions
        - ARC:        ~8K science reasoning questions
        - GSM8K:      math word problems with calculator tool use
        - HumanEval:  Python coding tasks
        - identity_conversations.jsonl: synthetic data teaching self-awareness

    identity_conversations.jsonl is downloaded fresh each time from Karpathy's
    S3. It's a small file (~a few hundred rows) that teaches the model its name,
    creator, and basic facts about itself. See dev/gen_synthetic_data.py for how
    to generate your own custom version.

    chat_eval -i sft runs task-specific evals (GSM8K accuracy, HumanEval pass@1,
    MMLU accuracy) on the SFT checkpoint and appends results to the report.
    """
    _setup_cache()

    # speedrun.sh: curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl $URL
    identity_dest = os.path.join(NANOCHAT_CACHE, "identity_conversations.jsonl")
    print("Downloading identity conversations for SFT personality layer...")
    _curl(IDENTITY_JSONL_URL, identity_dest)

    # speedrun.sh: torchrun ... -m scripts.chat_sft -- --run=$WANDB_RUN
    print("Running SFT...")
    _torchrun(
        "scripts.chat_sft",
        [
            f"--run={wandb_run}",
            "--model-step=4357",
            "--model-tag=d20"
        ],
        nproc=_N_FINETUNE_GPUS,
    )

    # speedrun.sh: torchrun ... -m scripts.chat_eval -- -i sft
    # -i sft tells chat_eval to load the SFT checkpoint (not base or rl)
    print("Evaluating SFT checkpoint on task benchmarks...")
    _torchrun("scripts.chat_eval", 
              [
                  "-i", "sft",
               ], 
              nproc=_N_FINETUNE_GPUS)

    volume.commit()
    print("SFT complete.")

# =============================================================================
# STAGE 5: REINFORCEMENT LEARNING (optional)
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu=GPU_FINETUNE,
    timeout=FINETUNE_TIMEOUT_SEC,
)
def stage_rl(wandb_run: str = WANDB_RUN) -> None:
    """
    Optional RL stage to boost math reasoning on GSM8K.

    speedrun.sh:
        torchrun ... -m scripts.chat_rl -- --run=$WANDB_RUN
        torchrun ... -m scripts.chat_eval -- -i rl

    Uses a simplified GRPO/REINFORCE variant trained on GSM8K math word
    problems. The model generates multiple candidate answers, checks each
    against the ground truth integer, and uses correct/incorrect as a binary
    reward signal. No value network, no KL penalty against the SFT reference.

    From the source comment: "I put GRPO in quotes because we actually end up
    with something a lot simpler and more similar to just REINFORCE."

    Expected improvement: GSM8K accuracy ~5% (SFT) -> ~15-20% (after RL).

    This stage is NOT part of the default speedrun.sh -- it's an optional
    extension. Run it separately after stage_sft:
        modal run nanochat_modal.py::stage_rl
    """
    _setup_cache()

    print("Running RL (GRPO on GSM8K)...")
    # speedrun.sh: torchrun ... -m scripts.chat_rl -- --run=$WANDB_RUN
    _torchrun(
        "scripts.chat_rl",
        [
            f"--run={wandb_run}",
            "--model-step=4357",
            "--model-tag=d20"
        ],
        nproc=_N_FINETUNE_GPUS,
    )

    # speedrun.sh: torchrun ... -m scripts.chat_eval -- -i rl
    print("Evaluating RL checkpoint...")
    _torchrun("scripts.chat_eval", ["-i", "rl"], nproc=_N_FINETUNE_GPUS)

    volume.commit()
    print("RL complete.")


# =============================================================================
# FULL SPEEDRUN PIPELINE (main entrypoint)
# =============================================================================

@app.local_entrypoint()
def main() -> None:
    """
    Run the complete speedrun pipeline, mirroring runs/speedrun.sh end-to-end.

    This is what executes when you run: modal run nanochat_modal.py

    Stage order (matches speedrun.sh top to bottom):
        0. Download FineWeb-EDU shards       (CPU, ~20 min for 240 shards)
        1. Train BPE tokenizer               (1 GPU, ~2 min)
        2. Pretrain base model               (8 GPU, ~3 hours for d24)
        3. Post-pretrain eval (loss + CORE)  (8 GPU, ~30 min)
        4. SFT + chat_eval                   (4 GPU, ~30-45 min)
        5. Chat sample                       (1 GPU, ~1 min)

    Each stage is a separate Modal function call with its own container, GPU
    allocation, and log stream. If a stage fails, re-run it individually:
        modal run nanochat_modal.py::stage_pretrain

    The optional RL stage is NOT included in the default pipeline. Run it
    manually after stage_sft if you want the math reasoning boost:
        modal run nanochat_modal.py::stage_rl
    """
    w = 64
    print("\n" + "=" * w)
    print("nanochat Speedrun -- Modal Edition")
    print(f"  Mirrors: runs/speedrun.sh")
    print(f"  depth={DEPTH}  shards={NUM_SHARDS}  gpu={GPU_PRETRAIN}  wandb={WANDB_RUN}")
    print("=" * w + "\n")

    # Stage 0: Data
    # speedrun.sh: python -m nanochat.dataset -n 240
    print("[0/5] Downloading FineWeb-EDU shards...")
    stage_data.remote(num_shards=NUM_SHARDS)

    # Stage 1: Tokenizer
    # speedrun.sh: python -m scripts.tok_train && python -m scripts.tok_eval
    print("[1/5] Training tokenizer...")
    stage_tokenizer.remote()

    # Stage 2: Pretrain
    # speedrun.sh: python -m nanochat.report reset
    #              torchrun ... -m scripts.base_train -- --depth=24 ...
    print("[2/5] Pretraining base model (the long one)...")
    stage_pretrain.remote(depth=DEPTH, device_batch_size=DEVICE_BATCH_SIZE, wandb_run=WANDB_RUN)

    # Stage 3: Post-pretrain eval
    #              torchrun ... -m scripts.base_eval
    print("[3/5] Evaluating base model (bits-per-byte + CORE)...")
    stage_post_pretrain_eval.remote()

    # Stage 4: SFT + eval
    # speedrun.sh: curl identity_conversations.jsonl
    #              torchrun ... -m scripts.chat_sft -- --run=...
    #              torchrun ... -m scripts.chat_eval -- -i sft
    print("[4/5] Supervised fine-tuning + eval...")
    stage_sft.remote(wandb_run=WANDB_RUN)

    print("\n" + "=" * w)
    print("Speedrun complete!")
    print("  Checkpoints + report are in the 'nanochat-vol' Modal Volume.")
    print("  Optional RL stage: modal run nanochat_modal.py::stage_rl")
    print("=" * w + "\n")


# =============================================================================
# QUICK TEST
# =============================================================================

@app.function(
    image=image,
    secrets=[secret],
    volumes={VOLUME_MOUNT: volume},
    gpu="H100:4",
    timeout=60 * 60 * 2,
)
def quick_test() -> None:
    """
    End-to-end smoke test using a tiny d12 model and only 8 data shards.

    d12 = 12-layer transformer, ~125M params (GPT-1 scale).
    Downloads only 8 shards (~800MB), trains in ~5 min on 4xH100.

    If this passes without errors, you know:
        - The container image built correctly (Rust/uv/maturin all work)
        - The volume mount is working (data persists)
        - The secret injection is working (HF_TOKEN for download)
        - torchrun multi-GPU distributed training works
        - The full code path from data -> tokenizer -> pretrain -> SFT -> chat runs
    """
    _setup_cache()

    nproc = 4

    # 1. Download a handful of shards to get data on the volume
    print("Downloading 8 shards for quick test...")
    _python("nanochat.dataset", ["-n 8"])
    volume.commit()

    # 2. Train tokenizer on 500M chars instead of 2B (much faster)
    print("Training tokenizer (500M chars)...")
    _python("scripts.tok_train", ["--max-chars=500000000"])
    _python("scripts.tok_eval")

    # 3. Quick pretrain: d12, skip CORE metric (slow), skip intermediate saves
    print("Pretraining d12 (no CORE metric, no intermediate saves)...")
    _torchrun(
        "scripts.base_train",
        [
            "--depth=12",
            "--device-batch-size=32",
            "--run=dummy",
            "--core-metric-every=999999",   # skip CORE during training (it's slow)
            "--sample-every=-1",            # skip intermediate samples
            "--save-every=-1",              # skip intermediate checkpoints
        ],
        nproc=nproc,
    )

    # 4. Minimal SFT to verify the code path runs end-to-end
    print("Quick SFT...")
    identity_dest = os.path.join(NANOCHAT_CACHE, "identity_conversations.jsonl")
    _curl(IDENTITY_JSONL_URL, identity_dest)
    _torchrun("scripts.chat_sft", ["--run=dummy"], nproc=nproc)

    # 5. Chat sample to confirm inference works
    print("Chat sample...")
    _python("scripts.chat_cli", ['-p "Hello, who are you?"', "-i sft"])

    volume.commit()
    print("\nQuick test passed! Ready for the full speedrun.")
