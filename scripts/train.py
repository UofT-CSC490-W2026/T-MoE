"""
scripts/train.py — Stage 2: Model Training

Trainer for T-MoE. Reads pre-packed binary shards, builds the model via
the ModelRegistry, and trains with WandB logging. Model-agnostic: adding
a new backbone requires no changes here.

Usage:
    # Local
    python -m scripts.train --config experiments/gptneo_125m_stress_v6-wikitext.yaml

    # With CLI overrides
    python -m scripts.train --config experiments/gptneo_125m_stress_v6-wikitext.yaml \\
        training.lr=1e-4 training.batch_size=8

    # Via Modal (set CONFIG = "experiments/..." in run_modal_training.py, then):
    modal run run_modal_training.py --skip-data
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import json
import math
import os
import random
import struct
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader

from src.training.fsdp_utils import (
    init_distributed,
    cleanup_distributed,
    is_main_process,
    wrap_model_for_distributed,
    get_model_for_attr_access,
)
from src.configs.dataset import get_shard_dir


class ShardDataset(Dataset):
    """
    Reads packed binary shards produced by scripts/prepare_data.py.

    Each .bin file is: 8-byte header (uint64 token_count) + optional 2-byte dtype_flag + tokens (uint16 or uint32).
    Legacy shards omit the dtype_flag and always use uint16 (GPT-Neo). Versioned shards (Qwen2) use uint32.
    Sequences are sliced out of the continuous token stream — no padding.

    Memory-efficient: only shard headers are read at init time. Token data
    is accessed via np.memmap on demand, so arbitrarily large datasets
    (C4, The Pile) work without OOM.
    """

    def __init__(self, shard_dir: Path, split: str, seq_len: int):
        self.seq_len = seq_len
        self.shards = sorted(shard_dir.glob(f"{split}_shard_*.bin"))
        if not self.shards:
            raise FileNotFoundError(
                f"No shards found for split '{split}' in {shard_dir}.\n"
                f"Run 'python -m scripts.prepare_data --config <your_config.yaml>' first."
            )

        # Read shard headers — detect legacy (8-byte) vs versioned (10-byte) format.
        # prepare_data.py writes: struct.pack("<QH", token_count, dtype_flag)
        #   dtype_flag=0 → uint16 (vocab ≤ 65535, e.g. GPT-Neo)
        #   dtype_flag=1 → uint32 (vocab > 65535, e.g. Qwen2 ~150k vocab)
        self.shard_sizes: list[int] = []
        self.shard_meta: list[tuple] = []  # (dtype, offset) per shard
        for path in self.shards:
            file_size = path.stat().st_size
            with open(path, "rb") as f:
                token_count = struct.unpack("<Q", f.read(8))[0]

            # Detect legacy (8-byte header) vs versioned (10-byte header)
            if file_size - 8 == token_count * 2:
                # Legacy uint16 shard
                dtype = np.uint16
                offset = 8
            else:
                with open(path, "rb") as f:
                    f.read(8)
                    dtype_flag = struct.unpack("<H", f.read(2))[0]
                if dtype_flag == 0:
                    dtype = np.uint16
                elif dtype_flag == 1:
                    dtype = np.uint32
                else:
                    raise ValueError(f"Unknown dtype_flag={dtype_flag} in {path}")
                offset = 10

            self.shard_sizes.append(token_count)
            self.shard_meta.append((dtype, offset))

        # Cumulative token offsets for O(log N) shard resolution in __getitem__.
        self.cumulative = [0]
        for s in self.shard_sizes:
            self.cumulative.append(self.cumulative[-1] + s)

        total_tokens = self.cumulative[-1]
        self.n_seqs = (total_tokens - 1) // seq_len

        # Cache one memmap per shard at init time — avoids re-opening file
        # descriptors and recreating OS memory mappings on every __getitem__ call.
        # Each mmap covers the full token payload of the shard (after the header).
        self.mmaps: list[np.memmap] = [
            np.memmap(path, dtype=dtype, mode="r", offset=offset, shape=(size,))
            for path, (dtype, offset), size in zip(
                self.shards, self.shard_meta, self.shard_sizes
            )
        ]

    def __len__(self):
        return self.n_seqs

    def __getitem__(self, idx):
        global_start = idx * self.seq_len
        tokens = np.empty(self.seq_len + 1, dtype=np.int64)
        filled = 0
        pos = global_start

        while filled < self.seq_len + 1:
            # Binary search: which shard owns `pos`?
            shard_idx = bisect.bisect_right(self.cumulative, pos) - 1
            shard_idx = min(shard_idx, len(self.shards) - 1)

            local_offset = pos - self.cumulative[shard_idx]
            available = self.shard_sizes[shard_idx] - local_offset
            need = (self.seq_len + 1) - filled
            n = min(available, need)

            chunk = self.mmaps[shard_idx][local_offset : local_offset + n].astype(
                np.int64
            )
            tokens[filled : filled + n] = chunk
            filled += n
            pos += n

            if pos >= self.cumulative[-1]:  # wrap around
                pos = 0

        ids = torch.from_numpy(tokens)
        return ids, ids


def load_config(config_path: str, overrides: list[str]) -> DictConfig:
    """
    Load experiment YAML config and merge CLI overrides.

    Example override: "training.lr=1e-4 training.batch_size=8"
    Identical semantics to Hydra, without Hydra's initialization complexity.
    """
    cfg = OmegaConf.load(config_path)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="T-MoE Trainer")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML (e.g. experiments/gptneo_125m_stress_v6-wikitext.yaml)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from (e.g. outputs/run/ckpt.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (used by Modal to write into the persistent Volume).",
    )
    parser.add_argument(
        "--shard-dir",
        type=str,
        default=None,
        help="Override shard directory (defaults to data/shards/<dataset_key>/).",
    )
    args, overrides = parser.parse_known_args()
    return args, overrides


def build_model(cfg) -> torch.nn.Module:
    """Build model + LoRAMoE layers from config. Model-agnostic via ModelRegistry."""
    from src.configs.model import model_lookup
    from src.core import ModelRegistry
    from src.layers.lora_moe import LoRAMoELayer
    from src.experts.lora import LoRAConfig
    from src.project_types import ExpertType
    from src.routers import create_router

    # Side-effect: triggers @ModelRegistry.register decorators
    import src.models  # noqa: F401

    model_key = cfg.model.model_key  # e.g. "gpt-neo-125m"
    model_info = model_lookup(model_key)
    model_type = model_info["model_type"]  # e.g. "gpt_neo"
    variant = model_info["variant"]  # e.g. "125m"

    model_cls = ModelRegistry.get(model_type)
    model = model_cls(
        variant=variant,
        freeze_backbone=cfg.model.freeze_backbone,
        moe_layer_indices=list(cfg.model.moe_layer_indices),
        device="cpu",  # FSDP handles GPU placement; single-GPU does model.to(device) later
    )

    # Build and inject MoE layers
    moe_layers = {}
    for layer_idx in cfg.model.moe_layer_indices:
        actual_idx = layer_idx if layer_idx >= 0 else model.num_layers + layer_idx
        original_mlp = model.get_mlp_at(actual_idx)

        router = create_router(
            router_type=cfg.router.type,
            hidden_dim=model.hidden_dim,
            num_experts=cfg.router.num_experts,
            top_k=cfg.router.top_k,
            noise_std=cfg.router.get("noise_std", 0.1),
            temperature=cfg.router.get("temperature", 1.0),
            eps=cfg.router.get("eps", 1e-3),
            # standard/switch-specific — aux loss; filtered out for non-standard routers
            use_aux_loss=cfg.router.get("use_aux_loss", False),
            aux_loss_coef=cfg.router.get("aux_loss_coef", 0.01),
            # deepseek-specific — filtered out for non-deepseek routers
            use_sigmoid=cfg.router.get("use_sigmoid", False),
            bias_update_rate=cfg.router.get("bias_update_rate", 1e-3),
            # SPAR-specific — filtered out for non-SPAR routers by create_router
            ema_alpha=cfg.router.get("ema_alpha", 0.01),
            lambda_calib_step=cfg.router.get("lambda_calib_step", 600),
            lambda_init=cfg.router.get("lambda_init", 0.1),
            tau_final=cfg.router.get("tau_final", cfg.router.get("temperature", 1.0)),
            tau_anneal_steps=cfg.router.get("tau_anneal_steps", 0),
            noise_anneal_steps=cfg.router.get("noise_anneal_steps", 0),
            # metabolic-specific
            **(
                {
                    "lambda_metabolic": cfg.router.metabolic.get(
                        "lambda_metabolic", 0.3
                    ),
                    "gamma_recovery": cfg.router.metabolic.get("gamma_recovery", 0.15),
                    "beta_cost": cfg.router.metabolic.get("beta_cost", 0.15),
                    "tau_specialization": cfg.router.metabolic.get(
                        "tau_specialization", 2.0
                    ),
                    "F_scale": cfg.router.metabolic.get("F_scale", 0.5),
                    "warmup_steps": cfg.router.metabolic.get("warmup_steps", 1200),
                }
                if cfg.router.type == "metabolic"
                else {}
            ),
        )

        lora_cfg = LoRAConfig(
            hidden_dim=model.hidden_dim,
            rank=cfg.expert.lora.rank,
            alpha=cfg.expert.lora.alpha,
            dropout=cfg.expert.lora.dropout,
            init_scale=cfg.expert.lora.init_scale,
            b_init_scale=cfg.expert.lora.get("b_init_scale", 0.0),
            trainable_base=cfg.expert.lora.get("trainable_base", False),
            shared_base_rank=cfg.expert.lora.get("shared_base_rank", 0),
            shared_base_alpha=cfg.expert.lora.get("shared_base_alpha", 0.0),
        )

        moe_layers[actual_idx] = LoRAMoELayer.from_pretrained_mlp(
            mlp=original_mlp,
            router=router,
            lora_config=lora_cfg,
            num_experts=cfg.expert.count,
            expert_type=ExpertType(cfg.expert.type),
        )

    model.inject_moe_layers(moe_layers)
    return model


@torch.no_grad()
def _initialize_router_prototypes(
    model: torch.nn.Module,
    train_loader,
    device: str,
    is_distributed: bool,
    n_warmup_batches: int = 2,
) -> None:
    """
    Collect layer activations from n_warmup_batches and initialize each
    StressCorrectedRouter's W from k-means centroids.

    DDP strategy: rank 0 runs k-means and broadcasts W to all ranks.
    All ranks register hooks and collect activations (needed for the broadcast
    target device), but only rank 0 computes centroids.
    """
    from src.layers.lora_moe import LoRAMoELayer
    from src.routers.stress_corrected import StressCorrectedRouter

    activations_by_layer: dict[str, list] = {}
    hooks = []

    def make_hook(name: str):
        def hook(module, input, output):
            # input[0]: [B, S, D] hidden states entering the MoE layer
            activations_by_layer.setdefault(name, []).append(
                input[0].detach().float().reshape(-1, input[0].shape[-1]).cpu()
            )

        return hook

    base_model = get_model_for_attr_access(model)
    routers_by_layer: dict[str, StressCorrectedRouter] = {}
    for name, module in base_model.named_modules():
        if isinstance(module, LoRAMoELayer) and isinstance(
            module.router, StressCorrectedRouter
        ):
            hooks.append(module.register_forward_hook(make_hook(name)))
            routers_by_layer[name] = module.router

    if not routers_by_layer:
        return  # no SPAR routers, nothing to do

    # Use the unwrapped model to avoid DDP/compile complications during the warmup forward.
    base_model.eval()
    with torch.no_grad():
        for i, (x, _) in enumerate(train_loader):
            if i >= n_warmup_batches:
                break
            base_model(input_ids=x.to(device), return_metrics=False)

    for h in hooks:
        h.remove()
    base_model.train()

    # Clear Dynamo's compilation cache. The eval-mode no_grad forward above traces
    # all tensors with requires_grad=False. Without this reset, Dynamo would hit a
    # requires_grad mismatch on the first training forward and recompile up to 8
    # times before falling back to eager.
    torch._dynamo.reset()

    if not is_distributed:
        for layer_name, router in routers_by_layer.items():
            acts = torch.cat(activations_by_layer.get(layer_name, []), dim=0).to(device)
            if acts.shape[0] >= router.num_experts:
                router.initialize_prototypes_from_data(acts)
        return

    import torch.distributed as dist

    for layer_name, router in routers_by_layer.items():
        if dist.get_rank() == 0:
            acts = torch.cat(activations_by_layer.get(layer_name, []), dim=0).to(device)
            if acts.shape[0] >= router.num_experts:
                router.initialize_prototypes_from_data(acts)
        # Broadcast rank-0 prototypes to all ranks so W is identical everywhere.
        dist.broadcast(router.W.data, src=0)


def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    opt_name = cfg.training.optimizer.lower()
    lr = cfg.training.lr
    lr_base = cfg.training.get("lr_base", None)
    betas = tuple(cfg.training.get("betas", [0.9, 0.95]))
    eps = cfg.training.get("eps", 1e-8)
    wd = cfg.training.get("weight_decay", 0.1)
    # Partition parameters: optional base weights (trainable shared MLP), rest.
    _BASE_PARAM_NAMES = {"shared_fc_weight", "shared_proj_weight"}

    base_param_ids: set[int] = set()

    # Collect optional base params (trainable shared MLP weights).
    base_params = []
    if lr_base is not None:
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(n in name for n in _BASE_PARAM_NAMES):
                base_params.append(p)
                base_param_ids.add(id(p))

    # Everything else.
    other_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in base_param_ids
    ]

    # Build param groups.
    param_groups: list = [{"params": other_params, "lr": lr}]
    if lr_base is not None and base_params:
        param_groups.append({"params": base_params, "lr": lr_base})

    # fused=True fuses per-param updates into a single CUDA kernel (5-15% faster on H100/A100).
    # Requires all params on CUDA — always true during training.
    _use_fused = torch.cuda.is_available()

    if opt_name == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=wd,
            fused=_use_fused,
        )
    elif opt_name == "adam":
        return torch.optim.Adam(
            param_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            fused=_use_fused,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")


@torch.no_grad()
def _broadcast_scalar(value: float, device: str, is_distributed: bool) -> float:
    """Broadcast a scalar from rank 0 to all ranks. No-op when not distributed."""
    if not is_distributed:
        return value
    import torch.distributed as dist

    t = torch.tensor(value, device=device)
    dist.broadcast(t, src=0)
    return t.item()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, val_loader: DataLoader, device: str, max_batches: int = 20
) -> float:
    """Compute validation loss over up to max_batches batches."""
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        _, loss, _ = model(
            input_ids=x, labels=y, return_metrics=False, record_usage=False
        )
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


def init_wandb(cfg) -> None:
    if not is_main_process():
        return
    logging_cfg = cfg.get("logging", {})
    if not logging_cfg.get("enabled", False):
        return
    mode = logging_cfg.get("mode")
    if mode == "disabled":
        print("WandB disabled by config. Skipping.")
        return
    try:
        import wandb

        if mode not in {"online", "offline"}:
            env_mode = os.environ.get("WANDB_MODE")
            mode = env_mode if env_mode in {"online", "offline"} else "online"

        init_kwargs = {
            "project": logging_cfg.get("project")
            or os.environ.get("WANDB_PROJECT")
            or "tmoe",
            "name": cfg.experiment_name,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "mode": mode,
        }
        entity = logging_cfg.get("entity") or os.environ.get("WANDB_ENTITY")
        if entity:
            init_kwargs["entity"] = entity

        run = wandb.init(**init_kwargs)
        run_url = getattr(run, "url", None)
        if run_url:
            print(f"WandB initialized: {run_url}")
        else:
            print("WandB initialized.")
    except ImportError:
        print("WandB not installed. Skipping.")
    except Exception as exc:
        print(f"WandB init failed. Skipping. ({exc})")


def log_wandb(metrics: dict) -> None:
    if not is_main_process():
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(metrics)
    except ImportError:
        pass


def main():
    args, overrides = parse_args()
    cfg = load_config(args.config, overrides)

    # Enable TF32 on Ampere/Hopper GPUs — uses tensor cores for float32 matmuls
    # with no precision loss for typical training workloads.
    torch.set_float32_matmul_precision("high")

    # Reproducibility seeds — set BEFORE distributed init so all processes start from the same base state.
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Distributed init (no-op when not launched with torchrun)
    is_distributed, rank, local_rank, world_size = init_distributed()

    # Per-rank seed offset — ensures DDP ranks don't see identical batches.
    torch.manual_seed(seed + rank)

    # Device
    gpu_name = ""
    if is_distributed:
        device = f"cuda:{local_rank}"
        gpu_name = (
            torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else ""
        )
    else:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""

    if is_main_process():
        print("=" * 80)
        print(
            f"Device: {device} | World Size: {world_size}"
            + (f" | GPU: {gpu_name}" if gpu_name else "")
        )
        print(
            f"Config:\n{json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2)}"
        )
        print("=" * 80)

    try:
        # Data
        dataset_key = cfg.dataset.dataset_key
        shard_dir = (
            Path(args.shard_dir)
            if args.shard_dir
            else get_shard_dir(dataset_key, cfg.model.model_key)
        )
        seq_len = cfg.dataset.max_seq_len
        batch_size = cfg.training.batch_size

        train_ds = ShardDataset(shard_dir, "train", seq_len)

        # Use DistributedSampler so each rank sees a different subset
        # In distributed mode, 4 ranks × 4 workers = 16 processes memmapping the
        # same shard file simultaneously can exhaust /dev/shm on cloud containers.
        # Default to 2 for DDP, 4 for single-GPU. Override via training.num_workers.
        default_workers = 2 if is_distributed else 4
        num_workers = (
            cfg.training.get("num_workers", default_workers)
            if torch.cuda.is_available()
            else 0
        )
        train_sampler = None
        if is_distributed:
            from torch.utils.data.distributed import DistributedSampler

            train_sampler = DistributedSampler(
                train_ds, shuffle=True, seed=cfg.get("seed", 42)
            )
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
                prefetch_factor=4 if num_workers > 0 else None,
            )
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
                prefetch_factor=4 if num_workers > 0 else None,
            )

        val_loader = None
        try:
            val_ds = ShardDataset(shard_dir, "val", seq_len)
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
                prefetch_factor=4 if num_workers > 0 else None,
            )
        except FileNotFoundError:
            print("No validation shards found. Skipping validation.")

        # Model
        if is_main_process():
            print("Building model...")
        model = build_model(cfg)

        from src.metrics.router_metrics import GlobalSpecializationTracker

        spec_trackers = {}

        # Move model to device, then wrap with DDP for multi-GPU.
        # DDP correctly preserves requires_grad for frozen backbone + trainable LoRA.
        model = model.to(device)

        # Consolidate shared frozen weights across experts: model.to(device) creates
        # N independent GPU copies of each SharedLoRALayer.shared_weight. This call
        # makes experts 1..N-1 alias expert 0's buffers — (N-1)/N GPU memory saved.
        # Must happen AFTER .to(device) and BEFORE DDP/FSDP wrapping.
        _pre_wrap_moe_layers = getattr(model, "moe_layers", {})
        _trainable_base = cfg.expert.lora.get("trainable_base", False)
        for _ml in _pre_wrap_moe_layers.values():
            if hasattr(_ml, "expert_pool"):
                _ml.expert_pool.consolidate_shared_weights()
                if _trainable_base:
                    _ml.expert_pool.make_base_trainable()

        if is_distributed:
            model = wrap_model_for_distributed(
                model, cfg, local_rank, torch.device(device)
            )

        # Post-wrapping attribution: get clean references to MoE layers for metric tracking.
        _moe_layers_ref = getattr(get_model_for_attr_access(model), "moe_layers", {})
        _layer_name_to_layer = {f"layer_{k}": v for k, v in _moe_layers_ref.items()}

        if _moe_layers_ref and hasattr(model, "vocab_size") and model.vocab_size:
            for layer_idx_str in _moe_layers_ref:
                spec_trackers[f"layer_{layer_idx_str}"] = GlobalSpecializationTracker(
                    vocab_size=model.vocab_size,
                    num_experts=cfg.router.num_experts,
                    device="cpu",
                )

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        if is_main_process():
            print(f"Parameters: {trainable:,} trainable / {total:,} total")
            print("=" * 80)

        # Optimizer
        optimizer = build_optimizer(model, cfg)

        # LR Scheduler — warmup then cosine decay to 10% of peak LR
        from torch.optim.lr_scheduler import LambdaLR

        warmup_steps = cfg.training.get("warmup_steps", 0)
        max_steps_cfg = cfg.training.steps  # needed by lambda before loop var

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, max_steps_cfg - warmup_steps)
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = LambdaLR(optimizer, _lr_lambda)

        # Compile backbone only — MoE expert loops and router forward have dynamic
        # control flow (Python for-loop, requires_grad changes between train/eval)
        # that causes torch._dynamo to hit the recompile limit and fall back to
        # eager. Compiling only the backbone gets the speedup where it matters
        # (attention + frozen MLP matmuls) without fighting dynamo on the MoE path.
        if cfg.get("compile", False):
            print("Compiling model with torch.compile...")
            # layer_idx is an integer attr on Qwen2/GPT-Neo attention layers.
            # Without this, Dynamo recompiles once per layer (28x for Qwen2-1.5B),
            # hits the recompile limit, and falls back to eager for later layers.
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
            base = get_model_for_attr_access(model)
            if hasattr(base, "backbone"):
                base.backbone = torch.compile(base.backbone)
            else:
                model = torch.compile(model)

        # Resume
        start_step = 0
        best_val_loss = float("inf")
        early_stopping_patience = cfg.training.get("early_stopping_patience", None)
        steps_since_best = 0

        # Output dir — must be set before CheckpointManager
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_out = Path("outputs") / f"{cfg.experiment_name}_{timestamp}"
        out_dir = Path(args.output_dir) if args.output_dir else default_out
        out_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint manager (replaces inline save_checkpoint / load_checkpoint)
        from src.training.checkpoint import CheckpointManager

        ckpt_manager = CheckpointManager(
            checkpoint_dir=str(out_dir / "checkpoints"),
            keep_last_n=cfg.training.get("keep_last_n_checkpoints", 3),
            save_best=True,
            trainable_only=True,
        )

        if args.resume:
            ckpt_info = ckpt_manager.load_checkpoint(
                model,
                optimizer,
                scheduler,
                checkpoint_path=Path(args.resume),
            )
            start_step = ckpt_info["step"]
            best_val_loss = ckpt_info["metrics"].get("val_loss", float("inf"))
            # Advance scheduler to the correct step
            for _ in range(start_step):
                scheduler.step()
            print(f"Resumed from step {start_step}, best val_loss={best_val_loss:.4f}")

        # WandB
        init_wandb(cfg)

        # Data-driven prototype initialization (SPAR only, opt-in via router.init_from_data: true)
        if (
            cfg.router.get("init_from_data", False)
            and cfg.router.type == "stress_corrected"
        ):
            if is_main_process():
                print("Initializing SPAR router prototypes from data (k-means)...")
            _initialize_router_prototypes(model, train_loader, device, is_distributed)
            if is_main_process():
                print("Prototype initialization complete.")

        # Training config
        grad_accum = cfg.training.get("gradient_accumulation_steps", 1)
        log_interval = cfg.training.get("log_interval", 10)
        eval_interval = cfg.training.get("eval_interval", 100)
        save_interval = cfg.training.get("save_interval", 500)
        clip_norm = cfg.training.get("clip_grad_norm", 1.0)

        # Chinchilla-optimal steps — computed on all ranks so max_steps is consistent.
        # Uses N_trainable (LoRA + router prototypes only; frozen backbone excluded).
        # If training.steps is set in the YAML it overrides; otherwise Chinchilla is used.
        _global_batch = batch_size * grad_accum * world_size
        _tokens_per_step = _global_batch * cfg.dataset.max_seq_len
        from src.configs.model import model_lookup as _ml

        _model_info_chinchilla = _ml(cfg.model.model_key)
        _hidden = _model_info_chinchilla["hidden_dim"]
        _inter = _model_info_chinchilla.get("intermediate_dim", 4 * _hidden)
        _rank = cfg.expert.lora.rank
        _n_exp = cfg.router.num_experts
        _n_moe = len(cfg.model.moe_layer_indices)
        # GPT-Neo: 2 projections (c_fc, c_proj); Qwen2 SwiGLU: 3 (gate, up, down)
        _n_proj = 3 if cfg.expert.type == "qwen2_lora" else 2
        _lora_per_expert = _n_proj * (_rank * _hidden + _rank * _inter)
        trainable_params = (
            _lora_per_expert * _n_exp * _n_moe + _n_exp * _hidden * _n_moe
        )
        chinchilla_optimal_steps = max(
            1, math.ceil(20 * trainable_params / _tokens_per_step)
        )

        _steps_cfg = cfg.training.get("steps", None)
        max_steps = _steps_cfg if _steps_cfg is not None else chinchilla_optimal_steps

        # Precision — bf16 autocast on sm>=8 (H100/A100), fp16 + GradScaler otherwise
        from src.training.precision import (
            COMPUTE_DTYPE,
            needs_grad_scaler,
            is_mixed_precision,
        )

        use_grad_scaler = needs_grad_scaler() and "cuda" in device
        scaler = torch.amp.GradScaler("cuda", enabled=True) if use_grad_scaler else None
        _autocast_enabled = is_mixed_precision() and "cuda" in device
        _autocast_ctx = torch.amp.autocast(
            "cuda", dtype=COMPUTE_DTYPE, enabled=_autocast_enabled
        )

        model.train()
        train_iter = iter(train_loader)
        current_epoch = 0

        # State Initialization (Pre-loop)
        ema_beta = 0.9
        smooth_train_loss = 0.0
        total_training_time = 0.0
        moe_metrics = {}
        accum_loss = 0.0
        t0 = time.time()

        if is_main_process():
            _using_chinchilla = _steps_cfg is None
            _override_note = ""
            if not _using_chinchilla:
                _ratio = max(max_steps, chinchilla_optimal_steps) / min(
                    max_steps, chinchilla_optimal_steps
                )
                _dir = "OVER" if max_steps > chinchilla_optimal_steps else "UNDER"
                _override_note = (
                    f" | Config override: {max_steps} steps ({_dir} by {_ratio:.1f}×)"
                )
            print(
                f"Starting training: {max_steps} steps "
                f"({'Chinchilla-optimal' if _using_chinchilla else 'from config'}), "
                f"batch={batch_size}/gpu, grad_accum={grad_accum}, "
                f"global_batch={_global_batch} ({batch_size}×{grad_accum}×{world_size}gpus), "
                f"dtype={COMPUTE_DTYPE}"
            )
            print(
                f"Trainable params: {trainable_params:,} (LoRA+router, backbone frozen) | "
                f"Chinchilla-optimal: {chinchilla_optimal_steps} steps "
                f"({20 * trainable_params / 1e6:.1f}M tokens)" + _override_note
            )

        for step in range(start_step, max_steps):
            # Evaluate periodically — ALL ranks evaluate together to avoid
            # the rank-0-only pattern that deadlocks NCCL with num_workers>0.
            if step % eval_interval == 0 and val_loader is not None:
                base_model = get_model_for_attr_access(model)
                val_loss = evaluate(base_model, val_loader, device)
                if is_distributed:
                    import torch.distributed as dist

                    _vl = torch.tensor(val_loss, device=device)
                    dist.all_reduce(_vl, op=dist.ReduceOp.AVG)
                    val_loss = _vl.item()
                val_ppl = math.exp(min(val_loss, 20.0))
                val_bpb = val_loss / math.log(2)
                if is_main_process():
                    print(
                        f"--- [step {step:5d}] val_loss={val_loss:.4f} | val_ppl={val_ppl:.1f} | val_bpb={val_bpb:.3f} ---"
                    )
                log_wandb(
                    {
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                        "val_bpb": val_bpb,
                        "step": step,
                    }
                )

                # Clear cached routing tensors written by eval forwards so that
                # the training-step metrics collected immediately after eval
                # (when step % log_interval == 0 == step % eval_interval) reflect
                # the training forward, not the last val-batch forward.
                if _moe_layers_ref:
                    for _moe_lyr in _moe_layers_ref.values():
                        _moe_lyr._last_routing_weights = None
                        _moe_lyr._last_routing_indices = None

                # Reset Welford accumulators at each eval to prevent fp32
                # precision degradation when welford_n grows large (~10^8).
                # Welford is metrics-only — does not affect routing.
                if _moe_layers_ref:
                    for _moe_lyr in _moe_layers_ref.values():
                        if hasattr(_moe_lyr, "router") and hasattr(
                            _moe_lyr.router, "reset_welford"
                        ):
                            _moe_lyr.router.reset_welford()

                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    steps_since_best = 0
                else:
                    steps_since_best += eval_interval

                ckpt_manager.save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    step=step,
                    metrics={"val_loss": val_loss, "loss": val_loss},
                    is_best=is_best,
                )

                # Early stopping
                if (
                    early_stopping_patience is not None
                    and steps_since_best >= early_stopping_patience
                ):
                    if is_main_process():
                        print(
                            f"\n{'=' * 80}\n"
                            f"Early stopping triggered: no improvement for {steps_since_best} steps "
                            f"(patience={early_stopping_patience})\n"
                            f"Best val_loss: {best_val_loss:.4f} at step {step - steps_since_best}\n"
                            f"{'=' * 80}\n"
                        )
                    break

            # Periodic save (regardless of validation)
            elif step > 0 and step % save_interval == 0:
                ckpt_manager.save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    step=step,
                    metrics={"train_loss": accum_loss if step > start_step else 0.0},
                    is_best=False,
                )

            # Reset step timer HERE — after eval/checkpoint, before training.
            # This ensures dt measures only the training step, not eval overhead.
            t0 = time.time()

            # Gradient accumulation loop
            accum_loss = 0.0

            for i in range(grad_accum):
                # Optimization: only sync gradients on the last accumulation step.
                # This significantly reduces NCCL overhead and prevents timeouts.
                last_accum_step = i == grad_accum - 1

                context = (
                    model.no_sync()
                    if is_distributed and not last_accum_step
                    else contextlib.nullcontext()
                )

                with context:
                    try:
                        x, y = next(train_iter)
                    except StopIteration:
                        current_epoch += 1
                        if is_distributed and train_sampler is not None:
                            train_sampler.set_epoch(current_epoch)
                        train_iter = iter(train_loader)
                        x, y = next(train_iter)

                    x, y = (
                        x.to(device, non_blocking=True),
                        y.to(device, non_blocking=True),
                    )

                    # Only compute metrics on the last accumulation step at log intervals.
                    # spec_trackers only needs indices (included in metrics), not full stats.
                    do_metrics = (step % log_interval == 0) and last_accum_step
                    with _autocast_ctx:
                        _, loss, moe_metrics = model(
                            input_ids=x,
                            labels=y,
                            return_metrics=do_metrics,
                        )

                    loss_float = loss.item()  # capture float before division
                    loss = loss / grad_accum

                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    accum_loss += loss_float / grad_accum

                    # Update specialization trackers with token→expert routing
                    if spec_trackers and moe_metrics:
                        for layer_name, layer_m in moe_metrics.items():
                            if layer_name in spec_trackers and "indices" in layer_m:
                                spec_trackers[layer_name].update(
                                    token_ids=x.detach(),
                                    expert_indices=layer_m["indices"].detach(),
                                )

            # All-reduce training loss across ranks so rank 0 logs the global mean.
            if is_distributed:
                import torch.distributed as dist

                _loss_t = torch.tensor(accum_loss, device=device)
                dist.all_reduce(_loss_t, op=dist.ReduceOp.AVG)
                accum_loss = _loss_t.item()

            # Gradient clip + optimizer step
            if clip_norm > 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

            # Diagnostic A: per-expert gradient norms, logged at log_interval.
            # Run AFTER clip (norms reflect what enters the update) and BEFORE optimizer.step() (grads still live).
            _diag_a_metrics = {}
            if step % log_interval == 0 and is_main_process() and _moe_layers_ref:
                _base_m_diag = get_model_for_attr_access(model)
                for _layer_idx, _moe_layer in getattr(
                    _base_m_diag, "moe_layers", {}
                ).items():
                    # LoRA expert gradient norms — flatten all grads per expert into one
                    # tensor and call norm once; one .item() per expert total.
                    _pool = getattr(_moe_layer, "expert_pool", None)
                    if _pool is not None:
                        for _expert_idx, _expert in enumerate(_pool.experts):
                            _grads = [
                                _p.grad.data.flatten()
                                for _p in _expert.parameters()
                                if _p.grad is not None
                            ]
                            if _grads:
                                _norm_val = torch.cat(_grads).norm(2).item()
                            else:
                                _norm_val = 0.0
                            _diag_a_metrics[
                                f"grad_norm/layer{_layer_idx}/expert{_expert_idx}"
                            ] = _norm_val

                    # Router prototype gradient norm (W for StressCorrectedRouter,
                    # gate for MetabolicRouter). Critical for diagnosing prototype collapse.
                    _router_diag = getattr(_moe_layer, "router", None)
                    if _router_diag is not None:
                        _grads = [
                            _p.grad.data.flatten()
                            for _p in _router_diag.parameters()
                            if _p.grad is not None
                        ]
                        if _grads:
                            _norm_val = torch.cat(_grads).norm(2).item()
                        else:
                            _norm_val = 0.0
                        _diag_a_metrics[f"grad_norm/layer{_layer_idx}/router"] = (
                            _norm_val
                        )

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            # Release aux-loss tensors (_last_probs, _last_weights, _last_indices) held
            # by StandardRouter / DynMoERouter. These are stale after optimizer.step()
            # and would otherwise persist until the next forward pass overwrites them.
            # MetabolicRouter's clear_aux_state() is a inherited no-op, so safe for all.
            if _moe_layers_ref:
                for moe_layer in _moe_layers_ref.values():
                    if hasattr(moe_layer, "router"):
                        moe_layer.router.clear_aux_state()

            # Step MoE router fatigue state, then sync across DDP ranks.
            if _moe_layers_ref:
                for moe_layer in _moe_layers_ref.values():
                    if hasattr(moe_layer, "step"):
                        moe_layer.step()

                    # DDP SYNC: MetabolicRouter fatigue — AVG-reduce is correct here
                    # because fatigue is a fractional EMA (not a count/sum).
                    # StressCorrectedRouter handles its own sync inside router.step():
                    #   - ema_load: AVG via _sync_ema_load_distributed()
                    #   - welford_n/mu/M2: parallel Welford via _sync_welford_distributed()
                    #     (simple AVG is mathematically wrong for Welford statistics)
                    if is_distributed and hasattr(moe_layer, "router"):
                        import torch.distributed as dist

                        router = moe_layer.router
                        if hasattr(router, "fatigue"):
                            dist.all_reduce(router.fatigue, op=dist.ReduceOp.AVG)

            # Compute timing — t0 was set right before training forward pass above
            dt = time.time() - t0

            # Smooth the training loss
            smooth_train_loss = (
                ema_beta * smooth_train_loss + (1 - ema_beta) * accum_loss
            )
            steps_active = step - start_step + 1
            debiased_loss = smooth_train_loss / (1 - ema_beta**steps_active)

            # Throughput — global across all GPUs
            tok_per_step = batch_size * grad_accum * seq_len * world_size
            tokens_per_sec = tok_per_step / dt

            # Training Time / ETA
            # Start accumulating time after the first step for more accurate averages
            if steps_active > 1:
                total_training_time += dt
                avg_dt = total_training_time / (steps_active - 1)
                remaining_steps = max_steps - step - 1
                eta_seconds = remaining_steps * avg_dt
                eta_str = f"| eta: {eta_seconds / 60:.1f}m"
            else:
                eta_str = ""

            # Sync specialization trackers across all DDP ranks.
            # dist.all_reduce is a collective — must be called by ALL ranks before
            # entering the rank-0-only logging block below.
            _synced_spec: dict = {}
            if spec_trackers and step % log_interval == 0:
                for _tname, _tracker in spec_trackers.items():
                    _synced_spec[_tname] = _tracker.sync_and_compute(
                        device, is_distributed
                    )

            # Logging
            if (
                step % log_interval == 0 or step % eval_interval == 0
            ) and is_main_process():
                lr = scheduler.get_last_lr()[0]
                pct_done = 100 * (step + 1) / max_steps

                # Perplexity = exp(loss). Clamp to prevent inf for early unstable steps.
                train_ppl = math.exp(min(debiased_loss, 20.0))
                # BPB (bits per byte) ≈ loss / ln(2). Treats tokens ≈ bytes; consistent
                # across runs so comparisons are valid. log2(e) = 1/ln(2) ≈ 1.4427.
                train_bpb = debiased_loss / math.log(2)

                metrics = {
                    "step": step,
                    "train/loss": debiased_loss,
                    "train/ppl": train_ppl,
                    "train/bpb": train_bpb,
                    "train/lr": lr,
                    "train/dt": dt,
                    "train/tok_per_sec": tokens_per_sec,
                }

                # Diagnostic A: merge per-expert gradient norms (collected before optimizer step)
                metrics.update(_diag_a_metrics)

                # Diagnostic B: lambda trajectory logged every log_interval (dense).
                # Captures the calibration jump at lambda_calib_step — critical for paper figures.
                if step % log_interval == 0 and _moe_layers_ref:
                    _base_m_diag_b = get_model_for_attr_access(model)
                    for _layer_idx_b, _moe_layer_b in getattr(
                        _base_m_diag_b, "moe_layers", {}
                    ).items():
                        _router_b = getattr(_moe_layer_b, "router", None)
                        if _router_b is not None and hasattr(_router_b, "lambda_val"):
                            metrics[f"router/layer_{_layer_idx_b}/lambda_val"] = (
                                _router_b.lambda_val.item()
                            )

                # Add specific metabolic metrics securely
                _ROUTER_SCALAR_KEYS = (
                    "load_balance",
                    "fatigue_mean",
                    "fatigue_max",
                    "fatigue_min",
                    "fatigue_std",
                    "effective_experts",
                    "routing_diversity_gini",
                    "expert_entropy_normalized",
                    "router_confidence_mean",
                    "router_confidence_std",
                    "top1_dominance",
                    # MetabolicRouter v6 diagnostics
                    "lambda_eff",
                    "fatigue_tanh_mean",
                    "fairshare",
                    "fraction_penalised",
                    # StressCorrectedRouter diagnostics
                    "stress_mean",
                    "stress_max",
                    "stress_std",
                    "ema_load_mean",
                    "ema_load_max",
                    "ema_load_std",
                    "lambda_val",
                    "tau",
                    "welford_mu_mean",
                    "welford_n_min",
                    "eff_E_hard",
                    "mean_k",
                    "all_zero_frac",
                    # Paper metrics
                    "aux_loss",
                    "noise_std",
                    "sigma_cos_at_calib",
                )
                if moe_metrics:
                    for layer_name, layer_m in moe_metrics.items():
                        for key in _ROUTER_SCALAR_KEYS:
                            if key in layer_m:
                                metrics[f"router/{layer_name}/{key}"] = layer_m[key]

                        # Log v6 router state diagnostics (lambda_eff, fairshare, etc.)
                        _moe_layer = _layer_name_to_layer.get(layer_name)
                        if _moe_layer is not None:
                            _router = getattr(_moe_layer, "router", None)
                            if _router is not None and hasattr(_router, "get_state"):
                                _rstate = _router.get_state()
                                for _rk in (
                                    "lambda_eff",
                                    "fatigue_tanh_mean",
                                    "fairshare",
                                    "fraction_penalised",
                                ):
                                    if _rk in _rstate:
                                        metrics[f"router/{layer_name}/{_rk}"] = _rstate[
                                            _rk
                                        ]

                        if "usage_distribution" in layer_m:
                            usage_dist = layer_m["usage_distribution"]
                            if hasattr(usage_dist, "__len__"):
                                for expert_i, usage_val in enumerate(usage_dist):
                                    metrics[
                                        f"router/{layer_name}/expert_{expert_i}_usage"
                                    ] = float(usage_val)

                        if "ema_load_per_expert" in layer_m:
                            for expert_i, load_val in enumerate(
                                layer_m["ema_load_per_expert"]
                            ):
                                metrics[
                                    f"router/{layer_name}/expert_{expert_i}_load"
                                ] = float(load_val)

                        if "lora_delta_norm_per_expert" in layer_m:
                            for expert_i, norm_val in enumerate(
                                layer_m["lora_delta_norm_per_expert"]
                            ):
                                metrics[
                                    f"lora/{layer_name}/expert_{expert_i}_delta_norm"
                                ] = float(norm_val)

                # Global specialization metrics — already all-reduced above
                for tracker_name, spec_metrics in _synced_spec.items():
                    for k, v in spec_metrics.items():
                        if isinstance(v, (int, float)):
                            metrics[f"specialization/{tracker_name}/{k}"] = v

                log_str = (
                    f"step {step:05d}/{max_steps:05d} ({pct_done:.2f}%) | "
                    f"loss: {debiased_loss:.4f} | ppl: {train_ppl:.1f} | bpb: {train_bpb:.3f} | "
                    f"lr: {lr:.2e} | dt: {dt * 1000:.1f}ms | tok/sec: {tokens_per_sec:,.0f} | "
                    f"time: {total_training_time / 60:.2f}m {eta_str}"
                )

                # Append key router health metrics from the first MoE layer
                if moe_metrics:
                    first_layer = next(iter(moe_metrics.values()), {})
                    eff_e = first_layer.get("effective_experts")
                    gini = first_layer.get("routing_diversity_gini")
                    conf = first_layer.get("router_confidence_mean")
                    f_std = first_layer.get("fatigue_std")
                    router_parts = []
                    if eff_e is not None:
                        router_parts.append(f"eff_E: {eff_e:.1f}")
                    if gini is not None:
                        router_parts.append(f"gini: {gini:.3f}")
                    if conf is not None:
                        router_parts.append(f"conf: {conf:.3f}")
                    if f_std is not None:
                        router_parts.append(f"F_σ: {f_std:.3f}")

                    # Adaptive k logic
                    mean_k = first_layer.get("mean_k")
                    stress_mean = first_layer.get("stress_mean")
                    if mean_k is not None:
                        router_parts.append(f"mean_k: {mean_k:.2f}")
                    if stress_mean is not None:
                        router_parts.append(f"stress: {stress_mean:.3f}")
                    # v6 diagnostics: show warmup ramp + penalty activity
                    _first_router = None
                    _first_moe = next(iter(_layer_name_to_layer.values()), None)
                    if _first_moe is not None:
                        _first_router = getattr(_first_moe, "router", None)
                    if _first_router is not None and hasattr(
                        _first_router, "get_state"
                    ):
                        _rs = _first_router.get_state()
                        lam_eff = _rs.get("lambda_eff")
                        frac_pen = _rs.get("fraction_penalised")
                        if lam_eff is not None:
                            router_parts.append(f"λ_eff: {lam_eff:.3f}")
                        if frac_pen is not None:
                            router_parts.append(f"pen%: {frac_pen * 100:.0f}%")
                    if router_parts:
                        log_str += f" | {' | '.join(router_parts)}"

                # Diagnostic C: 100-step print for lambda/stress baseline
                if step < 100 and is_main_process():
                    _diag_c_parts = []
                    for _layer_idx_diag, _moe_layer_diag in _moe_layers_ref.items():
                        r = getattr(_moe_layer_diag, "router", None)
                        if r is not None and hasattr(r, "lambda_val"):
                            _s = (
                                r._welford_variance()
                                if hasattr(r, "_welford_variance")
                                else torch.zeros(1)
                            )
                            _diag_c_parts.append(
                                f"L{_layer_idx_diag}: λ={r.lambda_val.item():.3f} "
                                f"var={_s.mean().item():.3f} "
                                f"n_min={r.welford_n.min().item():.1f}"
                            )
                    if _diag_c_parts:
                        print(f"  [DIAG] {' | '.join(_diag_c_parts)}")

                print(log_str)
                log_wandb(metrics)

            # Re-synchronize all ranks after rank-0 logging.
            # Without this, rank-0 (busy with WandB/print) falls behind ranks 1-N
            # who immediately start the next training step. Over multiple log intervals
            # the DDP gradient all-reduce sequence numbers diverge → NCCL timeout.
            # The metabolic router avoids this via its per-step fatigue all-reduce
            # (an implicit barrier). Routers without per-step collectives (standard,
            # topk, dynmoe) need this explicit barrier.
            if is_distributed and step % log_interval == 0:
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.barrier()

        # Final save
        print("\nTraining complete.")
        if val_loader is not None:
            base_model = get_model_for_attr_access(model)
            val_loss = evaluate(base_model, val_loader, device)
            if is_distributed:
                import torch.distributed as dist

                _vl = torch.tensor(val_loss, device=device)
                dist.all_reduce(_vl, op=dist.ReduceOp.AVG)
                val_loss = _vl.item()
            val_ppl = math.exp(min(val_loss, 20.0))
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            ckpt_manager.save_checkpoint(
                model,
                optimizer,
                scheduler,
                step=max_steps,
                metrics={"val_loss": val_loss, "val_ppl": val_ppl, "loss": val_loss},
                is_best=is_best,
            )
            _final_bpb = val_loss / math.log(2)
            log_wandb(
                {
                    "val_loss": val_loss,
                    "val_ppl": val_ppl,
                    "val_bpb": _final_bpb,
                    "step": max_steps,
                }
            )
            print(
                f"Final val_loss={val_loss:.4f} | val_ppl={val_ppl:.1f} | val_bpb={_final_bpb:.3f}"
            )

        print(f"Outputs saved to: {out_dir}")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
