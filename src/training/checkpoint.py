import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch import nn

from src.training.fsdp_utils import is_main_process, get_model_for_attr_access


def _serialize_metrics(metrics: dict) -> dict:
    return {
        k: float(v) if isinstance(v, (int, float)) else v for k, v in metrics.items()
    }


def _log_state_dict_result(result, label: str) -> None:
    if not is_main_process():
        return
    if result.missing_keys:
        print(
            f"[checkpoint] {label} missing keys ({len(result.missing_keys)}): {result.missing_keys[:5]}..."
        )
    if result.unexpected_keys:
        print(
            f"[checkpoint] {label} unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]}..."
        )


def _remap_legacy_moe_key(key: str) -> str | None:
    """
    Translate older MoE checkpoint keys into the current injected-backbone layout.

    Older checkpoints stored trainable MoE weights under:
      moe_layers.{layer}.router...
      moe_layers.{layer}.experts.{expert}.fc1/fc2...

    The current model stores them under:
      backbone.transformer.h.{layer}.mlp.router...
      backbone.transformer.h.{layer}.mlp.expert_pool.experts.{expert}.c_fc/c_proj...

    Legacy frozen base_weight/base_bias buffers are intentionally dropped because
    the current SharedLoRALayer reconstructs them from the pretrained MLP and they
    are non-persistent in state_dict.
    """

    def _map_expert_suffix(
        prefix_parts: list[str], suffix_parts: list[str]
    ) -> str | None:
        if len(suffix_parts) < 3:
            return ".".join(prefix_parts + suffix_parts)

        expert_idx, legacy_block, *tail = suffix_parts
        if tail and tail[0] in {"base_weight", "base_bias"}:
            return None

        block_map = {"fc1": "c_fc", "fc2": "c_proj"}
        mapped_block = block_map.get(legacy_block)
        if mapped_block is None:
            return ".".join(prefix_parts + suffix_parts)

        return ".".join(prefix_parts + [expert_idx, mapped_block] + tail)

    if key.startswith("moe_layers."):
        parts = key.split(".")
        if len(parts) < 4:
            return key

        _, layer_idx, section, *rest = parts

        if section == "router":
            return ".".join(
                ["backbone", "transformer", "h", layer_idx, "mlp", "router"] + rest
            )

        if section == "experts":
            return _map_expert_suffix(
                [
                    "backbone",
                    "transformer",
                    "h",
                    layer_idx,
                    "mlp",
                    "expert_pool",
                    "experts",
                ],
                rest,
            )

    if ".mlp.experts." in key:
        prefix, suffix = key.split(".mlp.experts.", maxsplit=1)
        mapped = _map_expert_suffix(
            prefix.split(".") + ["mlp", "expert_pool", "experts"],
            suffix.split("."),
        )
        return mapped

    return key


def _remap_legacy_moe_state_dict(state_dict: dict) -> tuple[dict, bool]:
    remapped = {}
    changed = False
    for key, value in state_dict.items():
        mapped_key = _remap_legacy_moe_key(key)
        if mapped_key is None:
            changed = True
            continue
        if mapped_key != key:
            changed = True
        remapped[mapped_key] = value
    return remapped, changed


# Router state buffers that must survive checkpointing for correct resume.
# Transient accumulators (_pending_*) are excluded — they're zero after step().
_ROUTER_STATE_BUFFERS = frozenset(
    {
        # MetabolicRouter
        "fatigue",
        # Shared
        "num_steps",
        # StressCorrectedRouter (SPAR) — ema_load and lambda_val define routing
        # behaviour at resume time; losing them resets load tracking and disables
        # the calibrated penalty for the remainder of training.
        "ema_load",
        "lambda_val",
        "lambda_initialized",
        "welford_n",
        "welford_mu",
        "welford_M2",
    }
)


def _get_state_dict(model: nn.Module) -> dict:
    """
    Return state dict for DDP, FSDP, or plain model.

    DDP   — model.module holds full params on every rank → rank 0 saves directly.
    FSDP  — ALL ranks must call this (collective all-gather). Uses the modern
            torch.distributed.checkpoint.state_dict API (PyTorch 2.3+).
    Plain — model.state_dict() as normal.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(model, DDP):
        return model.module.state_dict()

    if isinstance(model, FSDP):
        from torch.distributed.checkpoint.state_dict import (
            get_model_state_dict,
            StateDictOptions,
        )

        return get_model_state_dict(
            model,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

    return model.state_dict()


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 3,
        save_best: bool = True,
        trainable_only: bool = False,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.trainable_only = trainable_only

        self.checkpoints = []  # List of (step, path, metric) tuples
        self.best_metric = float("inf")
        self.best_checkpoint_path = None

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        step: int = 0,
        metrics: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        metrics = metrics or {}

        # FSDP: ALL ranks must participate in state dict gathering (all-gather op).
        # DDP/plain: only rank 0 needs to do anything.
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        is_fsdp = isinstance(model, FSDP)

        if is_fsdp:
            # FSDP: ALL ranks must call _get_state_dict (collective all-gather).
            model_state_dict = _get_state_dict(model)
            if not is_main_process():
                # Non-rank-0: skip save, but MUST hit the barrier below so rank 0
                # doesn't deadlock waiting. Previous code returned here, causing a
                # barrier mismatch (non-rank-0 hit barrier#1 and left; rank-0 hit
                # barrier#2 alone → deadlock).
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.barrier()
                return Path("/dev/null")
        else:
            if not is_main_process():
                # DDP: non-rank-0 waits for rank 0 to finish saving.
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.barrier()
                return Path("/dev/null")
            model_state_dict = _get_state_dict(model)

        # Only rank 0 reaches here.
        if self.trainable_only:
            base_model = get_model_for_attr_access(model)
            trainable_keys = {
                k for k, p in base_model.named_parameters() if p.requires_grad
            }
            model_state_dict = {
                k: v
                for k, v in model_state_dict.items()
                if k in trainable_keys or any(buf in k for buf in _ROUTER_STATE_BUFFERS)
            }

        checkpoint = {
            "step": step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "metadata": metadata or {},
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Atomic write
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        temp_path = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temp_path)
        temp_path.rename(checkpoint_path)

        meta_payload = {
            "step": step,
            "metrics": _serialize_metrics(metrics),
            "metadata": metadata or {},
        }
        with open(self.checkpoint_dir / f"checkpoint_step_{step}.json", "w") as f:
            json.dump(meta_payload, f, indent=2)

        current_metric = metrics.get("loss", float("inf"))
        self.checkpoints.append((step, checkpoint_path, current_metric))

        if is_best and self.save_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.best_checkpoint_path = best_path
            self.best_metric = current_metric
            with open(self.checkpoint_dir / "best_model.json", "w") as f:
                json.dump(
                    {"step": step, "metrics": _serialize_metrics(metrics)}, f, indent=2
                )

        self._cleanup_old_checkpoints()

        # Rank 0 signals completion — releases non-rank-0 processes waiting above.
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()

        return checkpoint_path

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        checkpoint_path: Optional[Path] = None,
        load_best: bool = False,
    ) -> Dict[str, Any]:
        if load_best:
            checkpoint_path = (
                self.best_checkpoint_path or self.checkpoint_dir / "best_model.pt"
            )
        elif checkpoint_path is None:
            checkpoint_path = self._get_latest_checkpoint()

        if not checkpoint_path or not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state_dict, remapped_legacy_keys = _remap_legacy_moe_state_dict(
            checkpoint["model_state_dict"]
        )
        if remapped_legacy_keys and is_main_process():
            print("[checkpoint] remapped legacy MoE checkpoint keys for compatibility")

        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(model, DDP):
            # Load into underlying module — DDP holds full params on every rank
            _log_state_dict_result(
                model.module.load_state_dict(model_state_dict, strict=False),
                "DDP",
            )
        elif isinstance(model, FSDP):
            # All ranks must participate in FSDP load
            from torch.distributed.checkpoint.state_dict import (
                set_model_state_dict,
                StateDictOptions,
            )

            set_model_state_dict(
                model,
                model_state_dict,
                options=StateDictOptions(
                    full_state_dict=True, cpu_offload=True, strict=False
                ),
            )
        else:
            _log_state_dict_result(
                model.load_state_dict(model_state_dict, strict=False),
                "plain",
            )

        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return {
            "step": checkpoint["step"],
            "metrics": checkpoint["metrics"],
            "metadata": checkpoint.get("metadata", {}),
        }

    def _get_latest_checkpoint(self) -> Optional[Path]:
        if not self.checkpoints:
            # Try to find checkpoints in directory
            checkpoints = sorted(
                self.checkpoint_dir.glob("checkpoint_step_*.pt"),
                key=lambda p: int(p.stem.split("_")[-1]),
            )
            if checkpoints:
                return checkpoints[-1]
            return None

        return self.checkpoints[-1][1]

    def _cleanup_old_checkpoints(self) -> None:
        if self.keep_last_n <= 0:
            return

        self.checkpoints.sort(key=lambda x: x[0])

        while len(self.checkpoints) > self.keep_last_n:
            step, path, _ = self.checkpoints.pop(0)
            if path.exists():
                path.unlink()
            metadata_path = path.parent / f"{path.stem}.json"
            if metadata_path.exists():
                metadata_path.unlink()

    def list_checkpoints(self) -> list:
        return [
            {"step": step, "path": str(path), "metric": metric}
            for step, path, metric in self.checkpoints
        ]
