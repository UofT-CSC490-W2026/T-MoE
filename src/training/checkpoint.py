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


def _align_orig_mod(state_dict: dict, target: nn.Module) -> dict:
    """
    Normalize _orig_mod presence in checkpoint keys to match the live model.

    torch.compile wraps a submodule (backbone) with OptimizedModule, inserting
    '._orig_mod.' into its state_dict keys.  A checkpoint saved from a compiled
    run has those keys; one saved from a non-compiled run does not.  Either way
    we want a clean load with no spurious missing/unexpected keys.

    Strategy: detect the mismatch by comparing a sample key from each side, then
    rewrite the checkpoint keys in one pass.
    """
    ckpt_keys = list(state_dict.keys())
    if not ckpt_keys:
        return state_dict

    target_keys = set(target.state_dict().keys())
    if not target_keys:
        return state_dict

    ckpt_has_orig = any("._orig_mod." in k for k in ckpt_keys)
    model_has_orig = any("._orig_mod." in k for k in target_keys)

    if ckpt_has_orig == model_has_orig:
        return state_dict

    if model_has_orig and not ckpt_has_orig:
        prefix_map: dict[str, str] = {}
        for k in target_keys:
            idx = k.find("._orig_mod.")
            if idx != -1:
                before = k[:idx]
                after = k[idx + len("._orig_mod.") :]
                prefix_map[before] = after
        insertion_prefixes = sorted(prefix_map.keys(), key=len, reverse=True)

        def _add(k: str) -> str:
            for pfx in insertion_prefixes:
                dot_pfx = pfx + "."
                if k.startswith(dot_pfx):
                    return pfx + "._orig_mod." + k[len(dot_pfx) :]
            return k

        return {_add(k): v for k, v in state_dict.items()}

    else:
        return {k.replace("._orig_mod.", "."): v for k, v in state_dict.items()}


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
        "fatigue",  # MetabolicRouter
        "num_steps",  # shared
        "ema_load",  # SPAR: losing these resets load tracking and disables
        "lambda_val",  # the calibrated penalty for the remainder of training
        "lambda_initialized",
        "welford_n",
        "welford_mu",
        "welford_M2",
    }
)


def _get_state_dict(model: nn.Module) -> dict:
    """Return state dict for DDP (rank-0 direct), FSDP (collective all-gather), or plain model."""
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

        self.checkpoints = []
        self.best_metric = float("inf")
        self.best_checkpoint_path = None

    def _write_checkpoint_files(
        self, checkpoint: dict, step: int, metrics: dict, metadata: dict
    ) -> Path:
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        temp_path = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temp_path)
        temp_path.rename(checkpoint_path)

        with open(self.checkpoint_dir / f"checkpoint_step_{step}.json", "w") as f:
            json.dump(
                {
                    "step": step,
                    "metrics": _serialize_metrics(metrics),
                    "metadata": metadata,
                },
                f,
                indent=2,
            )

        return checkpoint_path

    def _write_best_model(self, checkpoint: dict, step: int, metrics: dict) -> None:
        best_path = self.checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        self.best_checkpoint_path = best_path
        self.best_metric = metrics.get("loss", float("inf"))
        with open(self.checkpoint_dir / "best_model.json", "w") as f:
            json.dump(
                {"step": step, "metrics": _serialize_metrics(metrics)}, f, indent=2
            )

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
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        is_fsdp = isinstance(model, FSDP)

        if is_fsdp:
            # All ranks must call _get_state_dict (collective all-gather).
            model_state_dict = _get_state_dict(model)
            if not is_main_process():
                # Non-rank-0 must hit this barrier — returning early before it
                # caused a barrier mismatch that deadlocked rank 0.
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.barrier()
                return Path("/dev/null")
        else:
            if not is_main_process():
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.barrier()
                return Path("/dev/null")
            model_state_dict = _get_state_dict(model)

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

        metadata = metadata or {}
        checkpoint = {
            "step": step,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "metadata": metadata,
        }
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        checkpoint_path = self._write_checkpoint_files(
            checkpoint, step, metrics, metadata
        )
        current_metric = metrics.get("loss", float("inf"))
        self.checkpoints.append((step, checkpoint_path, current_metric))

        if is_best and self.save_best:
            self._write_best_model(checkpoint, step, metrics)

        self._cleanup_old_checkpoints()

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
            aligned = _align_orig_mod(model_state_dict, model.module)
            _log_state_dict_result(
                model.module.load_state_dict(aligned, strict=False),
                "DDP",
            )
        elif isinstance(model, FSDP):
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
            aligned = _align_orig_mod(model_state_dict, model)
            _log_state_dict_result(
                model.load_state_dict(aligned, strict=False),
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
