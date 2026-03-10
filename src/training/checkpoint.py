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
            f"[checkpoint] {label} missing keys ({len(result.missing_keys)}): "
            f"{result.missing_keys[:5]}..."
        )
    if result.unexpected_keys:
        print(
            f"[checkpoint] {label} unexpected keys ({len(result.unexpected_keys)}): "
            f"{result.unexpected_keys[:5]}..."
        )


# Router state buffers that must survive checkpointing for correct resume.
# Excludes: _pending_usage_sum / _pending_tokens (transient, zero after step()).
_ROUTER_STATE_BUFFERS = frozenset({"fatigue", "num_steps"})


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
            # All ranks call _get_state_dict; non-rank-0 get empty dict (rank0_only=True)
            model_state_dict = _get_state_dict(model)
            if not is_main_process():
                # Sync so rank 0 finishes saving before training resumes
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.barrier()
                return Path("/dev/null")
        else:
            if not is_main_process():
                # Block until rank 0 finishes saving, then return.
                # Without this barrier, non-rank-0 ranks race into the next
                # training step while rank 0 is doing checkpoint I/O, causing
                # DDP gradient all-reduce sequence numbers to diverge (NCCL timeout).
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

        # All strategies: barrier so non-rank-0 processes wait until rank 0
        # finishes saving before training resumes.
        # FSDP: non-rank-0 waited above (after all-gather), barriers here to release.
        # DDP: non-rank-0 blocked in the barrier above; rank 0 signals completion here.
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

        from torch.nn.parallel import DistributedDataParallel as DDP
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        if isinstance(model, DDP):
            # Load into underlying module — DDP holds full params on every rank
            _log_state_dict_result(
                model.module.load_state_dict(
                    checkpoint["model_state_dict"], strict=False
                ),
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
                checkpoint["model_state_dict"],
                options=StateDictOptions(
                    full_state_dict=True, cpu_offload=True, strict=False
                ),
            )
        else:
            _log_state_dict_result(
                model.load_state_dict(checkpoint["model_state_dict"], strict=False),
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
