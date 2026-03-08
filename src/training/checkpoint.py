import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch import nn

from src.project_types import ExecutionEnv
from src.training.fsdp_utils import is_main_process


def _get_base_model(model: nn.Module) -> nn.Module:
    """Unwrap DDP/FSDP to reach the underlying model for parameter name inspection."""
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(model, DDP):
        return model.module
    if isinstance(model, FSDP):
        return model._fsdp_wrapped_module
    return model


# Router state buffers that must survive checkpointing for correct resume.
# Excludes: expert_ids (constant arange), hardware_distance (constant zeros),
#           _pending_usage_sum / _pending_tokens (transient, zero after step()).
_ROUTER_STATE_BUFFERS = frozenset({"fatigue", "birth_step", "num_steps", "n_active"})


def _get_state_dict(model: nn.Module) -> dict:
    """
    Return state dict for DDP, FSDP, or plain model.

    DDP  — model.module holds full params on every rank → rank 0 saves directly.
    FSDP — ALL ranks must call this (triggers all-gather). Uses deprecated
           FSDP.state_dict_type() which is stable in PyTorch 2.x despite the
           FutureWarning. The modern get_state_dict() API deadlocks in some configs.
    Plain — model.state_dict() as normal.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if isinstance(model, DDP):
        return model.module.state_dict()

    if isinstance(model, FSDP):
        import warnings
        from torch.distributed.fsdp import StateDictType, FullStateDictConfig

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                state_dict_config=FullStateDictConfig(
                    offload_to_cpu=True, rank0_only=True
                ),
            ):
                return model.state_dict()

    return model.state_dict()


class CheckpointManager:
    """
    Manages model checkpoints with support for best model, periodic saves, and resumption.

    Supports both local filesystem and S3 (prepared for future integration).
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 3,
        save_best: bool = True,
        execution_env: ExecutionEnv = ExecutionEnv.LOCAL,
        trainable_only: bool = False,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of most recent checkpoints to keep
            save_best: Whether to save best model separately
            execution_env: ExecutionEnv.LOCAL or ExecutionEnv.AWS
            trainable_only: If True, save only LoRA and router parameters
                            (skips frozen backbone weights → smaller checkpoints)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.execution_env = execution_env
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
        """
        Save checkpoint with atomic write for crash safety.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Optional LR scheduler state
            step: Training step
            metrics: Current metrics
            is_best: Whether this is the best model
            metadata: Additional metadata to save

        Returns:
            Path to saved checkpoint
        """
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
                return Path("/dev/null")  # DDP/plain: non-rank-0 skip entirely
            model_state_dict = _get_state_dict(model)

        # Filter to trainable params only (smaller checkpoint files)
        if self.trainable_only:
            base_model = _get_base_model(model)
            trainable_keys = {
                k for k, p in base_model.named_parameters() if p.requires_grad
            }
            model_state_dict = {
                k: v
                for k, v in model_state_dict.items()
                if k in trainable_keys or any(buf in k for buf in _ROUTER_STATE_BUFFERS)
            }

        # Save checkpoint files (rank 0 only — non-rank-0 returned early above)
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

        # Save metadata JSON for easy inspection
        metadata_path = self.checkpoint_dir / f"checkpoint_step_{step}.json"
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "step": step,
                    "metrics": {
                        k: float(v) if isinstance(v, (int, float)) else v
                        for k, v in metrics.items()
                    },
                    "metadata": metadata or {},
                },
                f,
                indent=2,
            )

        # Track this checkpoint
        current_metric = metrics.get("loss", float("inf"))
        self.checkpoints.append((step, checkpoint_path, current_metric))

        # Save best model
        if is_best and self.save_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.best_checkpoint_path = best_path
            self.best_metric = current_metric

            with open(self.checkpoint_dir / "best_model.json", "w") as f:
                json.dump(
                    {
                        "step": step,
                        "metrics": {
                            k: float(v) if isinstance(v, (int, float)) else v
                            for k, v in metrics.items()
                        },
                    },
                    f,
                    indent=2,
                )

        self._cleanup_old_checkpoints()

        # FSDP: barrier so non-rank-0 processes wait until rank 0 finishes saving
        if is_fsdp:
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
        """
        Load checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            scheduler: LR scheduler to load state into (optional)
            checkpoint_path: Specific checkpoint to load (if None, loads latest)
            load_best: Load best model instead of latest

        Returns:
            Checkpoint metadata (step, metrics, etc.)
        """
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
            model.module.load_state_dict(checkpoint["model_state_dict"], strict=False)
        elif isinstance(model, FSDP):
            # All ranks must participate in FSDP load
            import warnings
            from torch.distributed.fsdp import StateDictType, FullStateDictConfig

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                with FSDP.state_dict_type(
                    model,
                    StateDictType.FULL_STATE_DICT,
                    state_dict_config=FullStateDictConfig(
                        offload_to_cpu=True, rank0_only=True
                    ),
                ):
                    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

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
        """Get the most recent checkpoint."""
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
        """Remove old checkpoints, keeping only the last N."""
        if self.keep_last_n <= 0:
            return

        # Sort by step
        self.checkpoints.sort(key=lambda x: x[0])

        # Remove old checkpoints
        while len(self.checkpoints) > self.keep_last_n:
            step, path, _ = self.checkpoints.pop(0)

            # Delete checkpoint file
            if path.exists():
                path.unlink()

            # Delete metadata file
            metadata_path = path.parent / f"{path.stem}.json"
            if metadata_path.exists():
                metadata_path.unlink()

    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        return [
            {"step": step, "path": str(path), "metric": metric}
            for step, path, metric in self.checkpoints
        ]
