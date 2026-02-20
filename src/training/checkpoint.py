import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch import nn

from src.project_types import ExecutionEnv


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
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of most recent checkpoints to keep
            save_best: Whether to save best model separately
            execution_env: ExecutionEnv.LOCAL or ExecutionEnv.AWS
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last_n = keep_last_n
        self.save_best = save_best
        self.execution_env = execution_env

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

        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "metadata": metadata or {},
        }

        # Save scheduler state if provided
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        # Save regular checkpoint with atomic write
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        temp_path = checkpoint_path.with_suffix(".pt.tmp")

        torch.save(checkpoint, temp_path)
        temp_path.rename(checkpoint_path)  # Atomic operation

        # Save metadata as JSON for easy inspection
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

        # Save best model if applicable
        if is_best and self.save_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.best_checkpoint_path = best_path
            self.best_metric = current_metric

            # Save best metadata
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

        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()

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

        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        model.load_state_dict(checkpoint["model_state_dict"])

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
