"""
Checkpointing utilities for T-MoE training.

Provides professional checkpoint management with automatic cleanup,
best checkpoint tracking, and robust save/load functionality.
"""
import torch
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime


class CheckpointManager:
    """
    Manages model checkpointing with automatic cleanup and best checkpoint tracking.

    Features:
    - Save/load checkpoints with full training state
    - Keep last N checkpoints automatically
    - Track best checkpoint by metric
    - Atomic writes (temp file + rename) for safety
    - Metadata tracking for reproducibility

    Usage:
        manager = CheckpointManager("./checkpoints", keep_last_n=3)
        manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            step=1000,
            metrics={"loss": 0.5},
            is_best=True
        )
        state = manager.load_latest_checkpoint()
    """

    def __init__(
        self,
        checkpoint_dir: str,
        keep_last_n: int = 3,
        keep_best_n: int = 1,
    ):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
            keep_best_n: Number of best checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.keep_best_n = keep_best_n

        # Subdirectories
        self.regular_dir = self.checkpoint_dir / "regular"
        self.best_dir = self.checkpoint_dir / "best"
        self.regular_dir.mkdir(exist_ok=True)
        self.best_dir.mkdir(exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        metrics: Dict[str, float],
        config: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        **extra_state,
    ) -> Path:
        """
        Save a complete training checkpoint.

        Args:
            model: Model to checkpoint
            optimizer: Optimizer to checkpoint
            step: Current training step
            metrics: Current metrics dict
            config: Optional config dict for reproducibility
            is_best: Whether this is the best checkpoint
            **extra_state: Additional state to save (e.g., scheduler, scaler)

        Returns:
            Path to saved checkpoint
        """
        # Prepare checkpoint dictionary
        checkpoint = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        if config is not None:
            checkpoint["config"] = config

        # Add extra state
        checkpoint.update(extra_state)

        # Determine save path
        filename = f"checkpoint_step_{step}.pt"
        if is_best:
            save_path = self.best_dir / filename
        else:
            save_path = self.regular_dir / filename

        # Atomic write: save to temp file, then rename
        temp_path = save_path.with_suffix(".pt.tmp")
        torch.save(checkpoint, temp_path)
        temp_path.rename(save_path)

        # Save metadata as JSON for easy inspection
        metadata_path = save_path.with_suffix(".json")
        metadata = {
            "step": step,
            "metrics": metrics,
            "timestamp": checkpoint["timestamp"],
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Cleanup old checkpoints
        if not is_best:
            self._cleanup_old_checkpoints(self.regular_dir, self.keep_last_n)
        else:
            self._cleanup_old_checkpoints(self.best_dir, self.keep_best_n)

        return save_path

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint from path.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            map_location: Device to map tensors to

        Returns:
            Complete checkpoint dictionary
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state if provided
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint

    def load_latest_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            map_location: Device to map tensors to

        Returns:
            Checkpoint dict or None if no checkpoints exist
        """
        checkpoints = self._get_checkpoints(self.regular_dir)

        if not checkpoints:
            return None

        latest = checkpoints[-1]
        return self.load_checkpoint(latest, model, optimizer, map_location)

    def load_best_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[str] = None,
        metric_name: str = "loss",
        minimize: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Load the best checkpoint by metric.

        Args:
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            map_location: Device to map tensors to
            metric_name: Metric to compare (default: 'loss')
            minimize: Whether lower is better (default: True)

        Returns:
            Checkpoint dict or None if no checkpoints exist
        """
        checkpoints = self._get_checkpoints(self.best_dir)

        if not checkpoints:
            return None

        # Load metadata to find best
        best_checkpoint = None
        best_metric = float("inf") if minimize else float("-inf")

        for ckpt_path in checkpoints:
            metadata_path = ckpt_path.with_suffix(".json")
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    metric_value = metadata.get("metrics", {}).get(metric_name)

                    if metric_value is not None:
                        is_better = (
                            (metric_value < best_metric)
                            if minimize
                            else (metric_value > best_metric)
                        )
                        if is_better:
                            best_metric = metric_value
                            best_checkpoint = ckpt_path

        if best_checkpoint is None:
            # Fallback to latest if no metadata
            best_checkpoint = checkpoints[-1]

        return self.load_checkpoint(best_checkpoint, model, optimizer, map_location)

    def _get_checkpoints(self, directory: Path) -> List[Path]:
        """Get sorted list of checkpoint files."""
        checkpoints = sorted(
            directory.glob("checkpoint_step_*.pt"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        return checkpoints

    def _cleanup_old_checkpoints(self, directory: Path, keep_n: int):
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = self._get_checkpoints(directory)

        if len(checkpoints) <= keep_n:
            return

        # Remove oldest checkpoints
        for ckpt_path in checkpoints[:-keep_n]:
            ckpt_path.unlink()
            # Also remove metadata
            metadata_path = ckpt_path.with_suffix(".json")
            if metadata_path.exists():
                metadata_path.unlink()

    def get_checkpoint_info(self) -> Dict[str, Any]:
        """Get information about available checkpoints."""
        regular_checkpoints = self._get_checkpoints(self.regular_dir)
        best_checkpoints = self._get_checkpoints(self.best_dir)

        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "num_regular": len(regular_checkpoints),
            "num_best": len(best_checkpoints),
            "latest_step": int(regular_checkpoints[-1].stem.split("_")[-1])
            if regular_checkpoints
            else None,
        }
