import time
from typing import Dict, Any, Optional
from collections import deque

import numpy as np

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class TrainingMetricsTracker:
    """
    Tracks comprehensive training metrics including loss, perplexity, throughput.

    Designed for research reproducibility with detailed step-level tracking.
    """

    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.

        Args:
            window_size: Window size for moving average computation
        """
        self.window_size = window_size

        # Loss tracking
        self.losses = deque(maxlen=window_size)
        self.cumulative_loss = 0.0
        self.num_steps = 0

        # Throughput tracking
        self.start_time = time.time()
        self.step_times = deque(maxlen=window_size)
        self.tokens_processed = 0

        # Best metrics for checkpointing
        self.best_loss = float("inf")
        self.best_perplexity = float("inf")
        self.best_step = 0

    def update(
        self,
        loss: float,
        batch_size: int,
        seq_len: int,
        step_time: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Update metrics with new training step.

        Args:
            loss: Loss value for current step
            batch_size: Batch size
            seq_len: Sequence length
            step_time: Time taken for step (seconds)

        Returns:
            Dictionary of computed metrics
        """
        self.num_steps += 1
        self.losses.append(loss)
        self.cumulative_loss += loss

        num_tokens = batch_size * seq_len
        self.tokens_processed += num_tokens

        if step_time is not None:
            self.step_times.append(step_time)

        # Compute metrics
        metrics = self._compute_metrics(loss)

        # Update best metrics
        if metrics["loss"] < self.best_loss:
            self.best_loss = metrics["loss"]
            self.best_perplexity = metrics["perplexity"]
            self.best_step = self.num_steps

        return metrics

    def _compute_metrics(self, current_loss: float) -> Dict[str, float]:
        """Compute all training metrics."""
        metrics = {}

        # Loss metrics
        metrics["loss"] = current_loss
        metrics["avg_loss"] = np.mean(self.losses) if self.losses else current_loss
        metrics["cumulative_avg_loss"] = self.cumulative_loss / self.num_steps

        # Perplexity (clip loss to prevent overflow: exp(100) ≈ 2.7e43)
        # Loss values above 100 indicate severe training issues anyway
        clipped_current = np.clip(current_loss, None, 100.0)
        clipped_avg = np.clip(metrics["avg_loss"], None, 100.0)
        metrics["perplexity"] = np.exp(clipped_current)
        metrics["avg_perplexity"] = np.exp(clipped_avg)

        # Throughput
        elapsed = time.time() - self.start_time
        metrics["tokens_per_sec"] = (
            self.tokens_processed / elapsed if elapsed > 0 else 0
        )

        if self.step_times:
            avg_step_time = np.mean(self.step_times)
            metrics["avg_step_time"] = avg_step_time
            metrics["steps_per_sec"] = 1 / avg_step_time if avg_step_time > 0 else 0

        # Best metrics
        metrics["best_loss"] = self.best_loss
        metrics["best_perplexity"] = self.best_perplexity
        metrics["best_step"] = self.best_step

        # Training progress
        metrics["num_steps"] = self.num_steps
        metrics["total_tokens"] = self.tokens_processed

        return metrics

    def should_save_checkpoint(self, loss: float, mode: str = "best") -> bool:
        """
        Determine if checkpoint should be saved.

        Args:
            loss: Current loss value
            mode: 'best' or 'periodic'

        Returns:
            True if checkpoint should be saved
        """
        if mode == "best":
            return loss < self.best_loss
        return False

    def get_state(self) -> Dict[str, Any]:
        """Get tracker state for checkpointing."""
        return {
            "num_steps": self.num_steps,
            "cumulative_loss": self.cumulative_loss,
            "tokens_processed": self.tokens_processed,
            "best_loss": self.best_loss,
            "best_perplexity": self.best_perplexity,
            "best_step": self.best_step,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Load tracker state from checkpoint."""
        self.num_steps = state["num_steps"]
        self.cumulative_loss = state["cumulative_loss"]
        self.tokens_processed = state["tokens_processed"]
        self.best_loss = state["best_loss"]
        self.best_perplexity = state["best_perplexity"]
        self.best_step = state["best_step"]

    def log_to_wandb(
        self,
        metrics: Dict[str, Any],
        step: int,
        prefix: str = "train",
    ) -> None:
        """
        Log metrics to Weights & Biases.

        Args:
            metrics: Metrics dictionary
            step: Training step
            prefix: Metric prefix
        """
        if not WANDB_AVAILABLE or not wandb.run:
            return

        wandb_metrics = {
            f"{prefix}/{k}": v
            for k, v in metrics.items()
            if isinstance(v, (int, float, np.number))
        }
        wandb.log(wandb_metrics, step=step)

    def reset(self) -> None:
        """Reset all metrics."""
        self.losses.clear()
        self.step_times.clear()
        self.cumulative_loss = 0.0
        self.num_steps = 0
        self.tokens_processed = 0
        self.start_time = time.time()
