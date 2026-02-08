"""
Metrics tracking infrastructure for T-MoE routers.

This module provides comprehensive metrics computation and logging utilities
for monitoring expert routing behavior, fatigue dynamics, and load balancing.
"""

import torch
import numpy as np
from typing import Dict, Any

# Optional WandB integration
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class RouterMetricsTracker:
    """
    Comprehensive metrics tracker for Metabolic Router.

    Tracks:
    - Expert entropy and routing diversity
    - Fatigue statistics (mean, std, min, max, per-expert)
    - Usage distribution and counts
    - Gini coefficient for load balancing
    - Effective number of experts

    Usage:
        tracker = RouterMetricsTracker(router)
        metrics = tracker.compute_all_metrics(indices, weights)
        tracker.log_to_wandb(metrics, step=100)
    """

    def __init__(self, router):
        """
        Initialize metrics tracker.

        Args:
            router: MetabolicRouter instance to track
        """
        self.router = router
        self.num_experts = router.num_experts

        # Cache index tensor for Gini coefficient computation (avoid repeated allocations)
        self.gini_index = torch.arange(1, self.num_experts + 1, dtype=torch.float32)

    def compute_expert_entropy(
        self, indices: torch.Tensor, weights: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute Shannon entropy of expert routing distribution.

        Higher entropy = more diverse routing
        Lower entropy = concentrated routing

        Args:
            indices: Expert indices [batch, seq, top_k]
            weights: Routing weights [batch, seq, top_k]

        Returns:
            Dict with 'expert_entropy' and 'expert_entropy_normalized'
        """
        device = indices.device

        # Aggregate usage across all tokens
        usage = torch.zeros(self.num_experts, device=device)
        flat_indices = indices.flatten()
        flat_weights = weights.flatten()
        usage.scatter_add_(0, flat_indices, flat_weights)

        # Normalize to probability distribution
        usage_prob = usage / (usage.sum() + 1e-10)

        # Shannon entropy: -sum(p * log(p))
        entropy = -(usage_prob * torch.log(usage_prob + 1e-10)).sum()

        # Normalized entropy: divide by max entropy (log(num_experts))
        max_entropy = np.log(self.num_experts)
        normalized_entropy = entropy / max_entropy

        return {
            "expert_entropy": entropy.item(),
            "expert_entropy_normalized": normalized_entropy.item(),
        }

    def compute_fatigue_stats(self) -> Dict[str, Any]:
        """
        Compute comprehensive fatigue statistics.

        Returns:
            Dict with mean, std, min, max, and per-expert fatigue
        """
        fatigue = self.router.fatigue

        return {
            "fatigue_mean": fatigue.mean().item(),
            "fatigue_std": fatigue.std().item(),
            "fatigue_min": fatigue.min().item(),
            "fatigue_max": fatigue.max().item(),
            "fatigue_per_expert": fatigue.clone().cpu().numpy(),
        }

    def compute_usage_distribution(
        self, indices: torch.Tensor, weights: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Compute expert usage counts and distribution.

        Args:
            indices: Expert indices [batch, seq, top_k]
            weights: Routing weights [batch, seq, top_k]

        Returns:
            Dict with usage_counts and usage_distribution
        """
        device = indices.device

        # Count expert usage
        usage_counts = torch.zeros(self.num_experts, device=device)
        flat_indices = indices.flatten()
        flat_weights = weights.flatten()
        usage_counts.scatter_add_(0, flat_indices, flat_weights)

        # Normalize to distribution
        usage_dist = usage_counts / (usage_counts.sum() + 1e-10)

        return {
            "usage_counts": usage_counts.detach().cpu().numpy(),
            "usage_distribution": usage_dist.detach().cpu().numpy(),
        }

    def compute_gini_coefficient(
        self, indices: torch.Tensor, weights: torch.Tensor
    ) -> float:
        """
        Compute Gini coefficient for load balancing assessment.

        Gini = 0: Perfect balance (all experts used equally)
        Gini = 1: Perfect imbalance (one expert does everything)

        Args:
            indices: Expert indices [batch, seq, top_k]
            weights: Routing weights [batch, seq, top_k]

        Returns:
            Gini coefficient in [0, 1]
        """
        device = indices.device

        # Aggregate usage
        usage = torch.zeros(self.num_experts, device=device)
        flat_indices = indices.flatten()
        flat_weights = weights.flatten()
        usage.scatter_add_(0, flat_indices, flat_weights)

        # Sort usage in ascending order
        sorted_usage, _ = torch.sort(usage)

        # Compute Gini coefficient
        n = self.num_experts
        # Use cached index tensor (ensure device compatibility)
        index = self.gini_index.to(device)
        gini = (2 * (index * sorted_usage).sum()) / (n * sorted_usage.sum() + 1e-10) - (
            n + 1
        ) / n

        return gini.item()

    def compute_effective_experts(
        self, indices: torch.Tensor, weights: torch.Tensor
    ) -> float:
        """
        Compute effective number of experts (exponential of entropy).

        This metric answers: "How many experts are effectively being used?"

        Args:
            indices: Expert indices [batch, seq, top_k]
            weights: Routing weights [batch, seq, top_k]

        Returns:
            Effective number of experts in [1, num_experts]
        """
        entropy_dict = self.compute_expert_entropy(indices, weights)
        entropy = entropy_dict["expert_entropy"]
        return np.exp(entropy)

    def compute_all_metrics(
        self, indices: torch.Tensor, weights: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Compute all metrics in one pass.

        Args:
            indices: Expert indices [batch, seq, top_k]
            weights: Routing weights [batch, seq, top_k]

        Returns:
            Complete metrics dictionary
        """
        metrics = {}

        # Expert entropy
        metrics.update(self.compute_expert_entropy(indices, weights))

        # Fatigue statistics (only for routers that track fatigue, e.g. MetabolicRouter)
        if hasattr(self.router, "fatigue"):
            metrics.update(self.compute_fatigue_stats())

        # Usage distribution
        metrics.update(self.compute_usage_distribution(indices, weights))

        # Routing diversity
        metrics["routing_diversity_gini"] = self.compute_gini_coefficient(
            indices, weights
        )
        metrics["effective_experts"] = self.compute_effective_experts(indices, weights)

        # Step counter (only for routers that track steps)
        if hasattr(self.router, "num_steps"):
            metrics["num_steps"] = self.router.num_steps.item()

        return metrics

    def log_to_wandb(
        self, metrics: Dict[str, Any], step: int, prefix: str = "router"
    ) -> None:
        """
        Log metrics to Weights & Biases.

        Args:
            metrics: Metrics dictionary from compute_all_metrics()
            step: Training step number
            prefix: Metric name prefix (default: "router")
        """
        if not WANDB_AVAILABLE:
            return  # WandB not installed

        if not wandb.run:
            return  # WandB not initialized

        # Prepare scalar metrics for logging
        scalar_metrics = {}

        for key, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                scalar_metrics[f"{prefix}/{key}"] = value

        # Log scalars
        wandb.log(scalar_metrics, step=step)

        # Log histograms for array data
        if "fatigue_per_expert" in metrics:
            wandb.log(
                {
                    f"{prefix}/fatigue_histogram": wandb.Histogram(
                        metrics["fatigue_per_expert"]
                    )
                },
                step=step,
            )

        if "usage_distribution" in metrics:
            wandb.log(
                {
                    f"{prefix}/usage_histogram": wandb.Histogram(
                        metrics["usage_distribution"]
                    )
                },
                step=step,
            )

    def to_dict(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert metrics to JSON-serializable dictionary (for checkpointing).

        Args:
            metrics: Metrics dictionary from compute_all_metrics()

        Returns:
            JSON-serializable dictionary
        """
        serializable = {}

        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable[key] = float(value)
            else:
                serializable[key] = value

        return serializable
