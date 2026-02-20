from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any

import torch
from torch import nn

from src.routers import BaseRouter


class BaseMoELayer(nn.Module, ABC):
    """
    Abstract base class for MoE layers.

    An MoE layer combines a router and multiple experts to process inputs.
    All concrete implementations must follow the forward() signature contract.
    """

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int):
        """
        Initialize MoE layer.

        Args:
            hidden_dim: Dimension of input/output embeddings
            num_experts: Number of expert networks
            top_k: Maximum number of experts per token (may be variable)

        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_experts <= 0:
            raise ValueError(f"num_experts must be positive, got {num_experts}")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")

        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router: Optional[BaseRouter] = None
        self.experts: Optional[nn.ModuleList] = None

        # Cached routing state from last forward pass for metric retrieval
        # without requiring a second forward pass through experts.
        self._last_routing_weights: Optional[torch.Tensor] = None
        self._last_routing_indices: Optional[torch.Tensor] = None

    @abstractmethod
    def forward(
        self, x: torch.Tensor, return_metrics: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Process input through the MoE layer.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            return_metrics: Whether to return routing metrics
            **kwargs: Router-specific arguments (e.g., noise_std, temperature)

        Returns:
            output: Processed tensor [batch_size, seq_len, hidden_dim]
            metrics: Optional routing metrics dict (None if return_metrics=False)

        Raises:
            RuntimeError: If router or experts not initialized
            ValueError: If input shape is invalid
        """
        pass

    def get_router(self) -> BaseRouter:
        """Get the router component."""
        if self.router is None:
            raise RuntimeError("Router not initialized")
        return self.router

    def get_cached_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve routing metrics from the last forward pass without
        re-running expert computation.

        Returns:
            Metrics dictionary if cached routing state is available, else None.
        """
        if self._last_routing_weights is None or self._last_routing_indices is None:
            return None
        if self.router is None:
            return None
        if not hasattr(self.router, "metrics_tracker"):
            return None

        metrics = self.router.metrics_tracker.compute_all_metrics(
            self._last_routing_indices, self._last_routing_weights
        )
        metrics["weights"] = self._last_routing_weights
        metrics["indices"] = self._last_routing_indices
        return metrics

    def clear_cached_metrics(self) -> None:
        """Clear cached routing state to free memory."""
        self._last_routing_weights = None
        self._last_routing_indices = None
        return self.router

    def get_experts(self) -> nn.ModuleList:
        """Get the expert modules."""
        if self.experts is None:
            raise RuntimeError("Experts not initialized")
        return self.experts

    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        return f"hidden_dim={self.hidden_dim}, num_experts={self.num_experts}, top_k={self.top_k}"
