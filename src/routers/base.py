from abc import abstractmethod, ABC
from typing import Tuple, Optional, Dict, Any

import torch
from torch import nn


class BaseRouter(nn.Module, ABC):
    """
    Abstract base class for all routers.
    A router takes input embeddings and produces routing weights and indices
    for selecting which experts process each token.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k
        self.hidden_dim = config.hidden_dim

    @abstractmethod
    def forward(
        self, x: torch.Tensor, return_metrics: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Route inputs to experts.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            return_metrics: Whether to return additional metrics

        Returns:
            weights: Routing weights [batch, seq, top_k]
            indices: Expert indices [batch, seq, top_k]
            metrics: Optional dict of routing metrics
        """
        pass

    @abstractmethod
    def compute_aux_loss(self) -> torch.Tensor:
        """
        Compute auxiliary loss for load balancing.

        Returns 0 for routers that don't use aux loss (like MetabolicRouter).
        """
        pass

    def reset_state(self) -> None:
        """Reset any internal state (e.g., fatigue buffers)."""
        pass

    def clear_aux_state(self) -> None:
        """Clear temporary tensors used for auxiliary loss computation."""
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get router state for logging/checkpointing."""
        return {}
