from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseExpert(nn.Module, ABC):
    """
    Abstract base class for expert networks.
    An expert is a small neural network that processes tokens routed to it.
    """

    def __init__(self, config):
        """
        Initialize expert.
        :param config: Configuration object for the expert
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process input through the expert.

        Args:
            x: Input tensor [num_tokens, hidden_dim]

        Returns:
            Output tensor [num_tokens, hidden_dim]
        """
        pass

    def get_param_count(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_flops(self, x: torch.Tensor) -> int:
        """Estimate FLOPs for processing input x."""
        # This is a placeholder implementation.
        # Actual FLOP counting would depend on the expert architecture.
        return 2 * x.numel() * self.get_param_count()  # Rough estimate

    def clone_from_parent(self, parent_expert: "BaseExpert"):
        """Clone parameters from a parent expert."""
        self.load_state_dict(parent_expert.state_dict())
