from abc import abstractmethod
from dataclasses import dataclass

import torch
from torch import nn

from src.experts.base import BaseExpert


@dataclass
class LoRAConfig:
    """Configuration for LoRA layers and experts."""

    hidden_dim: int
    intermediate_dim: int = None  # For MLP expansion, defaults to 4*hidden_dim
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    init_scale: float = 0.01

    def __post_init__(self):
        if self.intermediate_dim is None:
            self.intermediate_dim = 4 * self.hidden_dim

    @property
    def scaling(self) -> float:
        """LoRA scaling factor."""
        return self.alpha / self.rank


class LoRAMLPExpert(BaseExpert):
    """
    Abstract base class for LoRA-based MLP experts.

    Subclasses must implement architecture-specific MLP structure
    (e.g., GPT-Neo, Llama, Mixtral).

    All LoRA MLP experts share common patterns:
    - Multiple LoRA layers
    - Frozen base weights
    - Activation functions
    - Load from pretrained MLP
    """

    def __init__(self, config: LoRAConfig):
        super().__init__(config)
        self.config = config

    @abstractmethod
    def load_from_mlp(self, mlp: nn.Module) -> None:
        """
        Load frozen weights from a pretrained MLP module.

        Args:
            mlp: Pretrained MLP module (architecture-specific)
        """
        pass

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the expert MLP.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        pass

    def freeze_base_weights(self) -> None:
        """Ensure all base weights are frozen (LoRA adapters remain trainable)."""
        for name, param in self.named_parameters():
            if "base_weight" in name or "base_bias" in name:
                param.requires_grad = False
