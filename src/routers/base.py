from abc import abstractmethod, ABC
from typing import Tuple, Optional, Dict, Any

import torch
from torch import nn


class BaseRouter(nn.Module, ABC):
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
        pass

    @abstractmethod
    def compute_aux_loss(self) -> torch.Tensor:
        pass

    def step(self) -> None:
        """Called after optimizer.step(). Override for per-step state updates."""
        pass

    def reset_state(self) -> None:
        pass

    def clear_aux_state(self) -> None:
        """Clear cached tensors used for aux loss — call after optimizer.step()."""
        pass

    def get_state(self) -> Dict[str, Any]:
        return {}
