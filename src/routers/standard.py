from typing import Tuple, Dict, Any, Optional

import torch
from torch import nn
import torch.nn.functional as F

from src.configs import StandardRouterConfig, TopKRouterConfig, SwitchRouterConfig
from src.core import RouterRegistry
from src.routers.base import BaseRouter
from src.metrics import RouterMetricsTracker


@RouterRegistry.register("standard")
class StandardRouter(BaseRouter):
    def __init__(self, config: StandardRouterConfig):
        super().__init__(config)
        self.use_aux_loss = config.use_aux_loss
        self.aux_loss_coef = config.aux_loss_coef
        self.temperature = config.temperature

        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        nn.init.xavier_uniform_(self.gate.weight)

        self.metrics_tracker = RouterMetricsTracker(self)

        self._last_probs: Optional[torch.Tensor] = None
        self._last_indices: Optional[torch.Tensor] = None
        self._last_weights: Optional[torch.Tensor] = None

    def forward(
        self, x: torch.Tensor, return_metrics: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        logits = self.gate(x) / self.temperature
        probs = F.softmax(logits, dim=-1)

        top_k_values, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        weights = F.normalize(top_k_values, p=1, dim=-1)

        if self.training and self.use_aux_loss:
            self._last_probs = probs
            self._last_indices = top_k_indices
            self._last_weights = weights

        metrics = None
        if return_metrics:
            metrics = self.metrics_tracker.compute_all_metrics(top_k_indices, weights)

        return weights, top_k_indices, metrics

    def compute_aux_loss(self) -> torch.Tensor:
        if not self.use_aux_loss or not self.training:
            return torch.tensor(0.0, device=self.gate.weight.device)
        if (
            self._last_probs is None
            or self._last_indices is None
            or self._last_weights is None
        ):
            return torch.tensor(0.0, device=self.gate.weight.device)

        # aux = α * N * Σ_i (f_i * P_i), f_i = normalized routing weight, P_i = mean gate prob
        probs = self._last_probs
        indices = self._last_indices
        weights = self._last_weights

        bsz, seq_len, num_experts = probs.shape
        num_tokens = bsz * seq_len

        P = probs.mean(dim=(0, 1))
        usage = torch.zeros(num_experts, device=probs.device, dtype=torch.float32)
        flat_idx = indices.reshape(-1)
        flat_w = weights.reshape(-1).to(torch.float32)
        usage.scatter_add_(0, flat_idx, flat_w)
        usage = usage / max(num_tokens, 1)

        aux = self.aux_loss_coef * num_experts * (usage * P).sum()
        return aux

    def get_state(self) -> Dict[str, Any]:
        return {}

    def clear_aux_state(self) -> None:
        """Clear temporary tensors to release memory and avoid stale grads."""
        self._last_probs = None
        self._last_indices = None
        self._last_weights = None


@RouterRegistry.register("topk")
class TopKRouter(StandardRouter):
    def __init__(self, config: TopKRouterConfig):
        super().__init__(config)
        self.use_aux_loss = False
        self._last_probs = None
        self._last_indices = None
        self._last_weights = None


@RouterRegistry.register("switch")
class SwitchRouter(StandardRouter):
    def __init__(self, config: SwitchRouterConfig):
        super().__init__(config)
        self.use_aux_loss = getattr(config, "use_aux_loss", False)
        self.aux_loss_coef = getattr(config, "aux_loss_coef", 0.01)
        if self.top_k != 1:
            self.top_k = 1
