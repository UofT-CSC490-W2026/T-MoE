"""
DynMoE-style top-any router: sigmoid gate with thresholded selection.
"""

from typing import Tuple, Dict, Any, Optional

import torch
from torch import nn

from configs import DynMoERouterConfig
from src.core import RouterRegistry
from src.routers.base import BaseRouter
from src.metrics import RouterMetricsTracker


@RouterRegistry.register("dynmoe")
class DynMoERouter(BaseRouter):
    """
    DynMoE-style router: sigmoid gate and top-any selection via threshold.

    Uses `top_k` as a hard cap on selected experts per token. If fewer than
    `top_k` exceed the threshold, the remaining slots are filled by the next
    highest scores to keep fixed output shapes.
    """

    def __init__(self, config: DynMoERouterConfig):
        super().__init__(config)
        self.use_aux_loss = config.use_aux_loss
        self.aux_loss_coef = config.aux_loss_coef
        self.temperature = config.temperature
        self.gate_threshold = config.gate_threshold

        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        nn.init.xavier_uniform_(self.gate.weight)

        self.metrics_tracker = RouterMetricsTracker(self)

        # Last forward pass data for aux loss
        self._last_probs: Optional[torch.Tensor] = None
        self._last_indices: Optional[torch.Tensor] = None
        self._last_weights: Optional[torch.Tensor] = None

    def forward(
        self, x: torch.Tensor, return_metrics: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        # logits: [batch, seq, num_experts]
        logits = self.gate(x) / self.temperature
        probs = torch.sigmoid(logits)

        # Take top_k candidates, then apply threshold mask
        topk_probs, topk_indices = torch.topk(probs, self.top_k, dim=-1)
        mask = topk_probs >= self.gate_threshold

        # Ensure at least one expert is selected per token
        if mask.ndim == 3:
            mask[..., 0] = True

        weights = topk_probs * mask.float()
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = weights / denom

        if self.training and self.use_aux_loss:
            self._last_probs = probs
            self._last_indices = topk_indices
            self._last_weights = weights

        metrics = None
        if return_metrics:
            metrics = self.metrics_tracker.compute_all_metrics(topk_indices, weights)

        return weights, topk_indices, metrics

    def compute_aux_loss(self) -> torch.Tensor:
        if not self.use_aux_loss or not self.training:
            return torch.tensor(0.0, device=self.gate.weight.device)
        if (
            self._last_probs is None
            or self._last_indices is None
            or self._last_weights is None
        ):
            return torch.tensor(0.0, device=self.gate.weight.device)

        probs = self._last_probs  # [B, S, N]
        indices = self._last_indices  # [B, S, K]
        weights = self._last_weights  # [B, S, K]

        bsz, seq_len, num_experts = probs.shape
        num_tokens = bsz * seq_len

        P = probs.mean(dim=(0, 1))  # [N]
        usage = torch.zeros(num_experts, device=probs.device)
        flat_idx = indices.reshape(-1)
        flat_w = weights.reshape(-1)
        usage.scatter_add_(0, flat_idx, flat_w)
        usage = usage / max(num_tokens, 1)

        aux = self.aux_loss_coef * num_experts * (usage * P).sum()
        return aux

    def reset_state(self) -> None:
        pass

    def get_state(self) -> Dict[str, Any]:
        return {}

    def clear_aux_state(self) -> None:
        """Clear temporary tensors to release memory and avoid stale grads."""
        self._last_probs = None
        self._last_indices = None
        self._last_weights = None
