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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, Any]]]:
        batch, seq, _ = x.shape
        # Router weights stay in fp32 by default; mixed-precision backbones can
        # emit bf16 hidden states. Cast inputs to the gate dtype before the
        # linear op so validation/eval under bf16 does not trip dtype mismatch.
        x_for_gate = x.to(self.gate.weight.dtype)
        logits = self.gate(x_for_gate).view(-1, self.num_experts) / self.temperature
        probs = F.softmax(logits, dim=-1)  # (N, E)

        # Select top-k experts per token
        top_k_values, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Create a weight matrix of shape (N, E)
        # This unified format is used by the LoRAMoELayer dispatcher
        expert_weights = torch.zeros_like(probs)
        expert_weights.scatter_(
            1, top_k_indices, F.normalize(top_k_values, p=1, dim=-1)
        )

        if self.training and self.use_aux_loss:
            self._last_probs = probs.view(batch, seq, -1)
            self._last_indices = top_k_indices.view(batch, seq, -1)
            self._last_weights = expert_weights.view(batch, seq, -1)

        metrics = None
        if return_metrics:
            # For backward compatibility with the tracker, we pass the sparse format
            # but usually the layer only needs expert_weights
            metrics = self.metrics_tracker.compute_all_metrics(
                top_k_indices, top_k_values
            )

        return expert_weights, None, metrics

    def compute_aux_loss(self) -> torch.Tensor:
        if not self.use_aux_loss or not self.training:
            return torch.tensor(0.0, device=self.gate.weight.device)
        if (
            self._last_probs is None
            or self._last_indices is None
            or self._last_weights is None
        ):
            return torch.tensor(0.0, device=self.gate.weight.device)

        # aux = α * num_experts * Σ_i (f_i * P_i)
        # f_i = fraction of total weight assigned to expert i
        # P_i = average gate probability for expert i
        # Note: uses mean routing weights (soft f_i) rather than the hard dispatch
        # fraction in the original Switch Transformer (Fedus et al. 2021). Equivalent
        # in the limit but differs in gradient signal.
        probs = self._last_probs
        weights = self._last_weights

        bsz, seq_len, num_experts = probs.shape

        # P_i: Mean gate probability across the batch
        P = probs.mean(dim=(0, 1))

        # f_i: Mean routing weight across the batch
        usage = weights.mean(dim=(0, 1))

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


@RouterRegistry.register("switch")
class SwitchRouter(StandardRouter):
    def __init__(self, config: SwitchRouterConfig):
        super().__init__(config)
        self.use_aux_loss = getattr(config, "use_aux_loss", False)
        self.aux_loss_coef = getattr(config, "aux_loss_coef", 0.01)
        if self.top_k != 1:
            self.top_k = 1
