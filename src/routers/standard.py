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
            1, top_k_indices, F.normalize(top_k_values, p=1, dim=-1).to(probs.dtype)
        )

        if self.training and self.use_aux_loss:
            self._last_probs = probs.view(batch, seq, -1)
            self._last_indices = top_k_indices.view(batch, seq, -1)
            self._last_weights = expert_weights.view(batch, seq, -1)

        metrics = None
        if return_metrics:
            metrics = self.metrics_tracker.compute_all_metrics(
                top_k_indices, top_k_values
            )
            # eff_E_hard: hard assignment effective experts — matches StressCorrectedRouter
            # metric for paper comparison parity (WandB router/layer_*/eff_E_hard).
            hard = torch.zeros(
                self.num_experts, device=top_k_indices.device, dtype=torch.float32
            )
            hard.scatter_add_(
                0,
                top_k_indices.reshape(-1).clamp(min=0),
                torch.ones(top_k_indices.numel(), device=top_k_indices.device),
            )
            hard = hard / hard.sum().clamp(min=1e-8)
            metrics["eff_E_hard"] = (1.0 / (hard**2).sum().clamp(min=1e-8)).item()

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

        # Switch Transformer aux loss: α · N · Σᵢ(fᵢ · Pᵢ)
        # fᵢ = hard dispatch fraction (stop-gradient): fraction of tokens routed to expert i.
        #       Uses hard top-k selection counts, detached — no gradient through selection.
        # Pᵢ = mean full softmax probability for expert i — differentiable, carries gradient.
        # Reference: Fedus et al. 2021, equation (4).
        probs = self._last_probs
        indices = self._last_indices

        bsz, seq_len, num_experts = probs.shape

        # fᵢ: hard dispatch counts, stop-gradient
        indices_flat = indices.reshape(-1)
        dispatch = torch.zeros(num_experts, device=probs.device, dtype=probs.dtype)
        dispatch.scatter_add_(
            0, indices_flat, torch.ones_like(indices_flat, dtype=probs.dtype)
        )
        f = (dispatch / (bsz * seq_len * self.top_k)).detach()

        # Pᵢ: mean gate probability — differentiable
        P = probs.mean(dim=(0, 1))

        aux = self.aux_loss_coef * num_experts * (f * P).sum()
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
