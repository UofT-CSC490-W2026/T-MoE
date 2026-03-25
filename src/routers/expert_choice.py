import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

from src.configs import ExpertChoiceRouterConfig
from src.core import RouterRegistry
from src.routers.base import BaseRouter
from src.metrics import RouterMetricsTracker
from src.project_types import RouterType


@RouterRegistry.register(RouterType.EXPERT_CHOICE.value)
class ExpertChoiceRouter(BaseRouter):
    """
    Expert Choice Router: normalizes scores over tokens, and experts select top-c tokens.
    """

    is_expert_choice = True

    def __init__(self, config: ExpertChoiceRouterConfig):
        super().__init__(config)
        self.temperature = config.temperature

        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        nn.init.xavier_uniform_(self.gate.weight)

        self.metrics_tracker = RouterMetricsTracker(self)

    def forward(
        self, x: torch.Tensor, return_metrics: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        # Ensure x is contiguous for stable view/indexing
        x = x.contiguous()
        batch, seq, hidden = x.shape

        x_for_gate = x.to(self.gate.weight.dtype)
        logits_flat = (
            self.gate(x_for_gate).view(-1, self.num_experts) / self.temperature
        )

        # Stability Guard: Check for NaNs/Infs that could stall the softmax/topk
        if not torch.isfinite(logits_flat).all():
            logits_flat = torch.nan_to_num(
                logits_flat, nan=0.0, posinf=1.0, neginf=-1.0
            )

        # Normalize across TOKENS (dim=0) per expert column — Expert Choice paper §3.1.
        # Each expert's column forms a probability distribution over tokens in the batch,
        # so the router expresses "how much does expert i want each token" (not vice versa).
        probs = F.softmax(logits_flat, dim=0)  # (N, E)

        max_tokens = logits_flat.size(0)
        capacity = int(max_tokens / self.num_experts * self.top_k)
        capacity = max(1, min(capacity, max_tokens))

        # Select top-c tokens per expert
        top_probs, top_indices = torch.topk(probs, capacity, dim=0)  # (capacity, E)

        # Out-of-place scatter keeps top_probs in the autograd graph so
        # gate.weight receives gradient from the task loss through top_probs.
        # zeros_like + scatter_ (in-place) would detach gate.weight from the graph.
        expert_weights = torch.zeros(
            max_tokens, self.num_experts, dtype=probs.dtype, device=probs.device
        )
        expert_weights = expert_weights.scatter(0, top_indices, top_probs)

        metrics = None
        if return_metrics:
            selected_count = (expert_weights.sum(dim=1) > 0).sum().item()
            drop_rate = 1.0 - (selected_count / max_tokens)
            metrics = self.metrics_tracker.compute_all_metrics(None, expert_weights)
            metrics["token_drop_rate"] = drop_rate
            # eff_E_hard: tokens-per-expert hard counts — uniform by construction in expert
            # choice (each expert selects exactly `capacity` tokens), so eff_E_hard ≈ N.
            # Computed explicitly to catch early-training skew and for paper comparison parity.
            hard = (expert_weights > 0).float().sum(dim=0)  # (E,) tokens per expert
            hard = hard / hard.sum().clamp(min=1e-8)
            metrics["eff_E_hard"] = (1.0 / (hard**2).sum().clamp(min=1e-8)).item()

        # Return (expert_weights, None, metrics)
        return expert_weights, None, metrics

    def compute_aux_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.gate.weight.device)
