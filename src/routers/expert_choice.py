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
        
        logits_flat = self.gate(x).view(-1, self.num_experts) / self.temperature
        
        # Stability Guard: Check for NaNs/Infs that could stall the softmax/topk
        if not torch.isfinite(logits_flat).all():
             logits_flat = torch.nan_to_num(logits_flat, nan=0.0, posinf=1.0, neginf=-1.0)
             
        # Normalize across EXPERTS (dim=-1) instead of tokens (dim=0)
        # to ensure weights/gradients have sufficient magnitude (~1/E).
        probs = F.softmax(logits_flat, dim=-1) # (N, E)
        
        max_tokens = logits_flat.size(0)
        capacity = int(max_tokens / self.num_experts * self.top_k)
        capacity = max(1, min(capacity, max_tokens))
        
        # Select top-c tokens per expert
        top_probs, top_indices = torch.topk(probs, capacity, dim=0) # (capacity, E)
        
        # Create a unified weight matrix of shape (N, E)
        expert_weights = torch.zeros_like(probs) # (N, E)
        expert_weights.scatter_(0, top_indices, top_probs)
        
        metrics = None
        if return_metrics:
            selected_count = (expert_weights.sum(dim=1) > 0).sum().item()
            drop_rate = 1.0 - (selected_count / max_tokens)
            metrics = self.metrics_tracker.compute_all_metrics(None, expert_weights)
            metrics["token_drop_rate"] = drop_rate
            
        # Return (expert_weights, None, metrics)
        return expert_weights, None, metrics

    def compute_aux_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.gate.weight.device)
