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
        self.noise_std = config.noise_std
        
        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        nn.init.xavier_uniform_(self.gate.weight)
        
        self.metrics_tracker = RouterMetricsTracker(self)

    def forward(
        self, x: torch.Tensor, return_metrics: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        batch, seq, hidden = x.shape
        
        logits = self.gate(x) / self.temperature
        
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
            
        logits_flat = logits.view(-1, self.num_experts) # (N, E)
        probs = F.softmax(logits_flat, dim=0) # Normalize over tokens
        
        capacity = int((batch * seq / self.num_experts) * self.top_k)
        capacity = max(capacity, 1) # Ensure at least 1 token is selected
        
        top_probs, top_indices = torch.topk(probs, capacity, dim=0) # (capacity, E)
        
        indices = top_indices.t() # (E, capacity)
        weights = top_probs.t() # (E, capacity)
        
        metrics = None
        if return_metrics:
            selected_tokens = torch.unique(indices)
            drop_rate = 1.0 - (selected_tokens.numel() / (batch * seq))
            metrics = {"token_drop_rate": drop_rate}
            
        return weights, indices, metrics

    def compute_aux_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.gate.weight.device)
