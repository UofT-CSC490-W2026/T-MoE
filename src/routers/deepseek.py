import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

from src.configs import DeepSeekRouterConfig
from src.core import RouterRegistry
from src.routers.base import BaseRouter
from src.metrics import RouterMetricsTracker
from src.project_types import RouterType


@RouterRegistry.register(RouterType.DEEPSEEK.value)
class DeepSeekRouter(BaseRouter):
    """
    DeepSeek V2/V3 style router with bias correction for load balancing.
    Removes auxiliary losses completely.
    """
    def __init__(self, config: DeepSeekRouterConfig):
        super().__init__(config)
        self.temperature = config.temperature
        self.noise_std = config.noise_std
        self.bias_update_rate = config.bias_update_rate
        self.use_sigmoid = config.use_sigmoid
        
        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        nn.init.xavier_uniform_(self.gate.weight)
        
        self.bias = nn.Parameter(torch.zeros(config.num_experts), requires_grad=False)
        
        self.metrics_tracker = RouterMetricsTracker(self)
        
        self.register_buffer("_pending_usage_sum", torch.zeros(self.num_experts))
        self.register_buffer("_pending_tokens", torch.tensor(0, dtype=torch.long))
        self._usage_pending = False

    def _record_usage(self, indices: torch.Tensor) -> None:
        flat_indices = indices.flatten()
        batch_tokens = flat_indices.numel() // self.top_k
        
        usage = (
            torch.bincount(flat_indices, minlength=self.num_experts)
            .float()
            .div_(self.top_k)
        )
        
        if self._usage_pending:
            self._pending_usage_sum.add_(usage.to(self._pending_usage_sum.dtype))
            self._pending_tokens.add_(batch_tokens)
        else:
            self._pending_usage_sum.copy_(usage)
            self._pending_tokens.fill_(batch_tokens)
            self._usage_pending = True

    def forward(
        self, x: torch.Tensor, return_metrics: bool = False, record_usage: bool = True, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        logits = self.gate(x) / self.temperature
        
        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
            
        routed_logits = logits + self.bias
        
        if self.use_sigmoid:
            probs = torch.sigmoid(routed_logits)
        else:
            probs = F.softmax(routed_logits, dim=-1)
            
        top_k_values, top_k_indices = torch.topk(probs, self.top_k, dim=-1)

        # Create a unified dense weight matrix (N, E)
        bsz, seq, _ = probs.shape
        probs_flat = probs.view(-1, self.num_experts)
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)
        
        if self.use_sigmoid:
            top_k_weights = top_k_values
        else:
            top_k_weights = F.normalize(top_k_values, p=1, dim=-1)
        
        expert_weights = torch.zeros_like(probs_flat)
        expert_weights.scatter_(1, top_k_indices_flat, top_k_weights.view(-1, self.top_k))
            
        if self.training and record_usage:
            self._record_usage(top_k_indices)
            
        metrics = None
        if return_metrics:
            metrics = self.metrics_tracker.compute_all_metrics(
                top_k_indices,
                top_k_weights.view(-1, self.top_k) if not self.use_sigmoid
                else top_k_values.view(-1, self.top_k)
            )
            
        return expert_weights, None, metrics

    def step(self) -> None:
        if not self._usage_pending:
            return
            
        with torch.no_grad():
            usage_avg = (
                self._pending_usage_sum.float()
                / self._pending_tokens.float().clamp(min=1)
            )
            
            target_load = 1.0 / self.num_experts
            overloaded = (usage_avg > target_load).float()
            underloaded = (usage_avg < target_load).float()
            
            bias_update = torch.zeros_like(self.bias)
            bias_update -= overloaded * self.bias_update_rate
            bias_update += underloaded * self.bias_update_rate
            
            self.bias.add_(bias_update)
            
            self._pending_usage_sum.zero_()
            self._pending_tokens.zero_()
            self._usage_pending = False

    def compute_aux_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.gate.weight.device)

    def reset_state(self) -> None:
        with torch.no_grad():
            self.bias.zero_()
            self._pending_usage_sum.zero_()
            self._pending_tokens.zero_()
            self._usage_pending = False

    def get_state(self) -> Dict[str, Any]:
        return {
            "bias": self.bias.clone(),
            "mean_bias": self.bias.mean().item(),
            "max_bias": self.bias.max().item(),
            "min_bias": self.bias.min().item(),
        }
