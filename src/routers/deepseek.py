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
        self,
        x: torch.Tensor,
        return_metrics: bool = False,
        record_usage: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        x_for_gate = x.to(self.gate.weight.dtype)
        logits = self.gate(x_for_gate) / self.temperature

        if self.training and self.noise_std > 0.0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # DeepSeek V3 §2.1: selection uses biased logits; output weights use unbiased scores.
        # Bias corrects load imbalance at routing time but must not contaminate the output
        # weight gradient, which flows back through the gate weight matrix only.
        biased_logits = logits + self.bias

        # Selection: top-k by biased logits (stop-gradient on selection itself)
        _, top_k_indices = torch.topk(
            biased_logits, self.top_k, dim=-1
        )  # (B, S, top_k)

        # Output weights: unbiased scores gathered at selected indices, then L1-normalized
        if self.use_sigmoid:
            raw_scores = torch.sigmoid(logits)  # (B, S, E) — gradient flows here
        else:
            raw_scores = F.softmax(logits, dim=-1)  # (B, S, E)

        top_k_weights = raw_scores.gather(-1, top_k_indices)  # (B, S, top_k)
        top_k_weights = F.normalize(top_k_weights, p=1, dim=-1)  # L1 normalize

        bsz, seq, _ = logits.shape
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)  # (N, top_k)
        top_k_weights_flat = top_k_weights.view(-1, self.top_k)  # (N, top_k)
        raw_scores_flat = raw_scores.view(-1, self.num_experts)  # (N, E)

        expert_weights = torch.zeros_like(raw_scores_flat)
        expert_weights = expert_weights.scatter(
            1, top_k_indices_flat, top_k_weights_flat.to(expert_weights.dtype)
        )

        if self.training and record_usage:
            self._record_usage(top_k_indices)

        metrics = None
        if return_metrics:
            metrics = self.metrics_tracker.compute_all_metrics(
                top_k_indices_flat, top_k_weights_flat
            )
            # eff_E_hard: hard assignment effective experts — paper comparison parity
            hard = torch.zeros(
                self.num_experts, device=top_k_indices_flat.device, dtype=torch.float32
            )
            hard.scatter_add_(
                0,
                top_k_indices_flat.reshape(-1).clamp(min=0),
                torch.ones(
                    top_k_indices_flat.numel(), device=top_k_indices_flat.device
                ),
            )
            hard = hard / hard.sum().clamp(min=1e-8)
            metrics["eff_E_hard"] = (1.0 / (hard**2).sum().clamp(min=1e-8)).item()

        return expert_weights, None, metrics

    @torch.no_grad()
    def _sync_usage_distributed(self) -> None:
        try:
            import torch.distributed as dist
        except ImportError:
            return
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        dist.all_reduce(self._pending_usage_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._pending_tokens, op=dist.ReduceOp.SUM)

    def step(self) -> None:
        with torch.no_grad():
            # Sync BEFORE the early-return guard: dist.all_reduce is a collective —
            # all DDP ranks must call it together. Gating it behind _usage_pending
            # (a non-collective bool) risks NCCL deadlock if ranks ever diverge.
            self._sync_usage_distributed()

            if not self._usage_pending:
                return

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
