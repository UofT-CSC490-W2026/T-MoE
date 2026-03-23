import torch
import numpy as np
from typing import Dict, Any, Optional

# Optional WandB integration
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class GlobalSpecializationTracker:
    """
    Globally tracks token-to-expert routing to compute valid Information Theory metrics.
    Because token frequencies are low per-batch, H(E|T) must be tracked across many batches
    to form a true probability distribution.
    """

    def __init__(self, vocab_size: int, num_experts: int, device: str = "cpu"):
        self.vocab_size = vocab_size
        self.num_experts = num_experts
        # Accumulate on CPU to avoid wasting VRAM
        self.usage_counts = torch.zeros(
            (int(vocab_size), int(num_experts)),
            dtype=torch.long,
            device=device,
        )
        self.total_tokens = 0

    def update(self, token_ids: torch.Tensor, expert_indices: Optional[torch.Tensor]):
        if expert_indices is None:
            return
        with torch.no_grad():
            batch, seq, top_k = expert_indices.shape

            flat_tokens = token_ids.view(-1)
            flat_experts = expert_indices.reshape(-1, top_k)

            # Filter out padding tokens
            valid_mask = (flat_tokens >= 0) & (flat_tokens < self.vocab_size)
            valid_tokens = flat_tokens[valid_mask].cpu()
            valid_experts = flat_experts[valid_mask].cpu()

            if len(valid_tokens) == 0:
                return

            expanded_tokens = valid_tokens.unsqueeze(1).expand(-1, top_k).reshape(-1)
            flattened_experts = valid_experts.flatten()

            # Filter out padding experts (-1 used in adaptive k)
            expert_mask = flattened_experts >= 0
            if not expert_mask.any():
                return

            expanded_tokens = expanded_tokens[expert_mask]
            flattened_experts = flattened_experts[expert_mask]

            linear_indices = expanded_tokens * self.num_experts + flattened_experts
            counts = torch.bincount(
                linear_indices, minlength=self.vocab_size * self.num_experts
            )

            self.usage_counts += counts.view(self.vocab_size, self.num_experts)
            self.total_tokens += valid_tokens.numel()

    def sync_and_compute(
        self, device: str, is_distributed: bool = False
    ) -> Dict[str, float]:
        """
        All-reduce usage_counts across DDP ranks, then compute metrics.

        Non-destructive: each rank keeps its own local accumulation intact.
        At log time we reduce a temporary copy so the next accumulation window
        starts fresh on each rank's local data (no double-counting).

        Must be called by ALL ranks simultaneously (collective operation).
        Only rank 0 needs to use the returned metrics.

        Args:
            device: CUDA device string for the all-reduce (e.g. "cuda:0").
            is_distributed: Whether distributed training is active.

        Returns:
            Metrics dict computed from the globally-reduced histogram.
        """
        if not is_distributed:
            return self.compute_metrics()

        import torch.distributed as dist

        # All-reduce a GPU copy of usage_counts — O(vocab × experts) = ~3.2 MB for 125M
        counts_gpu = self.usage_counts.to(device)
        dist.all_reduce(counts_gpu, op=dist.ReduceOp.SUM)
        counts_synced = counts_gpu.cpu()

        # All-reduce total_tokens so global_tokens_seen is correct
        tok_t = torch.tensor(self.total_tokens, dtype=torch.long, device=device)
        dist.all_reduce(tok_t, op=dist.ReduceOp.SUM)
        total_synced = tok_t.item()

        # Temporarily swap in synced data, compute metrics, restore local state
        orig_counts, orig_total = self.usage_counts, self.total_tokens
        self.usage_counts = counts_synced
        self.total_tokens = total_synced
        result = self.compute_metrics()
        self.usage_counts = orig_counts
        self.total_tokens = orig_total

        return result

    def compute_metrics(self) -> Dict[str, float]:
        if self.total_tokens == 0:
            return {}

        with torch.no_grad():
            token_counts = self.usage_counts.sum(dim=1)
            active_mask = token_counts > 0

            if not active_mask.any():
                return {}

            active_usage = self.usage_counts[active_mask].float()
            active_token_counts = token_counts[active_mask].float()

            # P(E|T)
            p_e_given_t = active_usage / active_usage.sum(dim=1, keepdim=True)

            # H(E|T) — weighted by token frequency
            entropy_e_given_t = -(p_e_given_t * torch.log(p_e_given_t + 1e-10)).sum(
                dim=1
            )
            p_t = active_token_counts / active_token_counts.sum()
            expected_conditional_entropy = (p_t * entropy_e_given_t).sum()

            # H(E)
            total_expert_usage = self.usage_counts.sum(dim=0).float()
            p_e = total_expert_usage / total_expert_usage.sum()
            marginal_entropy = -(p_e * torch.log(p_e + 1e-10)).sum()

            if marginal_entropy.item() < 1e-5:
                specialization_score = 0.0
                collapse_score = 1.0
            else:
                specialization_score = 1.0 - (
                    expected_conditional_entropy / marginal_entropy
                )
                collapse_score = 1.0 - (marginal_entropy / np.log(self.num_experts))

            return {
                "specialization_score": float(specialization_score.item()),
                "collapse_score": float(collapse_score.item()),
                "marginal_entropy": float(marginal_entropy.item()),
                "conditional_entropy": float(expected_conditional_entropy.item()),
                "global_tokens_seen": float(self.total_tokens),
            }


class RouterMetricsTracker:
    """
    Per-router metrics tracker (all router types).

    Tracks entropy, usage distribution, Gini, effective experts,
    confidence, and fatigue stats (metabolic router only).
    """

    def __init__(self, router):
        self.router = router
        self.num_experts = router.num_experts
        self.gini_index = torch.arange(1, self.num_experts + 1, dtype=torch.float32)

    def _compute_usage(
        self, indices: Optional[torch.Tensor], weights: torch.Tensor
    ) -> torch.Tensor:
        """Aggregate routing weights into per-expert usage vector [num_experts]."""
        if indices is None:
            # Dense case: weights is (N, E)
            return weights.sum(dim=0).to(torch.float32)
        usage = torch.zeros(
            self.num_experts, device=indices.device, dtype=torch.float32
        )
        # clamp min to 0 to safely ignore -1 indices (weights will be 0 anyway)
        safe_indices = indices.flatten().clamp(min=0)
        usage.scatter_add_(0, safe_indices, weights.flatten().to(torch.float32))
        return usage

    def compute_expert_entropy(
        self,
        indices: torch.Tensor,
        weights: torch.Tensor,
        usage: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        if usage is None:
            usage = self._compute_usage(indices, weights)
        usage_prob = usage / (usage.sum() + 1e-10)
        entropy = -(usage_prob * torch.log(usage_prob + 1e-10)).sum()
        max_entropy = np.log(self.num_experts)
        normalized_entropy = entropy / max_entropy

        return {
            "expert_entropy": entropy.item(),
            "expert_entropy_normalized": normalized_entropy.item(),
        }

    def compute_fatigue_stats(self) -> Dict[str, Any]:
        if not hasattr(self.router, "fatigue"):
            return {}
        fatigue = self.router.fatigue

        return {
            "fatigue_mean": fatigue.mean().item(),
            "fatigue_std": fatigue.std().item(),
            "fatigue_min": fatigue.min().item(),
            "fatigue_max": fatigue.max().item(),
            "fatigue_per_expert": fatigue.cpu().float().numpy(),
        }

    def compute_usage_distribution(
        self,
        indices: torch.Tensor,
        weights: torch.Tensor,
        usage: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        usage_counts = (
            usage if usage is not None else self._compute_usage(indices, weights)
        )
        usage_dist = usage_counts / (usage_counts.sum() + 1e-10)

        return {
            "usage_counts": usage_counts.detach().cpu().float().numpy(),
            "usage_distribution": usage_dist.detach().cpu().float().numpy(),
        }

    def compute_gini_coefficient(
        self,
        indices: torch.Tensor,
        weights: torch.Tensor,
        usage: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute Gini coefficient for load balancing assessment.

        Gini = 0: Perfect balance (all experts used equally)
        Gini = 1: Perfect imbalance (one expert does everything)

        Args:
            indices: Expert indices [batch, seq, top_k]
            weights: Routing weights [batch, seq, top_k]
            usage: Precomputed usage tensor [num_experts] (optional)

        Returns:
            Gini coefficient in [0, 1]
        """
        usage = usage if usage is not None else self._compute_usage(indices, weights)
        sorted_usage, _ = torch.sort(usage)
        n = self.num_experts
        device = indices.device if indices is not None else weights.device
        if not hasattr(self, "_gini_index_device") or self._gini_index_device != device:
            self._gini_index_cache = self.gini_index.to(device)
            self._gini_index_device = device
        index = self._gini_index_cache
        gini = (2 * (index * sorted_usage).sum()) / (n * sorted_usage.sum() + 1e-10) - (
            n + 1
        ) / n

        return gini.item()

    def compute_effective_experts(
        self,
        indices: torch.Tensor,
        weights: torch.Tensor,
        entropy: float = None,
        usage: Optional[torch.Tensor] = None,
    ) -> float:
        """Effective number of experts = exp(entropy). Range: [1, num_experts]."""
        if entropy is None:
            entropy = self.compute_expert_entropy(indices, weights, usage=usage)[
                "expert_entropy"
            ]
        return np.exp(entropy)

    def compute_confidence_metrics(self, weights: torch.Tensor) -> Dict[str, float]:
        """
        Compute per-token routing confidence statistics.

        Measures how decisively the router assigns tokens to experts.
        High confidence (→1.0) means one expert dominates per token.
        Low confidence (→1/top_k) means uniform weighting across selected experts.

        Args:
            weights: Routing weights [batch, seq, top_k]

        Returns:
            Dict with confidence metrics:
            - router_confidence_mean: mean of max weight per token
            - router_confidence_std: std of max weight per token
            - top1_dominance: mean fraction of total weight on top-1 expert
        """
        # Max weight per token (how much weight goes to the most preferred expert)
        max_w = weights.max(dim=-1).values  # [B, S]

        # Top-1 dominance: fraction of weight on the strongest expert
        # For top_k=1 this is always 1.0; for top_k=2 it shows how unequal the split is
        weight_sum = weights.sum(dim=-1).clamp_min(1e-10)  # [B, S]
        top1_dominance = max_w / weight_sum  # [B, S]

        return {
            "router_confidence_mean": max_w.mean().item(),
            "router_confidence_std": max_w.std().item(),
            "top1_dominance": top1_dominance.mean().item(),
        }

    def compute_all_metrics(
        self,
        indices: Optional[torch.Tensor],
        weights: torch.Tensor,
    ) -> Dict[str, Any]:
        metrics = {}
        # Compute usage once — shared by entropy, distribution, gini, effective_experts.
        usage = self._compute_usage(indices, weights)
        entropy_metrics = self.compute_expert_entropy(indices, weights, usage=usage)
        metrics.update(entropy_metrics)
        if hasattr(self.router, "fatigue"):
            metrics.update(self.compute_fatigue_stats())
        metrics.update(self.compute_usage_distribution(indices, weights, usage=usage))
        metrics["routing_diversity_gini"] = self.compute_gini_coefficient(
            indices, weights, usage=usage
        )
        metrics["effective_experts"] = self.compute_effective_experts(
            indices, weights, entropy=entropy_metrics["expert_entropy"], usage=usage
        )
        metrics.update(self.compute_confidence_metrics(weights))
        if hasattr(self.router, "num_steps"):
            metrics["num_steps"] = self.router.num_steps.item()
        # Allow routers to inject their own custom metrics (e.g. stress, mean_k)
        if hasattr(self.router, "get_custom_metrics"):
            metrics.update(self.router.get_custom_metrics(indices, weights))
        return metrics

    def log_to_wandb(
        self, metrics: Dict[str, Any], step: int, prefix: str = "router"
    ) -> None:
        if not WANDB_AVAILABLE or not wandb.run:
            return

        scalar_metrics = {
            f"{prefix}/{key}": value
            for key, value in metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
        wandb.log(scalar_metrics, step=step)

        if "fatigue_per_expert" in metrics:
            wandb.log(
                {
                    f"{prefix}/fatigue_histogram": wandb.Histogram(
                        metrics["fatigue_per_expert"]
                    )
                },
                step=step,
            )

        if "usage_distribution" in metrics:
            wandb.log(
                {
                    f"{prefix}/usage_histogram": wandb.Histogram(
                        metrics["usage_distribution"]
                    )
                },
                step=step,
            )

        extra_log: Dict[str, Any] = {}
        for key, hist_name, scalar_name in (
            ("stress_per_expert", "stress_histogram", "stress"),
            ("ema_load_per_expert", "ema_load_histogram", "load"),
        ):
            if key in metrics:
                vals = metrics[key]
                extra_log[f"{prefix}/{hist_name}"] = wandb.Histogram(vals)
                extra_log.update(
                    {
                        f"{prefix}/expert_{i}_{scalar_name}": v
                        for i, v in enumerate(vals)
                    }
                )
        if extra_log:
            wandb.log(extra_log, step=step)
