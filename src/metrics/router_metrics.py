import torch
import numpy as np
from typing import Dict, Any

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
            device=torch.device(device) if isinstance(device, str) else device,
        )
        self.total_tokens = 0

    def update(self, token_ids: torch.Tensor, expert_indices: torch.Tensor):
        """
        Update global counts.
        token_ids: [batch, seq]
        expert_indices: [batch, seq, top_k]
        """
        with torch.no_grad():
            batch, seq, top_k = expert_indices.shape

            # Flatten inputs
            flat_tokens = token_ids.view(-1)  # [batch * seq]
            flat_experts = expert_indices.reshape(-1, top_k)  # [batch * seq, top_k]

            # Filter out padding tokens (-100 usually, or just < 0 and >= vocab_size)
            valid_mask = (flat_tokens >= 0) & (flat_tokens < self.vocab_size)
            valid_tokens = flat_tokens[valid_mask].cpu()
            valid_experts = flat_experts[valid_mask].cpu()

            if len(valid_tokens) == 0:
                return

            # Accumulate
            # We add 1 for each expert selection (a token selected top_k experts, so it adds 1 to each of those experts)
            # Flatten again for scatter/bincount
            expanded_tokens = valid_tokens.unsqueeze(1).expand(-1, top_k).flatten()
            flattened_experts = valid_experts.flatten()

            # Use 1D indices for bincount
            linear_indices = expanded_tokens * self.num_experts + flattened_experts
            counts = torch.bincount(
                linear_indices, minlength=self.vocab_size * self.num_experts
            )

            self.usage_counts += counts.view(self.vocab_size, self.num_experts)
            self.total_tokens += valid_tokens.numel()

    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute true H(E|T), H(E), Specialization Score, and Collapse Score.
        Returns empty dict if no tokens have been processed.
        """
        if self.total_tokens == 0:
            return {}

        with torch.no_grad():
            # 1. Active tokens mask (only consider tokens we've actually seen)
            token_counts = self.usage_counts.sum(dim=1)
            active_mask = token_counts > 0

            if not active_mask.any():
                return {}

            active_usage = self.usage_counts[active_mask].float()
            active_token_counts = token_counts[active_mask].float()

            # 2. P(E|T) - Probability of choosing expert E given token T
            p_e_given_t = active_usage / active_usage.sum(dim=1, keepdim=True)

            # 3. H(E|T) - Conditional Entropy
            entropy_e_given_t = -(p_e_given_t * torch.log(p_e_given_t + 1e-10)).sum(
                dim=1
            )

            # Weight by token frequency to get expected conditional entropy
            p_t = active_token_counts / active_token_counts.sum()
            expected_conditional_entropy = (p_t * entropy_e_given_t).sum()

            # 4. H(E) - Marginal Entropy
            total_expert_usage = self.usage_counts.sum(dim=0).float()
            p_e = total_expert_usage / total_expert_usage.sum()
            marginal_entropy = -(p_e * torch.log(p_e + 1e-10)).sum()

            # 5. Scores
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
    Comprehensive metrics tracker for Metabolic Router.

    Tracks:
    - Expert entropy and routing diversity
    - Fatigue statistics (mean, std, min, max, per-expert)
    - Usage distribution and counts
    - Gini coefficient for load balancing
    - Effective number of experts

    Usage:
        tracker = RouterMetricsTracker(router)
        metrics = tracker.compute_all_metrics(indices, weights)
        tracker.log_to_wandb(metrics, step=100)
    """

    def __init__(self, router):
        """
        Initialize metrics tracker.

        Args:
            router: MetabolicRouter instance to track
        """
        self.router = router
        self.num_experts = router.num_experts

        # Cache index tensor for Gini coefficient computation (avoid repeated allocations)
        self.gini_index = torch.arange(1, self.num_experts + 1, dtype=torch.float32)

    def compute_expert_entropy(
        self, indices: torch.Tensor, weights: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute Shannon entropy of expert routing distribution.

        Higher entropy = more diverse routing
        Lower entropy = concentrated routing

        Args:
            indices: Expert indices [batch, seq, top_k]
            weights: Routing weights [batch, seq, top_k]

        Returns:
            Dict with 'expert_entropy' and 'expert_entropy_normalized'
        """
        device = indices.device

        # float32: scatter_add_ requires matching dtypes (weights may be bfloat16 under FSDP)
        usage = torch.zeros(self.num_experts, device=device, dtype=torch.float32)
        flat_indices = indices.flatten()
        flat_weights = weights.flatten().to(torch.float32)
        usage.scatter_add_(0, flat_indices, flat_weights)

        # Normalize to probability distribution
        usage_prob = usage / (usage.sum() + 1e-10)

        # Shannon entropy: -sum(p * log(p))
        entropy = -(usage_prob * torch.log(usage_prob + 1e-10)).sum()

        # Normalized entropy: divide by max entropy (log(num_experts))
        max_entropy = np.log(self.num_experts)
        normalized_entropy = entropy / max_entropy

        return {
            "expert_entropy": entropy.item(),
            "expert_entropy_normalized": normalized_entropy.item(),
        }

    def compute_fatigue_stats(self) -> Dict[str, Any]:
        """
        Compute comprehensive fatigue statistics.

        Returns:
            Dict with mean, std, min, max, and per-expert fatigue
        """
        fatigue = self.router.fatigue

        return {
            "fatigue_mean": fatigue.mean().item(),
            "fatigue_std": fatigue.std().item(),
            "fatigue_min": fatigue.min().item(),
            "fatigue_max": fatigue.max().item(),
            "fatigue_per_expert": fatigue.clone().cpu().float().numpy(),
        }

    def compute_usage_distribution(
        self, indices: torch.Tensor, weights: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Compute expert usage counts and distribution.

        Args:
            indices: Expert indices [batch, seq, top_k]
            weights: Routing weights [batch, seq, top_k]

        Returns:
            Dict with usage_counts and usage_distribution
        """
        device = indices.device

        usage_counts = torch.zeros(self.num_experts, device=device, dtype=torch.float32)
        flat_indices = indices.flatten()
        flat_weights = weights.flatten().to(torch.float32)
        usage_counts.scatter_add_(0, flat_indices, flat_weights)

        # Normalize to distribution
        usage_dist = usage_counts / (usage_counts.sum() + 1e-10)

        return {
            "usage_counts": usage_counts.detach().cpu().float().numpy(),
            "usage_distribution": usage_dist.detach().cpu().float().numpy(),
        }

    def compute_gini_coefficient(
        self, indices: torch.Tensor, weights: torch.Tensor
    ) -> float:
        """
        Compute Gini coefficient for load balancing assessment.

        Gini = 0: Perfect balance (all experts used equally)
        Gini = 1: Perfect imbalance (one expert does everything)

        Args:
            indices: Expert indices [batch, seq, top_k]
            weights: Routing weights [batch, seq, top_k]

        Returns:
            Gini coefficient in [0, 1]
        """
        device = indices.device

        usage = torch.zeros(self.num_experts, device=device, dtype=torch.float32)
        flat_indices = indices.flatten()
        flat_weights = weights.flatten().to(torch.float32)
        usage.scatter_add_(0, flat_indices, flat_weights)

        # Sort usage in ascending order
        sorted_usage, _ = torch.sort(usage)

        # Compute Gini coefficient
        n = self.num_experts
        # Use cached index tensor (ensure device compatibility)
        index = self.gini_index.to(device)
        gini = (2 * (index * sorted_usage).sum()) / (n * sorted_usage.sum() + 1e-10) - (
            n + 1
        ) / n

        return gini.item()

    def compute_effective_experts(
        self, indices: torch.Tensor, weights: torch.Tensor
    ) -> float:
        """
        Compute effective number of experts (exponential of entropy).

        This metric answers: "How many experts are effectively being used?"

        Args:
            indices: Expert indices [batch, seq, top_k]
            weights: Routing weights [batch, seq, top_k]

        Returns:
            Effective number of experts in [1, num_experts]
        """
        entropy_dict = self.compute_expert_entropy(indices, weights)
        entropy = entropy_dict["expert_entropy"]
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
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Compute all metrics in one pass.

        Args:
            indices: Expert indices [batch, seq, top_k]
            weights: Routing weights [batch, seq, top_k]

        Returns:
            Complete metrics dictionary
        """
        metrics = {}

        # Expert entropy
        metrics.update(self.compute_expert_entropy(indices, weights))

        # Fatigue statistics (only for routers that track fatigue, e.g. MetabolicRouter)
        if hasattr(self.router, "fatigue"):
            metrics.update(self.compute_fatigue_stats())

        # Usage distribution
        metrics.update(self.compute_usage_distribution(indices, weights))

        # Routing diversity
        metrics["routing_diversity_gini"] = self.compute_gini_coefficient(
            indices, weights
        )
        metrics["effective_experts"] = self.compute_effective_experts(indices, weights)

        # Router confidence statistics
        metrics.update(self.compute_confidence_metrics(weights))

        # Step counter (only for routers that track steps)
        if hasattr(self.router, "num_steps"):
            metrics["num_steps"] = self.router.num_steps.item()

        return metrics

    def log_to_wandb(
        self, metrics: Dict[str, Any], step: int, prefix: str = "router"
    ) -> None:
        """
        Log metrics to Weights & Biases.

        Args:
            metrics: Metrics dictionary from compute_all_metrics()
            step: Training step number
            prefix: Metric name prefix (default: "router")
        """
        if not WANDB_AVAILABLE:
            return  # WandB not installed

        if not wandb.run:
            return  # WandB not initialized

        # Prepare scalar metrics for logging
        scalar_metrics = {}

        for key, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                scalar_metrics[f"{prefix}/{key}"] = value

        # Log scalars
        wandb.log(scalar_metrics, step=step)

        # Log histograms for array data
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

        # Specialization metrics (logged as scalars when present)
        for key in (
            "specialization_score",
            "collapse_score",
            "marginal_entropy",
            "conditional_entropy",
        ):
            if key in metrics:
                wandb.log({f"{prefix}/{key}": metrics[key]}, step=step)

    def to_dict(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert metrics to JSON-serializable dictionary (for checkpointing).

        Args:
            metrics: Metrics dictionary from compute_all_metrics()

        Returns:
            JSON-serializable dictionary
        """
        serializable = {}

        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable[key] = float(value)
            else:
                serializable[key] = value

        return serializable
