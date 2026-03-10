import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
import warnings

from src.configs import MetabolicRouterConfig
from src.core import RouterRegistry
from src.routers.base import BaseRouter
from src.metrics import RouterMetricsTracker
from src.project_types import RouterType

MIN_TEMPERATURE = 1e-3


@RouterRegistry.register(RouterType.METABOLIC.value)
class MetabolicRouter(BaseRouter):
    """
    Metabolic Router: fatigue-based load balancing without auxiliary loss.

    Formula:
        potential_i = cosine_sim(x, w_i) * g_i - λ * warmup_scale * softsign(F_i) + noise
        F_i = (1 - γ) * F_i + β * (U_i - 1/N)

    Where:
        g_i  - learnable per-expert magnitude (prototype_magnitude)
        F_i  - fatigue buffer per expert (EMA of excess usage)
        U_i  - fraction of tokens routed to expert i (uniform token count)
        λ    - fatigue penalty strength (lambda_metabolic)
        γ    - fatigue recovery rate (gamma_recovery)
        β    - fatigue accumulation rate (beta_cost)
    """

    def __init__(self, config: MetabolicRouterConfig):
        super().__init__(config)

        if config.top_k > config.num_experts:
            raise ValueError(
                f"top_k ({config.top_k}) cannot exceed num_experts ({config.num_experts})"
            )

        self.lambda_metabolic = config.lambda_metabolic
        self.gamma_recovery = config.gamma_recovery
        self.beta_cost = config.beta_cost
        self.warmup_steps = config.warmup_steps
        self.temperature = config.temperature
        self.noise_std = config.noise_std
        self.magnitude_min = config.magnitude_min
        self.magnitude_max = config.magnitude_max

        # Gate: cosine similarity between input and expert prototypes.
        # nn.Linear is FSDP-safe (vs raw Parameter - FSDP all-gather intercepts F.linear).
        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        nn.init.xavier_uniform_(self.gate.weight)

        # Per-expert learnable magnitude: decouples direction (specialization) from
        # scale (aggressiveness). Clamped to [magnitude_min, magnitude_max] during
        # forward to prevent any expert from dominating via unbounded scale.
        self.prototype_magnitude = nn.Parameter(torch.ones(config.num_experts))

        # Fatigue buffer: EMA of excess usage per expert.
        # Explicitly fp32 - under FSDP bf16, accumulated rounding errors grow as
        # O(T * ε_bf16) over thousands of steps via the (1-γ) EMA.
        self.register_buffer("fatigue", torch.zeros(self.num_experts))
        self.register_buffer("num_steps", torch.tensor(0, dtype=torch.long))

        # Deferred usage accumulators for gradient accumulation support.
        # Usage is applied once per optimizer step (in step()), not per forward pass.
        self.register_buffer("_pending_usage_sum", torch.zeros(self.num_experts))
        self.register_buffer("_pending_tokens", torch.tensor(0, dtype=torch.long))
        self._usage_pending = False
        # Python-int mirror of num_steps for warmup math - avoids GPU→CPU sync
        # (.item()) on every training forward. Kept in sync with the buffer in step().
        self._step_count: int = 0

        self.metrics_tracker = RouterMetricsTracker(self)

    def compute_alignment(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, p=2, dim=-1, eps=1e-8)
        w = F.normalize(self.gate.weight, p=2, dim=-1, eps=1e-8)
        # magnitude_max=0 disables clamping (useful for ablations).
        if self.magnitude_max > 0:
            mag = self.prototype_magnitude.clamp(
                min=self.magnitude_min, max=self.magnitude_max
            )
        else:
            mag = self.prototype_magnitude
        return F.linear(x, w * mag.unsqueeze(-1))

    def compute_routing_potential(
        self, alignment: torch.Tensor, noise_std: float = 0.0, lambda_scale: float = 1.0
    ) -> torch.Tensor:
        potential = alignment

        if self.lambda_metabolic > 0 and lambda_scale > 0:
            # Cast fatigue to fp32: under FSDP bf16 the buffer may be downcast,
            # losing precision in the penalty signal.
            fatigue_term = F.softsign(self.fatigue.float())
            potential = potential - (
                self.lambda_metabolic * lambda_scale * fatigue_term.view(1, 1, -1)
            )

        if noise_std > 0:
            # Gumbel noise: avoids per-call distribution object overhead vs torch.distributions
            u = torch.empty_like(potential).uniform_(1e-10, 1.0 - 1e-10)
            potential = potential + noise_std * (-torch.log(-torch.log(u)))

        return potential

    def _record_usage(self, indices: torch.Tensor) -> None:
        flat_indices = indices.flatten()
        batch_tokens = flat_indices.numel() // self.top_k

        # Uniform token weight (1/top_k per selection) rather than softmax weights.
        # Token count measures compute load (routing frequency) - the correct signal
        # for load balancing. Softmax weights add noise from logit magnitudes.
        # bincount without weights counts selections per expert; dividing by top_k
        # normalises to the same scale as 1/top_k-weighted bincount, without
        # allocating an O(batch*seq*top_k) constant tensor every forward pass.
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

    def update_fatigue(self, usage: torch.Tensor) -> None:
        """
        EMA fatigue update: F = (1 - γ) * F + β * (U - 1/N)

        Overloaded experts (U > 1/N) accumulate positive fatigue → routing penalty.
        Underloaded experts (U < 1/N) accumulate negative fatigue → routing bonus.
        Zero-sum by design: Σ (U_i - 1/N) = 0.
        """
        excess_usage = usage.float() - (1.0 / self.num_experts)
        with torch.no_grad():
            f = self.fatigue.float()
            f.mul_(1 - self.gamma_recovery).add_(self.beta_cost * excess_usage)
            self.fatigue.copy_(f)

    def forward(
        self,
        x: torch.Tensor,
        return_metrics: bool = False,
        noise_std: Optional[float] = None,
        temperature: Optional[float] = None,
        record_usage: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        alignment = self.compute_alignment(x)

        if noise_std is None:
            noise_std = self.noise_std if self.training else 0.0

        # Global λ warmup: ramp fatigue penalty 0 → λ over warmup_steps.
        # Lets the gate converge to good routing directions before fatigue feedback
        # kicks in - without this, fatigue can lock in a biased pattern in early steps.
        if self.warmup_steps > 0 and self.training:
            lambda_scale = min(1.0, self._step_count / self.warmup_steps)
        else:
            lambda_scale = 1.0

        potential = self.compute_routing_potential(alignment, noise_std, lambda_scale)

        top_k_values, top_k_indices = torch.topk(potential, self.top_k, dim=-1)

        temp = max(
            temperature if temperature is not None else self.temperature,
            MIN_TEMPERATURE,
        )
        weights = F.softmax(top_k_values / temp, dim=-1)

        if self.training and record_usage:
            self._record_usage(top_k_indices)

        metrics = None
        if return_metrics:
            metrics = self.metrics_tracker.compute_all_metrics(top_k_indices, weights)

        return weights, top_k_indices, metrics

    def step(self) -> None:
        """
        Apply pending usage to fatigue and increment step counter.
        Call after optimizer.step(), once per logical batch.
        """
        if not self._usage_pending:
            return

        with torch.no_grad():
            usage_avg = (
                self._pending_usage_sum.float()
                / self._pending_tokens.float().clamp(min=1)
            )
            self.update_fatigue(usage_avg)
            self.num_steps += 1
            self._step_count += 1
            self._usage_pending = False

    def compute_aux_loss(self) -> torch.Tensor:
        """Always zero - fatigue IS the load balancing mechanism."""
        return torch.tensor(0.0, device=self.fatigue.device)

    def reset_state(self) -> None:
        """Reset all expert state buffers to initial values."""
        with torch.no_grad():
            self.fatigue.zero_()
            self.num_steps.zero_()
            self._pending_usage_sum.zero_()
            self._pending_tokens.zero_()
            self._usage_pending = False
            self._step_count = 0

    def get_state(self) -> Dict[str, Any]:
        return {
            "fatigue": self.fatigue.clone(),
            "num_steps": self.num_steps.item(),
            "mean_fatigue": self.fatigue.mean().item(),
            "max_fatigue": self.fatigue.max().item(),
            "min_fatigue": self.fatigue.min().item(),
        }

    def state_dict(self, *args, **kwargs):
        state = super().state_dict(*args, **kwargs)
        state["_metabolic_metadata"] = {
            "num_steps": self.num_steps.item(),
            "lambda_metabolic": self.lambda_metabolic,
            "gamma_recovery": self.gamma_recovery,
            "beta_cost": self.beta_cost,
            "warmup_steps": self.warmup_steps,
            "temperature": self.temperature,
            "magnitude_min": self.magnitude_min,
            "magnitude_max": self.magnitude_max,
        }
        return state

    def load_state_dict(self, state_dict, strict=True):
        metadata = state_dict.get("_metabolic_metadata")

        if metadata is not None:
            state_dict_to_load = state_dict.copy()
            state_dict_to_load.pop("_metabolic_metadata")
        else:
            state_dict_to_load = state_dict
            metadata = {}

        super().load_state_dict(state_dict_to_load, strict=strict)

        if "num_steps" in metadata:
            self.num_steps.fill_(metadata["num_steps"])
            self._step_count = metadata["num_steps"]

        for key in ("lambda_metabolic", "gamma_recovery", "beta_cost", "warmup_steps"):
            if key in metadata:
                current_val = getattr(self, key, None)
                loaded_val = metadata[key]
                if current_val != loaded_val:
                    warnings.warn(
                        f"Router config mismatch: {key} = {current_val} (current) "
                        f"vs {loaded_val} (checkpoint). Using current value."
                    )
