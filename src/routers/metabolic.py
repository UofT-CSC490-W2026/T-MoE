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

# Constants
MIN_TEMPERATURE = 1e-3  # Minimum temperature to prevent division by zero in softmax


@RouterRegistry.register(RouterType.METABOLIC.value)
class MetabolicRouter(BaseRouter):
    """
    Metabolic Router with Heavy-Tailed Fatigue Dynamics.
    Implements:
    - Equation 1: Heavy-Tailed & Hardware-Aware Potential (SoftSign activation)
    - Equation 2: Age-Aware Fatigue Dynamics with newborn warmup
    - Equation 3: Adaptive Cost Scaling
    """

    def __init__(self, config: MetabolicRouterConfig):
        super().__init__(config)

        # Validate top_k <= num_experts
        if config.top_k > config.num_experts:
            raise ValueError(
                f"top_k ({config.top_k}) cannot exceed num_experts ({config.num_experts})"
            )

        # Metabolic Parameters
        self.lambda_metabolic = config.lambda_metabolic
        self.mu_silicon = config.mu_silicon
        self.gamma_recovery = config.gamma_recovery
        self.beta_cost = config.beta_cost
        self.warmup_steps = config.warmup_steps
        self.temperature = config.temperature

        # Alignment Configuration
        self.normalize_inputs = config.normalize_inputs
        self.normalize_weights = config.normalize_weights

        # Magnitude clamping (prevents expert dominance via unbounded g_i)
        self.magnitude_min = config.magnitude_min
        self.magnitude_max = config.magnitude_max

        # Adaptive Cost Scaling (Equation 3)
        self.n_start = config.num_experts  # Initial expert count for scaling

        # Track active expert count (for dynamic pruning support)
        self.register_buffer(
            "n_active", torch.tensor(config.num_experts, dtype=torch.long)
        )

        # Learnable Router Gate
        # We use nn.Linear instead of a raw nn.Parameter for the expert
        # prototypes.  Under FSDP with use_orig_params=True, raw parameter
        # references (self.prototypes) can point to empty placeholder tensors
        # between all-gather operations.  nn.Linear's forward path goes
        # through F.linear which FSDP intercepts correctly, ensuring the
        # weight is always materialized during the forward pass.
        #
        # The gate maps hidden_dim → num_experts (no bias) and is
        # functionally equivalent to: matmul(x, prototypes.T)
        self.gate = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        nn.init.xavier_uniform_(self.gate.weight)

        # Per-expert learnable magnitude for cosine similarity routing.
        # Decouples direction (which tokens to specialize on, via gate.weight)
        # from magnitude (how aggressively to claim tokens). Fatigue prevents
        # any expert from dominating even with high learned magnitude.
        if self.normalize_weights:
            self.prototype_magnitude = nn.Parameter(torch.ones(config.num_experts))
        else:
            self.prototype_magnitude = None

        # Exploration Parameters
        self.noise_std = config.noise_std

        # Expert State Buffers
        self.register_buffer("fatigue", torch.zeros(self.num_experts))
        self.register_buffer("birth_step", torch.zeros(self.num_experts))
        self.register_buffer("num_steps", torch.tensor(0, dtype=torch.long))

        # Usage tracking for deferred fatigue update (gradient accumulation support)
        # $O(E)$ accumulators replacing $O(N)$ concatenations
        self.register_buffer("_pending_usage_sum", torch.zeros(self.num_experts))
        self.register_buffer("_pending_tokens", torch.tensor(0, dtype=torch.long))
        self._usage_pending = False

        # Initialize Metrics Tracker
        self.metrics_tracker = RouterMetricsTracker(self)

        # Cache expert IDs tensor for hardware distance computations (avoid repeated allocations)
        self.register_buffer("expert_ids", torch.arange(self.num_experts))

        # Cache hardware distance vector (constant for single-device, can be overridden for multi-device)
        # Hardware Distance (Placeholder - NOOP)
        # TODO: Implement proper hardware topology when needed
        # For now, this is disabled (all zeros = no penalty)
        self.register_buffer("hardware_distance", torch.zeros(self.num_experts))

    def compute_alignment(self, x: torch.Tensor) -> torch.Tensor:
        """Compute alignment between input and expert prototypes via self.gate (FSDP-safe)."""
        if self.normalize_inputs:
            x = F.normalize(x, p=2, dim=-1, eps=1e-8)

        if self.normalize_weights:
            w = self.gate.weight  # [num_experts, hidden_dim]
            w_normalized = F.normalize(w, p=2, dim=-1, eps=1e-8)
            if self.prototype_magnitude is not None:
                # Clamp magnitude to [min, max] to prevent expert dominance.
                # This is a structural bound — no aux loss needed.
                # magnitude_max=0 disables clamping entirely.
                if self.magnitude_max > 0:
                    mag = self.prototype_magnitude.clamp(
                        min=self.magnitude_min, max=self.magnitude_max
                    )
                else:
                    mag = self.prototype_magnitude
                # Per-expert magnitude: [num_experts] → [num_experts, 1] for broadcasting
                w_normalized = w_normalized * mag.unsqueeze(-1)
            alignment = F.linear(x, w_normalized)
        else:
            alignment = self.gate(x)

        return alignment

    def compute_routing_potential(
        self, alignment: torch.Tensor, noise_std: float = 0.0
    ) -> torch.Tensor:
        """
        Compute routing potentials for all experts (Equation 1).
        """
        potential = alignment

        # 1. Metabolic Fatigue Penalty (Heavy-Tailed with SoftSign)
        # Cast to float32 before SoftSign: under FSDP mixed precision the fatigue
        # buffer is cast to bfloat16, which has insufficient range for accumulated
        # fatigue values across thousands of steps.
        if self.lambda_metabolic > 0:
            fatigue_term = F.softsign(self.fatigue.float())
            potential = potential - (
                self.lambda_metabolic * fatigue_term.view(1, 1, -1)
            )

        # 2. Silicon Tax (Hardware Distance Penalty)
        if self.mu_silicon > 0:
            dist_vector = self.hardware_distance
            potential = potential - (self.mu_silicon * dist_vector.view(1, 1, -1))

        # 3. Exploration Noise (Gumbel for differentiable sampling)
        # Allow noise in eval mode for exploration studies (no training check)
        if noise_std > 0:
            # Gumbel(0,1) = -log(-log(U)), U ~ Uniform(0,1)
            # Equivalent to using torch.distributions.Gumbel but avoids
            # per-call distribution object creation overhead
            u = torch.empty_like(potential).uniform_(1e-10, 1.0 - 1e-10)
            noise = -torch.log(-torch.log(u))
            potential = potential + (noise * noise_std)

        return potential

    def _record_usage(self, indices: torch.Tensor, weights: torch.Tensor) -> None:
        """
        Record expert usage for deferred fatigue update.

        This method accumulates usage across forward passes within a logical batch
        (i.e., across gradient accumulation steps). Call step() after optimizer.step()
        to apply the accumulated usage to fatigue.

        Args:
            indices: Expert indices [batch, seq, top_k]
            weights: Routing weights [batch, seq, top_k]
        """
        flat_indices = indices.flatten()
        flat_weights = weights.flatten().to(torch.float32)  # buffers are float32
        batch_tokens = flat_indices.numel() // self.top_k

        # Pre-aggregate usage inside the forward pass footprint
        usage = torch.bincount(
            flat_indices, weights=flat_weights, minlength=self.num_experts
        )

        # usage is float32 (from bincount with float32 weights).
        # _pending_usage_sum may be bfloat16 under FSDP; cast to match so
        # PyTorch does not silently downcast the accumulation.
        if self._usage_pending:
            self._pending_usage_sum.add_(usage.to(self._pending_usage_sum.dtype))
            self._pending_tokens.add_(batch_tokens)
        else:
            self._pending_usage_sum.copy_(usage)
            self._pending_tokens.fill_(batch_tokens)
            self._usage_pending = True

    def update_fatigue(self, usage: torch.Tensor) -> None:
        """
        Update expert fatigue with age-aware dynamics (Equation 2).

        Args:
            usage: Pre-aggregated and normalized float usage per expert.
        """
        device = self.fatigue.device

        # 1. Age-Aware Cost Scaling (prevents newborn apoptosis)
        # η_i(t) = β_cost · min(1.0, (t - birth_i) / T_warmup)
        # Apply warmup ONLY to dynamically spawned experts (birth_step > 0).
        # Initial experts (birth_step == 0) start fully mature.
        age_factor = torch.ones(self.num_experts, device=device)
        if self.warmup_steps > 0:
            newborn_mask = self.birth_step > 0
            if newborn_mask.any():
                age = (self.num_steps + 1 - self.birth_step[newborn_mask]).float()
                age = age.clamp(min=1)  # defensive: guard against zero/negative age
                age_factor[newborn_mask] = torch.clamp(age / self.warmup_steps, max=1.0)

        eta_i = self.beta_cost * age_factor

        # 2. Adaptive Cost Scaling (Equation 3)
        # η_eff = η_base · (N_current / N_start)
        n_current = self.n_active.item()  # Use tracked active expert count
        eta_eff = eta_i * (n_current / max(self.n_start, 1))

        # 3. Differential Fatigue Update (tracks EXCESS usage relative to fair share)
        # F_i(t+1) = (1-γ)F_i(t) + η_eff·(U_i(t) - 1/N)
        # - Balanced expert (U=1/N): fatigue → 0  (no interference)
        # - Overloaded expert (U>1/N): fatigue → positive (penalty)
        # - Neglected expert (U<1/N): fatigue → negative (bonus)
        #
        # Arithmetic is always done in float32 regardless of buffer storage dtype.
        # Under FSDP mixed precision, self.fatigue is a bfloat16 buffer. Because
        # fatigue errors accumulate multiplicatively over T steps via the (1-γ) EMA,
        # even small per-step rounding errors (bfloat16 ≈ 0.4% relative error) grow
        # to ~O(T · ε_bf16 · Δ) over a 3000-step run. Computing in float32 and
        # writing back keeps per-step error below float32 machine epsilon (~1e-7).
        excess_usage = usage.float() - (1.0 / self.num_experts)
        with torch.no_grad():
            f = self.fatigue.float()
            f.mul_(1 - self.gamma_recovery).add_(eta_eff * excess_usage)
            self.fatigue.copy_(f)

    def forward(
        self,
        x: torch.Tensor,
        return_metrics: bool = False,
        noise_std: Optional[float] = None,
        temperature: Optional[float] = None,
        record_usage: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        """
        End-to-end routing forward pass.

        Args:
            x: Input tensor [batch, seq, hidden_dim]
            return_metrics: Whether to compute and return routing metrics
            noise_std: Optional override for exploration noise standard deviation.
                       If None, uses self.noise_std during training and 0.0 during eval.
            temperature: Optional override for softmax temperature.
                        If None, uses self.temperature. Useful for stochastic eval routing.
            record_usage: Whether to record usage for fatigue updates. Set to False
                         when collecting metrics from a separate forward pass to avoid
                         double-counting usage.

        Returns:
            weights: Routing probabilities [batch, seq, top_k]
            indices: Selected expert indices [batch, seq, top_k]
            metrics: Optional dictionary of routing metrics
        """
        # 1. Compute Alignment (Semantic similarity)
        alignment = self.compute_alignment(x)

        # 2. Compute Routing Potential (Apply penalties & noise)
        if noise_std is None:
            noise_std = self.noise_std if self.training else 0.0

        potential = self.compute_routing_potential(alignment, noise_std)

        # 3. Top-K Expert Selection
        top_k_values, top_k_indices = torch.topk(potential, self.top_k, dim=-1)

        # 4. Normalize Weights (Softmax)
        temp = temperature if temperature is not None else self.temperature
        # Ensure temperature is bounded to prevent division by zero and overflow
        temp = max(temp, MIN_TEMPERATURE)
        weights = F.softmax(top_k_values / temp, dim=-1)

        # 5. Record Usage (only during training and if requested)
        if self.training and record_usage:
            self._record_usage(top_k_indices, weights)

        # 6. Prepare Metrics
        metrics = None
        if return_metrics:
            metrics = self.metrics_tracker.compute_all_metrics(top_k_indices, weights)

        return weights, top_k_indices, metrics

    def step(self) -> None:
        """
        Apply pending usage to fatigue and increment step counter.

        **IMPORTANT**: Call this method after `optimizer.step()` to ensure
        fatigue updates occur once per logical batch (not per forward pass).
        This is critical for correct behavior with gradient accumulation.

        Example:
            ```python
            for batch in dataloader:
                output, loss = model(batch)
                loss.backward()

                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                    # Update fatigue after optimizer step
                    model.router.step()  # or layer.router.step() for MoE layers
            ```
        """
        if not self._usage_pending:
            # No usage recorded since last step (eval mode or first call)
            return

        # Apply accumulated usage to fatigue
        with torch.no_grad():
            # Promote to float32 before division: _pending_usage_sum may be bfloat16
            # under FSDP and dividing in bfloat16 loses 2-3 decimal places of precision
            # in the usage fraction (which is small: ~1/N per expert).
            usage_avg = self._pending_usage_sum.float() / max(
                self._pending_tokens.item(), 1
            )

            # update_fatigue now accepts pre-aggregated float usage
            self.update_fatigue(usage_avg)

            # Increment global step counter
            self.num_steps += 1

            # Clear pending usage flag
            self._usage_pending = False

    def register_birth(self, expert_id: int) -> None:
        """
        Register the birth of a new expert for age-aware warmup tracking.

        This method is intended for future dynamic expert spawning functionality.
        Currently not integrated into the main training loop but provided as a
        public API for external management of expert lifecycles.

        Args:
            expert_id: ID of the newborn expert

        Note:
            For dynamic MoE with expert birth/death, call this when adding new experts
            to ensure proper age-aware warmup behavior.
        """
        with torch.no_grad():
            self.birth_step[expert_id] = self.num_steps

    def compute_aux_loss(self) -> torch.Tensor:
        """
        MetabolicRouter does not use auxiliary load-balancing loss.

        Load balancing is achieved entirely through fatigue dynamics (Equation 2),
        which is the core design thesis of this router. Adding differentiable
        auxiliary loss would undermine the fatigue-based feedback control and
        reintroduce the very mechanism this router was designed to replace.

        Returns:
            Zero tensor on the correct device
        """
        return torch.tensor(0.0, device=self.fatigue.device)

    def reset_state(self) -> None:
        """Reset all expert state buffers to initial values."""
        with torch.no_grad():
            self.fatigue.zero_()
            self.birth_step.zero_()
            self.num_steps.zero_()
            # Clear pending usage
            self._pending_usage_sum.zero_()
            self._pending_tokens.zero_()
            self._usage_pending = False

    def get_state(self) -> Dict[str, Any]:
        """
        Get a router state for logging/checkpointing.
        """
        return {
            "fatigue": self.fatigue.clone(),
            "birth_step": self.birth_step.clone(),
            "num_steps": self.num_steps.item(),
            "mean_fatigue": self.fatigue.mean().item(),
            "max_fatigue": self.fatigue.max().item(),
            "min_fatigue": self.fatigue.min().item(),
        }

    def state_dict(self, *args, **kwargs):
        """
        Enhanced state_dict with metabolic-specific metadata.

        Preserves all router state for complete checkpointing including:
        - Model parameters (prototypes)
        - State buffers (fatigue, birth_step, num_steps)
        - Metabolic configuration for reproducibility

        Returns:
            State dictionary with metabolic metadata
        """
        state = super().state_dict(*args, **kwargs)

        # Add metabolic-specific metadata for reproducibility
        state["_metabolic_metadata"] = {
            "num_steps": self.num_steps.item(),
            "n_start": self.n_start,
            "lambda_metabolic": self.lambda_metabolic,
            "mu_silicon": self.mu_silicon,
            "gamma_recovery": self.gamma_recovery,
            "beta_cost": self.beta_cost,
            "warmup_steps": self.warmup_steps,
            "temperature": self.temperature,
            "normalize_inputs": self.normalize_inputs,
            "normalize_weights": self.normalize_weights,
            "magnitude_min": self.magnitude_min,
            "magnitude_max": self.magnitude_max,
        }

        return state

    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dict with metabolic metadata restoration.

        Args:
            state_dict: State dictionary to load
            strict: Whether to strictly enforce key matching
        """
        # Extract metadata without mutating the input dict
        metadata = state_dict.get("_metabolic_metadata")

        if metadata is not None:
            # Metadata present: must copy to remove it for strict loading
            state_dict_to_load = state_dict.copy()
            state_dict_to_load.pop("_metabolic_metadata")
        else:
            # No metadata: load directly (avoids copy overhead)
            state_dict_to_load = state_dict
            metadata = {}

        # Load standard state (parameters and buffers)
        super().load_state_dict(state_dict_to_load, strict=strict)

        # Restore metabolic-specific state
        if "num_steps" in metadata:
            self.num_steps.fill_(metadata["num_steps"])

        # Validate configuration consistency (warn if mismatch)
        config_keys = [
            "lambda_metabolic",
            "gamma_recovery",
            "beta_cost",
            "warmup_steps",
        ]
        for key in config_keys:
            if key in metadata:
                current_val = getattr(self, key, None)
                loaded_val = metadata[key]
                if current_val != loaded_val:
                    warnings.warn(
                        f"Router config mismatch: {key} = {current_val} (current) "
                        f"vs {loaded_val} (checkpoint). Using current value."
                    )
