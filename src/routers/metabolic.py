import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional
import warnings

from configs import MetabolicRouterConfig
from src.core import RouterRegistry
from src.routers.base import BaseRouter
from src.metrics import RouterMetricsTracker


@RouterRegistry.register("metabolic")
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

        # Adaptive Cost Scaling (Equation 3)
        self.n_start = config.num_experts  # Initial expert count for scaling

        # Track active expert count (for dynamic pruning support)
        self.register_buffer(
            "n_active", torch.tensor(config.num_experts, dtype=torch.long)
        )

        # Learnable Router Prototypes
        self.prototypes = nn.Linear(config.hidden_dim, config.num_experts, bias=False)
        nn.init.xavier_uniform_(
            self.prototypes.weight
        )  # TODO can we use Kaiming initialization instead?

        # Apply weight normalization for automatic L2 normalization
        if self.normalize_weights:
            self.prototypes = nn.utils.parametrizations.weight_norm(
                self.prototypes, name="weight", dim=1
            )

        # Exploration Parameters
        self.noise_std = config.noise_std

        # Expert State Buffers
        self.register_buffer("fatigue", torch.zeros(self.num_experts))
        self.register_buffer("birth_step", torch.zeros(self.num_experts))
        self.register_buffer("num_steps", torch.tensor(0, dtype=torch.long))

        # Initialize Metrics Tracker
        self.metrics_tracker = RouterMetricsTracker(self)

        # Cache expert IDs tensor for hardware distance computations (avoid repeated allocations)
        self.register_buffer("expert_ids", torch.arange(self.num_experts))

        # Cache hardware distance vector (constant for single-device, can be overridden for multi-device)
        self.register_buffer("hardware_distance", torch.zeros(self.num_experts))

    def compute_alignment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute alignment between input and expert prototypes.
        """
        # Normalize input for cosine similarity
        if self.normalize_inputs:
            x = F.normalize(x, p=2, dim=-1)

        # Compute alignment (weights are automatically normalized by weight_norm if enabled in __init__)
        alignment = self.prototypes(x)

        return alignment

    def compute_routing_potential(
        self, alignment: torch.Tensor, noise_std: float = 0.0
    ) -> torch.Tensor:
        """
        Compute routing potentials for all experts (Equation 1).
        """
        potential = alignment

        # 1. Metabolic Fatigue Penalty (Heavy-Tailed with SoftSign)
        if self.lambda_metabolic > 0:
            fatigue_term = F.softsign(self.fatigue)
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
            # Standard Gumbel distribution
            gumbel_uniform = torch.rand_like(potential)
            noise = -torch.log(-torch.log(gumbel_uniform + 1e-20) + 1e-20)
            potential = potential + (noise * noise_std)

        return potential

    def update_fatigue(self, indices: torch.Tensor, weights: torch.Tensor) -> None:
        """
        Update expert fatigue with age-aware dynamics (Equation 2).
        """
        device = self.fatigue.device

        # 1. Compute usage U_i(t) from routing weights using bincount
        # Use raw counts (no normalization) to preserve fatigue signal strength
        flattened_expert_indices = indices.flatten()
        flattened_routing_weights = weights.flatten()
        usage = torch.bincount(
            flattened_expert_indices,
            weights=flattened_routing_weights,
            minlength=self.num_experts,
        )

        # 2. Age-Aware Cost Scaling (prevents newborn apoptosis)
        # η_i(t) = β_cost · min(1.0, (t - birth_i) / T_warmup)
        # Use num_steps + 1 to account for current step (prevents free first step)
        if self.warmup_steps > 0:
            age = (self.num_steps + 1 - self.birth_step).float()
            age_factor = torch.clamp(age / self.warmup_steps, min=0.0, max=1.0)
        else:
            age_factor = torch.ones(self.num_experts, device=device)

        eta_i = self.beta_cost * age_factor

        # 3. Adaptive Cost Scaling (Equation 3)
        # η_eff = η_base · (N_current / N_start)
        n_current = self.n_active.item()  # Use tracked active expert count
        eta_eff = eta_i * (n_current / max(self.n_start, 1))

        # 4. Fatigue Update: F_i(t+1) = (1-γ)F_i(t) + η_eff·U_i(t)
        with torch.no_grad():
            self.fatigue.mul_(1 - self.gamma_recovery)  # Exponential recovery
            self.fatigue.add_(eta_eff * usage)  # Accumulate from usage

    def forward(
        self,
        x: torch.Tensor,
        return_metrics: bool = False,
        noise_std: Optional[float] = None,
        temperature: Optional[float] = None,
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
        weights = F.softmax(top_k_values / temp, dim=-1)

        # 5. Update Fatigue (only during training)
        if self.training:
            self.update_fatigue(top_k_indices, weights)
            # Increment global step counter (used for age-aware warmup)
            with torch.no_grad():
                self.num_steps += 1

        # 6. Prepare Metrics
        metrics = None
        if return_metrics:
            metrics = self.metrics_tracker.compute_all_metrics(top_k_indices, weights)

        return weights, top_k_indices, metrics

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
