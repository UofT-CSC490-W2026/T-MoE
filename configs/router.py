from dataclasses import dataclass
from .base import BaseConfig


@dataclass
class RouterConfig(BaseConfig):
    """Base router configuration."""

    router_type: str = "metabolic"  # Router type identifier

    # Core Dimensions
    hidden_dim: int = 256
    num_experts: int = 8
    top_k: int = 2

    # Exploration & Softmax
    noise_std: float = 0.1
    temperature: float = 1.0


@dataclass
class StandardRouterConfig(RouterConfig):
    """Configuration for standard softmax router (baseline)."""

    router_type: str = "standard"
    # Load balancing loss (auxiliary loss used by Switch Transformer, etc.)
    use_aux_loss: bool = True
    aux_loss_coef: float = 0.01

    # Capacity factor (for token dropping)
    capacity_factor: float = 1.25
    drop_tokens: bool = False


@dataclass
class MetabolicRouterConfig(RouterConfig):
    """Configuration for metabolic router.

    Implements upgraded equations from T-MoE research:
    - Equation 1: Heavy-Tailed & Hardware-Aware Potential with SoftSign
    - Equation 2: Age-Aware Fatigue Dynamics with warmup
    - Equation 3: Adaptive Cost Scaling
    """

    router_type: str = "metabolic"

    # Equation 1: Heavy-Tailed & Hardware-Aware Potential
    # z_i(x,t) = g·cos(x, W_i) - λ·SoftSign(F_i(t)) - μ·Dist(i)
    lambda_metabolic: float = 0.1  # λ: Metabolic pressure coefficient
    mu_silicon: float = 0.0  # μ: Silicon Tax coefficient (hardware distance penalty) should be 0 for T-MoE v1.

    # Alignment function configuration
    normalize_inputs: bool = True  # Enable cosine similarity via input L2 normalization
    normalize_weights: bool = True  # Normalize expert prototypes for cosine similarity (this is essential for cosine gating)

    # Equation 2: Age-Aware Fatigue Dynamics
    # F_i(t+1) = (1-γ)F_i(t) + η_i(t)·U_i(t)
    gamma_recovery: float = 0.01  # γ: Recovery rate (silence recovery)
    beta_cost: float = 0.04  # β_cost: Base activation cost
    warmup_steps: int = 100  # T_warmup: Warmup period for newborn experts

    # Equation 3: Adaptive Cost Scaling (n_start auto-set to num_experts at init)


@dataclass
class TopKRouterConfig(RouterConfig):
    """Top-K router: standard softmax + top-k selection, no aux loss."""

    router_type: str = "topk"
    use_aux_loss: bool = False
    aux_loss_coef: float = 0.0


@dataclass
class SwitchRouterConfig(StandardRouterConfig):
    """Switch (Top-1) router: standard router with top_k=1."""

    router_type: str = "switch"
    top_k: int = 1


@dataclass
class DynMoERouterConfig(RouterConfig):
    """DynMoE-style top-any router with sigmoid gating."""

    router_type: str = "dynmoe"
    gate_threshold: float = 0.5
    use_aux_loss: bool = False
    aux_loss_coef: float = 0.0
