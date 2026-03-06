from __future__ import annotations

from dataclasses import dataclass

from src.project_types import RouterType


@dataclass
class BaseRouterConfig:
    """Base configuration shared by all routers."""

    hidden_dim: int
    num_experts: int = 4
    top_k: int = 1
    router_type: RouterType = RouterType.STANDARD


@dataclass
class StandardRouterConfig(BaseRouterConfig):
    """Standard Top-K router configuration."""

    router_type: RouterType = RouterType.STANDARD
    temperature: float = 1.0
    noise_std: float = 0.0
    use_aux_loss: bool = False
    aux_loss_coef: float = 0.01


@dataclass
class TopKRouterConfig(StandardRouterConfig):
    """Top-K router: same as Standard but no load-balancing aux loss."""

    use_aux_loss: bool = False


@dataclass
class SwitchRouterConfig(StandardRouterConfig):
    """Switch (Top-1) router: forces top_k=1."""

    top_k: int = 1


@dataclass
class MetabolicRouterConfig(BaseRouterConfig):
    """Metabolic Router configuration with fatigue dynamics."""

    router_type: RouterType = RouterType.METABOLIC
    temperature: float = 1.0
    noise_std: float = 0.1

    # Metabolic parameters
    lambda_metabolic: float = 0.1
    mu_silicon: float = 0.0
    gamma_recovery: float = 0.01
    beta_cost: float = 0.04
    warmup_steps: int = 100
    normalize_inputs: bool = True
    normalize_weights: bool = True


@dataclass
class DynMoERouterConfig(BaseRouterConfig):
    """DynMoE-style sigmoid-gate router configuration."""

    temperature: float = 1.0
    gate_threshold: float = 0.5
    use_aux_loss: bool = False
    aux_loss_coef: float = 0.01
