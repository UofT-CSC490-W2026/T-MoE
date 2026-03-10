from __future__ import annotations

from dataclasses import dataclass

from src.project_types import RouterType


@dataclass
class BaseRouterConfig:
    hidden_dim: int
    num_experts: int = 4
    top_k: int = 1
    router_type: RouterType = RouterType.STANDARD


@dataclass
class StandardRouterConfig(BaseRouterConfig):
    router_type: RouterType = RouterType.STANDARD
    temperature: float = 1.0
    noise_std: float = 0.0
    use_aux_loss: bool = False
    aux_loss_coef: float = 0.01


@dataclass
class TopKRouterConfig(StandardRouterConfig):
    use_aux_loss: bool = False


@dataclass
class SwitchRouterConfig(StandardRouterConfig):
    top_k: int = 1


@dataclass
class MetabolicRouterConfig(BaseRouterConfig):
    router_type: RouterType = RouterType.METABOLIC
    temperature: float = 1.0
    noise_std: float = 0.1

    # Metabolic parameters
    lambda_metabolic: float = 0.1
    gamma_recovery: float = 0.01
    beta_cost: float = 0.04
    # Global λ warmup: ramp fatigue penalty 0 → λ over these steps.
    # Prevents the gate locking into biased routing before fatigue builds up.
    warmup_steps: int = 100

    # Prototype magnitude clamp (prevents expert dominance via unbounded g_i).
    # Set magnitude_max=0 to disable clamping entirely (useful for ablations).
    magnitude_min: float = 0.1
    magnitude_max: float = 5.0


@dataclass
class DynMoERouterConfig(BaseRouterConfig):
    temperature: float = 1.0
    gate_threshold: float = 0.5
    use_aux_loss: bool = False
    aux_loss_coef: float = 0.01
