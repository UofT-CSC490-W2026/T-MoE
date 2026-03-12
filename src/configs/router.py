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

    # λ in cosine space [-1, 1]: trained gate advantages ≈ 0.3–0.6 on real data.
    # λ=1.0 exceeds the practical maximum → provably forces rebalancing via raw F_i.
    lambda_metabolic: float = 1.0
    gamma_recovery: float = 0.05
    beta_cost: float = 0.4
    # Global λ warmup: ramp fatigue penalty 0 → λ over warmup_steps.
    # Prevents gate locking into biased routing before fatigue builds up.
    warmup_steps: int = 400


@dataclass
class DynMoERouterConfig(BaseRouterConfig):
    temperature: float = 1.0
    gate_threshold: float = 0.5
    use_aux_loss: bool = False
    aux_loss_coef: float = 0.01
