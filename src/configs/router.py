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
    router_type: RouterType = RouterType.TOPK_ROUTER
    use_aux_loss: bool = False


@dataclass
class SwitchRouterConfig(StandardRouterConfig):
    router_type: RouterType = RouterType.SWITCH
    top_k: int = 1


@dataclass
class MetabolicRouterConfig(BaseRouterConfig):
    """
    Configuration for MetabolicRouter v6.

    Routing potential:
        z_i = cos(x, W_i) - λ_eff(t) · tanh(F_i(t) / F_s)

    Fatigue update (one-sided):
        F_i(t+1) = (1 - γ) · F_i(t) + β · max(0, U_i(t) - τ/N)

    Parameters
    ----------
    lambda_metabolic : float
        Penalty scale λ. In cosine space [-1, 1]: trained gate advantages
        ≈ 0.3–0.6; λ=0.3 is conservative, allowing specialisation while
        still penalising extreme overload.
    gamma_recovery : float
        Fatigue decay rate γ. Memory horizon ≈ 1/γ steps.
    beta_cost : float
        Fatigue accumulation rate β. At equilibrium for an expert at
        exactly τ/N excess: F_eq = β/γ * excess.
    tau_specialization : float
        Specialisation tolerance τ > 1. Experts can take up to τ × fair-share
        tokens freely with zero fatigue accumulation. τ=2 means 2× fair share
        is penalty-free.
    F_scale : float
        Fatigue scale F_s for tanh normalisation. tanh(F_i / F_s) saturates
        at ±1; F_s sets the "half-saturation" point.
    warmup_steps : int
        Ramp λ_eff from 0 → λ over this many optimizer steps before metabolic
        penalty is active. Stagger from LR warmup to let the gate find
        semantic directions before fatigue fires.
    """

    router_type: RouterType = RouterType.METABOLIC

    lambda_metabolic: float = 0.3
    gamma_recovery: float = 0.15
    beta_cost: float = 0.15
    tau_specialization: float = 2.0
    F_scale: float = 0.5
    warmup_steps: int = 1200


@dataclass
class DeepSeekRouterConfig(BaseRouterConfig):
    router_type: RouterType = RouterType.DEEPSEEK
    temperature: float = 1.0
    noise_std: float = 0.0
    bias_update_rate: float = 1e-3
    use_sigmoid: bool = False


@dataclass
class ExpertChoiceRouterConfig(BaseRouterConfig):
    router_type: RouterType = RouterType.EXPERT_CHOICE
    temperature: float = 1.0


@dataclass
class StressCorrectedRouterConfig(BaseRouterConfig):
    """
    SPAR Router configuration.

    Selection:     z_i = cos(x,W_i) - λ · max(0, L_i - 1/N)
    Output weight: w_i = softmax(cos(x,W_i) / τ_t)   [over top-k selected experts]
    Lambda calib:  λ = min(σ_cos · N, 5.0)   [auto at step lambda_calib_step]
                   (equiv. σ_cos / mean(L) at equilibrium when mean(L) = 1/N)

    Free hyperparameters: τ (one value). λ is data-derived, α is standard.
    τ anneals linearly from temperature → tau_final over tau_anneal_steps optimizer steps.
    Set tau_anneal_steps=0 (default) or tau_final==temperature to disable annealing.
    """

    router_type: RouterType = RouterType.STRESS_CORRECTED
    temperature: float = 0.5  # τ_0 — initial output weight sharpness
    noise_std: float = 0.05  # Gumbel exploration noise during training
    eps: float = 1e-6  # numerical floor
    ema_alpha: float = 0.01  # EMA decay for load tracking (~100-step window)
    lambda_calib_step: int = (
        600  # warmup_steps + 200; calibrate after LR warmup completes
    )
    tau_final: float = 0.5  # τ at end of annealing; if == temperature, no annealing
    tau_anneal_steps: int = (
        0  # steps over which τ anneals from temperature → tau_final; 0 = disabled
    )
    noise_anneal_steps: int = (
        0  # steps over which noise_std anneals to 0; 0 = disabled (fixed noise_std)
    )
    init_from_data: bool = (
        False  # if True, initialize W from k-means centroids of layer activations
    )
