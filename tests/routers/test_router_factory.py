from src.routers import create_router
from src.routers.metabolic import MetabolicRouter
from src.routers.standard import StandardRouter


def test_create_router_ignores_unknown_kwargs_for_standard_router():
    router = create_router(
        router_type="standard",
        hidden_dim=32,
        num_experts=4,
        top_k=2,
        temperature=0.7,
        noise_std=0.05,
        eps=1e-3,
        ema_alpha=0.01,
        lambda_calib_step=600,
        tau_final=1.0,
        tau_anneal_steps=0,
    )
    assert isinstance(router, StandardRouter)
    assert router.temperature == 0.7
    assert router.top_k == 2


def test_create_router_keeps_metabolic_specific_kwargs():
    router = create_router(
        router_type="metabolic",
        hidden_dim=32,
        num_experts=4,
        top_k=2,
        lambda_metabolic=0.3,
        gamma_recovery=0.15,
        beta_cost=0.15,
        warmup_steps=1200,
        eps=1e-3,
    )
    assert isinstance(router, MetabolicRouter)
    assert router.lambda_metabolic == 0.3
    assert router.gamma_recovery == 0.15
    assert router.beta_cost == 0.15
    assert router.warmup_steps == 1200
