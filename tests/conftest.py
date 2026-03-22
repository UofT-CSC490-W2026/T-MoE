import pytest
import torch
from typing import Tuple
from src.configs.router import MetabolicRouterConfig


@pytest.fixture
def device() -> torch.device:
    """Auto-detect available device for device-agnostic testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture
def standard_config() -> MetabolicRouterConfig:
    """Standard metabolic router configuration for testing (v6 formulation)."""
    return MetabolicRouterConfig(
        hidden_dim=256,
        num_experts=8,
        top_k=2,
        lambda_metabolic=0.5,
        gamma_recovery=0.15,
        beta_cost=0.15,
        tau_specialization=2.0,
        F_scale=0.5,
        warmup_steps=100,
    )


@pytest.fixture
def minimal_config() -> MetabolicRouterConfig:
    """Minimal configuration for edge case testing."""
    return MetabolicRouterConfig(
        hidden_dim=64,
        num_experts=2,
        top_k=1,
        lambda_metabolic=0.0,
        gamma_recovery=0.0,
        beta_cost=0.0,
        tau_specialization=1.0,
        F_scale=1.0,
        warmup_steps=0,
    )


@pytest.fixture
def router(standard_config, device):
    """Initialized metabolic router instance."""
    from src.routers.metabolic import MetabolicRouter

    router = MetabolicRouter(standard_config)
    return router.to(device)


@pytest.fixture
def test_input(device) -> torch.Tensor:
    """Standard test input tensor [batch=2, seq=4, hidden=256]."""
    return torch.randn(2, 4, 256, device=device)


@pytest.fixture(
    params=[
        (1, 1, 64),  # Minimal: single token
        (2, 4, 256),  # Standard: small batch
        (8, 16, 512),  # Large: bigger hidden dim
    ]
)
def parametric_input(request, device) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Parametric test inputs with various shapes."""
    batch, seq, hidden = request.param
    return torch.randn(batch, seq, hidden, device=device), (batch, seq, hidden)


@pytest.fixture
def zero_fatigue_router(standard_config, device):
    """Router with fatigue explicitly reset to zero."""
    from src.routers.metabolic import MetabolicRouter

    router = MetabolicRouter(standard_config).to(device)
    router.reset_state()
    return router
