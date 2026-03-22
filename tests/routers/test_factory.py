import pytest

from src.configs.router import (
    MetabolicRouterConfig,
    StandardRouterConfig,
    SwitchRouterConfig,
    TopKRouterConfig,
    DeepSeekRouterConfig,
    ExpertChoiceRouterConfig,
    StressCorrectedRouterConfig,
)
from src.routers.factory import create_router, create_router_from_config
from src.routers.metabolic import MetabolicRouter
from src.routers.standard import StandardRouter, SwitchRouter, TopKRouter
from src.routers.deepseek import DeepSeekRouter
from src.routers.expert_choice import ExpertChoiceRouter
from src.routers.stress_corrected import StressCorrectedRouter


COMMON = dict(hidden_dim=32, num_experts=4, top_k=2)


class TestCreateRouter:
    def test_standard(self):
        router = create_router("standard", **COMMON)
        assert isinstance(router, StandardRouter)

    def test_topk(self):
        router = create_router("topk", **COMMON)
        assert isinstance(router, TopKRouter)

    def test_switch(self):
        router = create_router("switch", hidden_dim=32, num_experts=4, top_k=1)
        assert isinstance(router, SwitchRouter)

    def test_metabolic(self):
        router = create_router("metabolic", **COMMON)
        assert isinstance(router, MetabolicRouter)

    def test_deepseek(self):
        router = create_router("deepseek", **COMMON)
        assert isinstance(router, DeepSeekRouter)

    def test_expert_choice(self):
        router = create_router("expert_choice", **COMMON)
        assert isinstance(router, ExpertChoiceRouter)

    def test_stress_corrected(self):
        router = create_router("stress_corrected", **COMMON)
        assert isinstance(router, StressCorrectedRouter)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown router type"):
            create_router("dynmoe", **COMMON)


class TestCreateRouterFromConfig:
    def test_standard_config(self):
        config = StandardRouterConfig(**COMMON)
        router = create_router_from_config(config)
        assert isinstance(router, StandardRouter)

    def test_switch_config_creates_switch_router(self):
        # Regression: SwitchRouterConfig must dispatch to SwitchRouter, not StandardRouter
        config = SwitchRouterConfig(hidden_dim=32, num_experts=4, top_k=1)
        router = create_router_from_config(config)
        assert isinstance(router, SwitchRouter), (
            f"Expected SwitchRouter, got {type(router).__name__}"
        )

    def test_topk_config(self):
        config = TopKRouterConfig(**COMMON)
        router = create_router_from_config(config)
        assert isinstance(router, TopKRouter)

    def test_metabolic_config(self):
        config = MetabolicRouterConfig(**COMMON)
        router = create_router_from_config(config)
        assert isinstance(router, MetabolicRouter)

    def test_deepseek_config(self):
        config = DeepSeekRouterConfig(**COMMON)
        router = create_router_from_config(config)
        assert isinstance(router, DeepSeekRouter)

    def test_expert_choice_config(self):
        config = ExpertChoiceRouterConfig(**COMMON)
        router = create_router_from_config(config)
        assert isinstance(router, ExpertChoiceRouter)

    def test_stress_corrected_config(self):
        config = StressCorrectedRouterConfig(**COMMON)
        router = create_router_from_config(config)
        assert isinstance(router, StressCorrectedRouter)
