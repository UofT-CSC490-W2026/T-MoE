import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock


def test_base_moe_layer_invalid_hidden_dim():
    from src.layers.base import BaseMoELayer

    class ConcreteMoE(BaseMoELayer):
        def forward(self, x, **kw):
            pass

    with pytest.raises(ValueError, match="hidden_dim"):
        ConcreteMoE(hidden_dim=0, num_experts=4, top_k=2)


def test_base_moe_layer_invalid_num_experts():
    from src.layers.base import BaseMoELayer

    class ConcreteMoE(BaseMoELayer):
        def forward(self, x, **kw):
            pass

    with pytest.raises(ValueError, match="num_experts"):
        ConcreteMoE(hidden_dim=64, num_experts=0, top_k=1)


def test_base_moe_layer_invalid_top_k():
    from src.layers.base import BaseMoELayer

    class ConcreteMoE(BaseMoELayer):
        def forward(self, x, **kw):
            pass

    with pytest.raises(ValueError, match="top_k"):
        ConcreteMoE(hidden_dim=64, num_experts=4, top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        ConcreteMoE(hidden_dim=64, num_experts=4, top_k=5)


def test_base_moe_layer_get_router_not_initialized():
    from src.layers.base import BaseMoELayer

    class ConcreteMoE(BaseMoELayer):
        def forward(self, x, **kw):
            pass

    layer = ConcreteMoE(hidden_dim=64, num_experts=4, top_k=2)

    with pytest.raises(RuntimeError, match="Router not initialized"):
        layer.get_router()


def test_base_moe_layer_get_experts_not_initialized():
    from src.layers.base import BaseMoELayer

    class ConcreteMoE(BaseMoELayer):
        def forward(self, x, **kw):
            pass

    layer = ConcreteMoE(hidden_dim=64, num_experts=4, top_k=2)

    with pytest.raises(RuntimeError, match="Experts not initialized"):
        layer.get_experts()


def test_base_moe_layer_get_cached_metrics_none():
    from src.layers.base import BaseMoELayer

    class ConcreteMoE(BaseMoELayer):
        def forward(self, x, **kw):
            pass

    layer = ConcreteMoE(hidden_dim=64, num_experts=4, top_k=2)

    assert layer.get_cached_metrics() is None


def test_base_moe_layer_get_cached_metrics_no_router():
    from src.layers.base import BaseMoELayer

    class ConcreteMoE(BaseMoELayer):
        def forward(self, x, **kw):
            pass

    layer = ConcreteMoE(hidden_dim=64, num_experts=4, top_k=2)

    layer._last_routing_weights = torch.randn(8, 4)

    layer._last_routing_indices = torch.randint(0, 4, (8, 2))

    assert layer.get_cached_metrics() is None


def test_base_moe_layer_get_cached_metrics_no_tracker():
    from src.layers.base import BaseMoELayer

    class ConcreteMoE(BaseMoELayer):
        def forward(self, x, **kw):
            pass

    layer = ConcreteMoE(hidden_dim=64, num_experts=4, top_k=2)

    layer._last_routing_weights = torch.randn(8, 4)

    layer._last_routing_indices = torch.randint(0, 4, (8, 2))

    layer.router = MagicMock(spec=[])

    assert layer.get_cached_metrics() is None


def test_base_moe_layer_get_cached_metrics_with_tracker():
    from src.layers.base import BaseMoELayer

    class ConcreteMoE(BaseMoELayer):
        def forward(self, x, **kw):
            pass

    layer = ConcreteMoE(hidden_dim=64, num_experts=4, top_k=2)

    layer._last_routing_weights = torch.randn(8, 4)

    layer._last_routing_indices = torch.randint(0, 4, (8, 2))

    mock_router = MagicMock()

    mock_router.metrics_tracker.compute_all_metrics.return_value = {"entropy": 1.5}

    layer.router = mock_router

    metrics = layer.get_cached_metrics()

    assert metrics is not None

    assert "weights" in metrics


def test_base_moe_layer_clear_cached_metrics():
    from src.layers.base import BaseMoELayer

    class ConcreteMoE(BaseMoELayer):
        def forward(self, x, **kw):
            pass

    layer = ConcreteMoE(hidden_dim=64, num_experts=4, top_k=2)

    layer._last_routing_weights = torch.randn(8, 4)

    layer._last_routing_indices = torch.randint(0, 4, (8, 2))

    layer.clear_cached_metrics()

    assert layer._last_routing_weights is None

    assert layer._last_routing_indices is None


def test_base_moe_layer_extra_repr():
    from src.layers.base import BaseMoELayer

    class ConcreteMoE(BaseMoELayer):
        def forward(self, x, **kw):
            pass

    layer = ConcreteMoE(hidden_dim=64, num_experts=4, top_k=2)

    repr_str = layer.extra_repr()

    assert "hidden_dim=64" in repr_str


def _make_gptneo_moe(hidden=64, num_experts=4, top_k=2):
    from src.experts.lora import LoRAConfig
    from src.layers.lora_moe import LoRAMoELayer
    from src.routers.metabolic import MetabolicRouter
    from src.configs.router import MetabolicRouterConfig

    mlp = nn.Module()

    mlp.c_fc = nn.Linear(hidden, hidden * 4)

    mlp.c_proj = nn.Linear(hidden * 4, hidden)

    cfg = LoRAConfig(hidden_dim=hidden, rank=4, alpha=8)

    router_cfg = MetabolicRouterConfig(
        hidden_dim=hidden, num_experts=num_experts, top_k=top_k
    )

    router = MetabolicRouter(router_cfg)

    return LoRAMoELayer.from_pretrained_mlp(
        mlp=mlp, router=router, lora_config=cfg, num_experts=num_experts
    )


def test_lora_moe_layer_forward_basic():
    moe = _make_gptneo_moe()

    x = torch.randn(2, 4, 64)

    out = moe(x)

    assert out.shape == (2, 4, 64)


def test_lora_moe_layer_forward_return_metrics():
    moe = _make_gptneo_moe()

    x = torch.randn(2, 4, 64)

    out, metrics = moe(x, return_metrics=True)

    assert out.shape == (2, 4, 64)

    assert metrics is not None


def test_lora_moe_layer_forward_record_usage_false():
    moe = _make_gptneo_moe()

    moe.train()

    x = torch.randn(2, 4, 64)

    out = moe(x, record_usage=False)

    assert out.shape == (2, 4, 64)


def test_lora_moe_layer_step():
    moe = _make_gptneo_moe()

    moe.train()

    x = torch.randn(2, 4, 64)

    moe(x)

    moe.step()


def test_lora_moe_layer_get_cached_metrics():
    moe = _make_gptneo_moe()

    x = torch.randn(2, 4, 64)

    moe(x)

    metrics = moe.get_cached_metrics()

    assert metrics is not None


def test_lora_moe_layer_get_cached_metrics_none():
    moe = _make_gptneo_moe()

    moe._last_routing_weights = None

    metrics = moe.get_cached_metrics()

    assert metrics is None


def test_lora_moe_layer_forced_record_usage():
    moe = _make_gptneo_moe()

    moe._forced_record_usage = False

    x = torch.randn(2, 4, 64)

    out = moe(x)

    assert out.shape == (2, 4, 64)


def test_lora_moe_layer_with_shared_base_lora():
    from src.experts.lora import LoRAConfig
    from src.layers.lora_moe import LoRAMoELayer
    from src.routers.metabolic import MetabolicRouter
    from src.configs.router import MetabolicRouterConfig

    hidden = 64

    mlp = nn.Module()

    mlp.c_fc = nn.Linear(hidden, hidden * 4)

    mlp.c_proj = nn.Linear(hidden * 4, hidden)

    cfg = LoRAConfig(
        hidden_dim=hidden, rank=4, alpha=8, shared_base_rank=2, shared_base_alpha=2.0
    )

    router_cfg = MetabolicRouterConfig(hidden_dim=hidden, num_experts=4, top_k=2)

    router = MetabolicRouter(router_cfg)

    moe = LoRAMoELayer.from_pretrained_mlp(
        mlp=mlp, router=router, lora_config=cfg, num_experts=4
    )

    x = torch.randn(2, 4, hidden)

    out = moe(x)

    assert out.shape == (2, 4, hidden)


def test_lora_moe_layer_router_no_record_usage_param():
    from src.experts.lora import LoRAConfig
    from src.layers.lora_moe import LoRAMoELayer
    from src.routers.standard import StandardRouter
    from src.configs.router import StandardRouterConfig

    hidden = 64

    mlp = nn.Module()

    mlp.c_fc = nn.Linear(hidden, hidden * 4)

    mlp.c_proj = nn.Linear(hidden * 4, hidden)

    cfg = LoRAConfig(hidden_dim=hidden, rank=4, alpha=8)

    router_cfg = StandardRouterConfig(hidden_dim=hidden, num_experts=4, top_k=2)

    router = StandardRouter(router_cfg)

    moe = LoRAMoELayer.from_pretrained_mlp(
        mlp=mlp, router=router, lora_config=cfg, num_experts=4
    )

    x = torch.randn(2, 4, hidden)

    out = moe(x)

    assert out.shape == (2, 4, hidden)


def test_lora_moe_layer_with_qwen2_shared_base():
    from src.experts.lora import LoRAConfig
    from src.layers.lora_moe import LoRAMoELayer
    from src.routers.metabolic import MetabolicRouter
    from src.configs.router import MetabolicRouterConfig
    from src.project_types import ExpertType

    hidden = 64

    intermediate = 256

    mlp = nn.Module()

    mlp.gate_proj = nn.Linear(hidden, intermediate, bias=False)

    mlp.up_proj = nn.Linear(hidden, intermediate, bias=False)

    mlp.down_proj = nn.Linear(intermediate, hidden, bias=False)

    cfg = LoRAConfig(
        hidden_dim=hidden,
        intermediate_dim=intermediate,
        rank=4,
        alpha=8,
        shared_base_rank=2,
        shared_base_alpha=2.0,
    )

    router_cfg = MetabolicRouterConfig(hidden_dim=hidden, num_experts=4, top_k=2)

    router = MetabolicRouter(router_cfg)

    moe = LoRAMoELayer.from_pretrained_mlp(
        mlp=mlp,
        router=router,
        lora_config=cfg,
        num_experts=4,
        expert_type=ExpertType.QWEN2_LORA,
    )

    x = torch.randn(2, 4, hidden)

    out = moe(x)

    assert out.shape == (2, 4, hidden)


def test_lora_moe_layer_get_cached_metrics_with_lora_norms():
    moe = _make_gptneo_moe()

    x = torch.randn(2, 4, 64)

    moe(x)

    metrics = moe.get_cached_metrics()

    assert metrics is not None

    assert "lora_delta_norm_per_expert" in metrics


def test_lora_moe_layer_init_shared_base_lora_out_proj_none():
    from src.experts.lora import LoRAConfig
    from src.layers.lora_moe import LoRAMoELayer
    from src.routers.metabolic import MetabolicRouter
    from src.configs.router import MetabolicRouterConfig
    from src.experts.lora import LoRAMLPExpert

    class NoOutProjExpert(LoRAMLPExpert):
        def __init__(self, config):
            super().__init__(config)

        def load_from_mlp(self, mlp):
            pass

        def forward(self, x):
            return x

    mlp = nn.Module()

    mlp.c_fc = nn.Linear(64, 256)

    mlp.c_proj = nn.Linear(256, 64)

    cfg = LoRAConfig(hidden_dim=64, rank=4, alpha=8, shared_base_rank=4)

    router_cfg = MetabolicRouterConfig(hidden_dim=64, num_experts=2, top_k=1)

    router = MetabolicRouter(router_cfg)

    from src.core.registry import ExpertRegistry

    ExpertRegistry._registries["experts"]["no_out_proj"] = NoOutProjExpert

    layer = LoRAMoELayer(mlp, router, cfg, num_experts=2)

    layer._init_shared_base_lora()

    assert layer.shared_proj_lora is None


def test_lora_moe_layer_get_cached_metrics_no_weights():
    moe = _make_gptneo_moe()

    moe._last_routing_weights = None

    assert moe.get_cached_metrics() is None


def test_lora_moe_layer_get_cached_metrics_indices_none():
    moe = _make_gptneo_moe()

    x = torch.randn(2, 4, 64)

    moe(x)

    moe._last_routing_indices = None

    metrics = moe.get_cached_metrics()

    assert metrics is not None

    assert "indices" not in metrics


def test_lora_moe_layer_forward_metrics_none_indices():
    moe = _make_gptneo_moe()

    x = torch.randn(2, 4, 64)

    out, metrics = moe(x, return_metrics=True)

    assert "weights" in metrics

    assert out.shape == (2, 4, 64)


def test_lora_moe_layer_forward_all_experts_zero_weight():
    from src.experts.lora import LoRAConfig
    from src.layers.lora_moe import LoRAMoELayer
    from src.routers.base import BaseRouter

    class ZeroWeightRouter(BaseRouter):
        def forward(self, x, return_metrics=False, **kw):
            B, S, _ = x.shape
            N = B * S
            weights = torch.zeros(N, 4)
            weights[:, 0] = 1.0
            return weights, None, {} if return_metrics else None

        def compute_aux_loss(self):
            return torch.tensor(0.0)

    class _Cfg:
        hidden_dim = 64
        num_experts = 4
        top_k = 1

    mlp = nn.Module()

    mlp.c_fc = nn.Linear(64, 256)

    mlp.c_proj = nn.Linear(256, 64)

    cfg = LoRAConfig(hidden_dim=64, rank=4, alpha=8)

    router = ZeroWeightRouter(_Cfg())

    moe = LoRAMoELayer.from_pretrained_mlp(
        mlp=mlp, router=router, lora_config=cfg, num_experts=4
    )

    x = torch.randn(2, 4, 64)

    out = moe(x)

    assert out.shape == (2, 4, 64)


def test_lora_moe_layer_shared_proj_lora_gptneo():
    from src.experts.lora import LoRAConfig
    from src.layers.lora_moe import LoRAMoELayer
    from src.routers.metabolic import MetabolicRouter
    from src.configs.router import MetabolicRouterConfig

    mlp = nn.Module()

    mlp.c_fc = nn.Linear(64, 256)

    mlp.c_proj = nn.Linear(256, 64)

    cfg = LoRAConfig(
        hidden_dim=64, rank=4, alpha=8, shared_base_rank=4, shared_base_alpha=4.0
    )

    router_cfg = MetabolicRouterConfig(hidden_dim=64, num_experts=2, top_k=1)

    router = MetabolicRouter(router_cfg)

    moe = LoRAMoELayer.from_pretrained_mlp(
        mlp=mlp, router=router, lora_config=cfg, num_experts=2
    )

    assert moe.shared_proj_lora is not None

    x = torch.randn(2, 4, 64)

    out = moe(x)

    assert out.shape == (2, 4, 64)


def test_lora_moe_layer_forward_return_metrics_with_indices():
    from src.experts.lora import LoRAConfig
    from src.layers.lora_moe import LoRAMoELayer
    from src.routers.base import BaseRouter

    class IndexRouter(BaseRouter):
        def forward(self, x, return_metrics=False, **kw):
            B, S, _ = x.shape
            N = B * S
            weights = torch.zeros(N, 2)
            weights[:, 0] = 1.0
            indices = torch.zeros(N, 1, dtype=torch.long)
            return weights, indices, {} if return_metrics else None

        def compute_aux_loss(self):
            return torch.tensor(0.0)

    class _Cfg:
        hidden_dim = 64
        num_experts = 2
        top_k = 1

    mlp = nn.Module()

    mlp.c_fc = nn.Linear(64, 256)

    mlp.c_proj = nn.Linear(256, 64)

    cfg = LoRAConfig(hidden_dim=64, rank=4, alpha=8)

    router = IndexRouter(_Cfg())

    moe = LoRAMoELayer.from_pretrained_mlp(
        mlp=mlp, router=router, lora_config=cfg, num_experts=2
    )

    x = torch.randn(2, 4, 64)

    out, metrics = moe(x, return_metrics=True)

    assert "indices" in metrics

    assert out.shape == (2, 4, 64)


def test_lora_moe_layer_get_cached_metrics_with_indices():
    moe = _make_gptneo_moe()

    x = torch.randn(2, 4, 64)

    moe(x)

    moe._last_routing_indices = torch.zeros(8, 1, dtype=torch.long)

    metrics = moe.get_cached_metrics()

    assert metrics is not None

    assert "indices" in metrics
