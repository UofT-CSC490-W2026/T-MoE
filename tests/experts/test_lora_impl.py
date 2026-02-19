import pytest
import torch
import torch.nn as nn

from src.experts.lora import LoRALayer, LoRAConfig, SharedLoRALayer
from src.experts.gpt_neo_lora import GPTNeoLoRAMLP
from src.experts.pool import ExpertPool
from src.layers.lora_moe import LoRAMoELayer
from src.routers.base import BaseRouter
from src.project_types import ExpertType


# ── helpers ──


class MockRouter(BaseRouter):
    """Always routes every token to experts 0 and 1 with equal weight."""

    def forward(self, x, return_metrics=False, record_usage=True):
        B, S, _ = x.shape
        weights = torch.ones(B, S, 2) * 0.5
        indices = torch.zeros(B, S, 2, dtype=torch.long)
        indices[:, :, 1] = 1
        return weights, indices, {}

    def compute_aux_loss(self):
        return torch.tensor(0.0)


class _RouterCfg:
    hidden_dim = 32
    num_experts = 2
    top_k = 2


@pytest.fixture
def lora_config():
    return LoRAConfig(hidden_dim=32, intermediate_dim=128, rank=4, alpha=8)


# ── LoRALayer ──


def test_lora_layer_init():
    layer = LoRALayer(32, 64, rank=4, alpha=16)

    assert layer.lora_A.weight.shape == (4, 32)
    assert layer.lora_B.weight.shape == (64, 4)
    assert torch.all(layer.lora_B.weight == 0), "B should be zero-init"
    assert not torch.all(layer.lora_A.weight == 0), (
        "A should be Kaiming-init (non-zero)"
    )
    assert layer.base_weight is None


def test_lora_layer_forward_zero_at_init():
    layer = LoRALayer(32, 32, rank=4, alpha=16)
    x = torch.randn(2, 10, 32)
    out = layer(x)

    # B=0 and no base weight → output should be all zeros
    assert torch.allclose(out, torch.zeros_like(out))


def test_lora_layer_forward_nonzero_after_perturb():
    layer = LoRALayer(32, 32, rank=4, alpha=16)
    nn.init.ones_(layer.lora_B.weight)

    out = layer(torch.randn(2, 10, 32))
    assert not torch.all(out == 0)
    assert out.shape == (2, 10, 32)


# ── SharedLoRALayer ──


def test_shared_lora_layer_memory_sharing():
    w = torch.randn(64, 32)
    layers = [SharedLoRALayer(w, None, rank=4, alpha=16) for _ in range(4)]
    # All layers should share the same underlying data pointer
    assert all(
        layer.shared_weight.data_ptr() == layers[0].shared_weight.data_ptr()
        for layer in layers
    )


def test_shared_lora_layer_forward():
    w = torch.randn(64, 32)
    layer = SharedLoRALayer(w, None, rank=4, alpha=16)
    x = torch.randn(2, 10, 32)
    out = layer(x)
    # B is zero-init so output should equal base output
    expected = nn.functional.linear(x, w)
    assert torch.allclose(out, expected, atol=1e-6)


# ── GPTNeoLoRAMLP ──


def test_gpt_neo_lora_mlp_structure():
    mlp = GPTNeoLoRAMLP(LoRAConfig(hidden_dim=32, rank=4, alpha=8))
    assert isinstance(mlp.c_fc, LoRALayer)
    assert isinstance(mlp.c_proj, LoRALayer)


def test_gpt_neo_lora_mlp_zero_at_init():
    mlp = GPTNeoLoRAMLP(LoRAConfig(hidden_dim=32, rank=4, alpha=8))
    out = mlp(torch.randn(1, 10, 32))
    assert torch.allclose(out, torch.zeros_like(out))


def test_gpt_neo_lora_load_from_mlp_raises_on_missing():
    """load_from_mlp should raise ValueError if MLP is missing c_fc/c_proj."""
    mlp = GPTNeoLoRAMLP(LoRAConfig(hidden_dim=32, rank=4, alpha=8))
    dummy = nn.Module()  # has no c_fc or c_proj
    with pytest.raises(ValueError, match="missing c_fc/c_proj"):
        mlp.load_from_mlp(dummy)


# ── ExpertPool ──


def test_expert_pool(lora_config):
    pool = ExpertPool(lora_config, num_experts=3, expert_type=ExpertType.GPTNEO_LORA)

    assert pool.num_experts == 3
    assert isinstance(pool[0], GPTNeoLoRAMLP)
    assert isinstance(pool[2], GPTNeoLoRAMLP)


# ── LoRAMoELayer ──


def test_lora_moe_layer_matches_base_at_init(lora_config):
    base_mlp = nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 32))
    router = MockRouter(_RouterCfg())

    layer = LoRAMoELayer(base_mlp, router, lora_config, num_experts=2)

    x = torch.randn(2, 5, 32)
    # Default return is plain tensor (HF compatible)
    out = layer(x)
    expected = base_mlp(x)

    assert isinstance(out, torch.Tensor), "Default should return plain tensor"
    assert torch.allclose(out, expected, atol=1e-6)


def test_lora_moe_layer_returns_tuple_with_metrics(lora_config):
    base_mlp = nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 32))
    router = MockRouter(_RouterCfg())
    layer = LoRAMoELayer(base_mlp, router, lora_config, num_experts=2)

    x = torch.randn(2, 5, 32)
    result = layer(x, return_metrics=True)
    assert isinstance(result, tuple), "return_metrics=True should return tuple"
    out, metrics = result
    assert isinstance(out, torch.Tensor)


def test_lora_moe_layer_changes_after_perturb(lora_config):
    base_mlp = nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 32))
    router = MockRouter(_RouterCfg())
    layer = LoRAMoELayer(base_mlp, router, lora_config, num_experts=2)

    x = torch.randn(2, 5, 32)
    expected = base_mlp(x)

    # Perturb expert 0 (both layers so the delta propagates)
    e0 = layer.expert_pool[0]
    nn.init.ones_(e0.c_fc.lora_B.weight)
    nn.init.ones_(e0.c_fc.lora_A.weight)
    nn.init.ones_(e0.c_proj.lora_B.weight)
    nn.init.ones_(e0.c_proj.lora_A.weight)

    out = layer(x)
    assert not torch.allclose(out, expected)
