import pytest

import torch

import torch.nn as nn

from src.experts.lora import LoRALayer, LoRAConfig, SharedLoRALayer

from src.experts.gpt_neo_lora import GPTNeoLoRAMLP

from src.experts.pool import ExpertPool

from src.layers.lora_moe import LoRAMoELayer

from src.routers.base import BaseRouter

from src.project_types import ExpertType

               

class MockRouter(BaseRouter):

    

    def forward(self, x, return_metrics=False, record_usage=True):

        B, S, _ = x.shape

        N = B * S

                                                                  

        weights = torch.zeros(N, 2)

        weights[:, 0] = 0.5

        weights[:, 1] = 0.5

        return weights, None, {}

    def compute_aux_loss(self):

        return torch.tensor(0.0)

class _RouterCfg:

    hidden_dim = 32

    num_experts = 2

    top_k = 2

class MockMLP(nn.Module):

    def __init__(self):

        super().__init__()

        self.c_fc = nn.Linear(32, 128)

        self.act = nn.GELU(approximate="tanh")

        self.c_proj = nn.Linear(128, 32)

    def forward(self, x):

        return self.c_proj(self.act(self.c_fc(x)))

@pytest.fixture

def lora_config():

    return LoRAConfig(hidden_dim=32, intermediate_dim=128, rank=4, alpha=8)

                 

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

                                                         

    assert torch.allclose(out, torch.zeros_like(out))

def test_lora_layer_forward_nonzero_after_perturb():

    layer = LoRALayer(32, 32, rank=4, alpha=16)

    nn.init.ones_(layer.lora_B.weight)

    out = layer(torch.randn(2, 10, 32))

    assert not torch.all(out == 0)

    assert out.shape == (2, 10, 32)

                       

def test_shared_lora_layer_memory_sharing():

    w = torch.randn(64, 32)

    layers = [SharedLoRALayer(w, None, rank=4, alpha=16) for _ in range(4)]

                                                              

    assert all(

        layer.shared_weight.data_ptr() == layers[0].shared_weight.data_ptr()

        for layer in layers

    )

def test_shared_lora_layer_not_in_state_dict():

    
    w = torch.randn(64, 32)

    layer = SharedLoRALayer(w, None, rank=4, alpha=16)

    sd = layer.state_dict()

    assert "shared_weight" not in sd

    assert "lora_A.weight" in sd

    assert "lora_B.weight" in sd

def test_shared_lora_layer_forward():

    w = torch.randn(64, 32)

    layer = SharedLoRALayer(w, None, rank=4, alpha=16)

    x = torch.randn(2, 10, 32)

    out = layer(x)

                                                       

    expected = nn.functional.linear(x, w)

    assert torch.allclose(out, expected, atol=1e-6)

                     

def test_gpt_neo_lora_mlp_structure():

    mlp = GPTNeoLoRAMLP(LoRAConfig(hidden_dim=32, rank=4, alpha=8))

    mock_base = MockMLP()

    mlp.load_from_mlp(mock_base)

    assert isinstance(mlp.c_fc, SharedLoRALayer)

    assert isinstance(mlp.c_proj, SharedLoRALayer)

def test_gpt_neo_lora_mlp_zero_at_init():

    mlp = GPTNeoLoRAMLP(LoRAConfig(hidden_dim=32, rank=4, alpha=8))

    mock_base = MockMLP()

    mlp.load_from_mlp(mock_base)

    x = torch.randn(1, 10, 32)

    out = mlp(x)

    expected = mock_base(x)

    assert torch.allclose(out, expected, atol=1e-6)

def test_gpt_neo_lora_load_from_mlp_raises_on_missing():

    
    mlp = GPTNeoLoRAMLP(LoRAConfig(hidden_dim=32, rank=4, alpha=8))

    dummy = nn.Module()                         

    with pytest.raises(ValueError, match="missing c_fc/c_proj"):

        mlp.load_from_mlp(dummy)

                  

def test_expert_pool(lora_config):

    pool = ExpertPool(lora_config, num_experts=3, expert_type=ExpertType.GPTNEO_LORA)

    assert pool.num_experts == 3

    assert isinstance(pool[0], GPTNeoLoRAMLP)

    assert isinstance(pool[2], GPTNeoLoRAMLP)

                    

def test_lora_moe_layer_matches_base_at_init(lora_config):

    base_mlp = MockMLP()

    router = MockRouter(_RouterCfg())

    layer = LoRAMoELayer.from_pretrained_mlp(

        base_mlp, router, lora_config, num_experts=2

    )

    x = torch.randn(2, 5, 32)

                                                    

    out = layer(x)

    expected = base_mlp(x)

    assert isinstance(out, torch.Tensor), "Default should return plain tensor"

    assert torch.allclose(out, expected, atol=1e-6)

def test_lora_moe_layer_returns_tuple_with_metrics(lora_config):

    base_mlp = MockMLP()

    router = MockRouter(_RouterCfg())

    layer = LoRAMoELayer.from_pretrained_mlp(

        base_mlp, router, lora_config, num_experts=2

    )

    x = torch.randn(2, 5, 32)

    result = layer(x, return_metrics=True)

    assert isinstance(result, tuple), "return_metrics=True should return tuple"

    out, metrics = result

    assert isinstance(out, torch.Tensor)

def test_lora_moe_layer_changes_after_perturb(lora_config):

    base_mlp = MockMLP()

    router = MockRouter(_RouterCfg())

    layer = LoRAMoELayer.from_pretrained_mlp(

        base_mlp, router, lora_config, num_experts=2

    )

    x = torch.randn(2, 5, 32)

    expected = base_mlp(x)

    e0 = layer.expert_pool[0]

    nn.init.ones_(e0.c_fc.lora_B.weight)

    nn.init.ones_(e0.c_fc.lora_A.weight)

    nn.init.ones_(e0.c_proj.lora_B.weight)

    nn.init.ones_(e0.c_proj.lora_A.weight)

    out = layer(x)

    assert not torch.allclose(out, expected)

def test_consolidate_shared_weights_aliases_buffers(lora_config):

    
    base_mlp = MockMLP()

    pool = ExpertPool(lora_config, num_experts=4)

    pool.load_from_mlp(base_mlp)

    pool.consolidate_shared_weights()

    e0 = pool.experts[0]

    for expert in pool.experts[1:]:

        assert (

            expert.c_fc._buffers["shared_weight"].data_ptr()

            == e0.c_fc._buffers["shared_weight"].data_ptr()

        ), "Experts should share the same weight buffer after consolidation"

def test_gptneo_lora_forward_raises_before_load():

    
    from src.experts.gpt_neo_lora import GPTNeoLoRAMLP

    from src.experts.lora import LoRAConfig

    expert = GPTNeoLoRAMLP(LoRAConfig(hidden_dim=32, rank=4, alpha=8))

    with pytest.raises(RuntimeError, match="load_from_mlp"):

        expert(torch.randn(2, 4, 32))

def test_b_init_scale_breaks_expert_symmetry():

    
    config = LoRAConfig(hidden_dim=32, rank=4, alpha=8, b_init_scale=0.01)

    base_mlp = MockMLP()

    experts = []

    for _ in range(4):

        e = GPTNeoLoRAMLP(config)

        e.load_from_mlp(base_mlp)

        experts.append(e)

    x = torch.randn(1, 5, 32)

    outputs = [e(x) for e in experts]

                                                                            

    any_different = False

    for i in range(len(outputs)):

        for j in range(i + 1, len(outputs)):

            if not torch.allclose(outputs[i], outputs[j], atol=1e-7):

                any_different = True

                break

    assert any_different, (

        "With b_init_scale > 0, at least two experts should produce different outputs at init"

    )

def test_b_init_scale_zero_preserves_base_output():

    
    config = LoRAConfig(hidden_dim=32, rank=4, alpha=8, b_init_scale=0.0)

    base_mlp = MockMLP()

    expert = GPTNeoLoRAMLP(config)

    expert.load_from_mlp(base_mlp)

    x = torch.randn(1, 5, 32)

    assert torch.allclose(expert(x), base_mlp(x), atol=1e-6)
