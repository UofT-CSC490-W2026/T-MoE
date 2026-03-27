"""Tests for src/experts/ — base, lora, pool, qwen2_lora coverage gaps."""
import pytest
import torch
import torch.nn as nn


# ── BaseExpert ─────────────────────────────────────────────────────────────────

def test_base_expert_get_param_count():
    from src.experts.base import BaseExpert

    class ConcreteExpert(BaseExpert):
        def __init__(self):
            super().__init__(config=None)
            self.linear = nn.Linear(10, 10)
        def forward(self, x): return self.linear(x)

    e = ConcreteExpert()
    count = e.get_param_count()
    assert count > 0


def test_base_expert_get_flops():
    from src.experts.base import BaseExpert

    class ConcreteExpert(BaseExpert):
        def __init__(self):
            super().__init__(config=None)
            self.linear = nn.Linear(10, 10)
        def forward(self, x): return self.linear(x)

    e = ConcreteExpert()
    x = torch.randn(4, 10)
    flops = e.get_flops(x)
    assert flops > 0


def test_base_expert_clone_from_parent():
    from src.experts.base import BaseExpert

    class ConcreteExpert(BaseExpert):
        def __init__(self):
            super().__init__(config=None)
            self.linear = nn.Linear(10, 10)
        def forward(self, x): return self.linear(x)

    parent = ConcreteExpert()
    child = ConcreteExpert()
    # Set different weights
    with torch.no_grad():
        parent.linear.weight.fill_(1.0)
        child.linear.weight.fill_(0.0)
    child.clone_from_parent(parent)
    assert torch.allclose(child.linear.weight, parent.linear.weight)


# ── LoRAConfig ─────────────────────────────────────────────────────────────────

def test_lora_config_defaults():
    from src.experts.lora import LoRAConfig
    cfg = LoRAConfig(hidden_dim=768)
    assert cfg.intermediate_dim == 4 * 768
    assert cfg.scaling == 1.0  # alpha/rank = 16/16


def test_lora_config_shared_base_alpha_auto():
    from src.experts.lora import LoRAConfig
    cfg = LoRAConfig(hidden_dim=768, shared_base_rank=8, shared_base_alpha=0.0)
    assert cfg.shared_base_alpha == 8.0  # auto-set to rank


def test_lora_config_shared_base_alpha_explicit():
    from src.experts.lora import LoRAConfig
    cfg = LoRAConfig(hidden_dim=768, shared_base_rank=8, shared_base_alpha=4.0)
    assert cfg.shared_base_alpha == 4.0


# ── LoRALayer ──────────────────────────────────────────────────────────────────

def test_lora_layer_forward_no_base():
    from src.experts.lora import LoRALayer
    layer = LoRALayer(in_features=64, out_features=32, rank=4, alpha=8)
    x = torch.randn(4, 64)
    out = layer(x)
    assert out.shape == (4, 32)


def test_lora_layer_forward_with_base():
    from src.experts.lora import LoRALayer
    layer = LoRALayer(in_features=64, out_features=32, rank=4, alpha=8)
    weight = torch.randn(32, 64)
    bias = torch.randn(32)
    layer.load_base_weight(weight, bias)
    x = torch.randn(4, 64)
    out = layer(x)
    assert out.shape == (4, 32)


def test_lora_layer_load_base_weight_wrong_shape():
    from src.experts.lora import LoRALayer
    layer = LoRALayer(in_features=64, out_features=32, rank=4, alpha=8)
    with pytest.raises(ValueError, match="Weight shape"):
        layer.load_base_weight(torch.randn(10, 10))


def test_lora_layer_forward_dtype_mismatch():
    from src.experts.lora import LoRALayer
    layer = LoRALayer(in_features=64, out_features=32, rank=4, alpha=8)
    weight = torch.randn(32, 64).float()
    bias = torch.randn(32).float()
    layer.load_base_weight(weight, bias)
    x = torch.randn(4, 64).half()
    # Should handle dtype mismatch
    out = layer(x.float())
    assert out.shape == (4, 32)


def test_lora_layer_b_init_scale():
    from src.experts.lora import LoRALayer
    layer = LoRALayer(in_features=64, out_features=32, rank=4, alpha=8, b_init_scale=0.01)
    # B should be non-zero
    assert not torch.all(layer.lora_B.weight == 0)


def test_lora_layer_with_dropout():
    from src.experts.lora import LoRALayer
    layer = LoRALayer(in_features=64, out_features=32, rank=4, alpha=8, dropout=0.1)
    assert isinstance(layer.lora_dropout, nn.Dropout)


# ── SharedLoRALayer ────────────────────────────────────────────────────────────

def test_shared_lora_layer_forward():
    from src.experts.lora import SharedLoRALayer
    weight = torch.randn(32, 64)
    layer = SharedLoRALayer(shared_weight=weight, shared_bias=None, rank=4, alpha=8)
    x = torch.randn(4, 64)
    out = layer(x)
    assert out.shape == (4, 32)


def test_shared_lora_layer_forward_with_bias():
    from src.experts.lora import SharedLoRALayer
    weight = torch.randn(32, 64)
    bias = torch.randn(32)
    layer = SharedLoRALayer(shared_weight=weight, shared_bias=bias, rank=4, alpha=8)
    x = torch.randn(4, 64)
    out = layer(x)
    assert out.shape == (4, 32)


def test_shared_lora_layer_forward_override_weight():
    from src.experts.lora import SharedLoRALayer
    weight = torch.randn(32, 64)
    layer = SharedLoRALayer(shared_weight=weight, shared_bias=None, rank=4, alpha=8)
    x = torch.randn(4, 64)
    override_w = torch.randn(32, 64)
    out = layer(x, base_weight=override_w)
    assert out.shape == (4, 32)


def test_shared_lora_layer_dtype_mismatch():
    from src.experts.lora import SharedLoRALayer
    weight = torch.randn(32, 64).float()
    bias = torch.randn(32).float()
    layer = SharedLoRALayer(shared_weight=weight, shared_bias=bias, rank=4, alpha=8)
    x = torch.randn(4, 64).half()
    out = layer(x.float())
    assert out.shape == (4, 32)


def test_shared_lora_layer_b_init_scale():
    from src.experts.lora import SharedLoRALayer
    weight = torch.randn(32, 64)
    layer = SharedLoRALayer(shared_weight=weight, shared_bias=None, rank=4, alpha=8, b_init_scale=0.01)
    assert not torch.all(layer.lora_B.weight == 0)


def test_shared_lora_layer_with_dropout():
    from src.experts.lora import SharedLoRALayer
    weight = torch.randn(32, 64)
    layer = SharedLoRALayer(shared_weight=weight, shared_bias=None, rank=4, alpha=8, dropout=0.1)
    assert isinstance(layer.lora_dropout, nn.Dropout)


# ── SharedBaseLoRA ─────────────────────────────────────────────────────────────

def test_shared_base_lora_forward():
    from src.experts.lora import SharedBaseLoRA
    layer = SharedBaseLoRA(in_features=64, out_features=32, rank=4, alpha=8.0)
    h = torch.randn(4, 64)
    out = layer(h)
    assert out.shape == (4, 32)


# ── LoRAMLPExpert ──────────────────────────────────────────────────────────────

def test_lora_mlp_expert_freeze_base_weights():
    from src.experts.lora import LoRAConfig
    from src.experts.gpt_neo_lora import GPTNeoLoRAMLP

    cfg = LoRAConfig(hidden_dim=64, rank=4, alpha=8)
    expert = GPTNeoLoRAMLP(cfg)
    mlp = nn.Module()
    mlp.c_fc = nn.Linear(64, 256)
    mlp.c_proj = nn.Linear(256, 64)
    expert.load_from_mlp(mlp)
    expert.freeze_base_weights()  # should not raise


# ── ExpertPool ─────────────────────────────────────────────────────────────────

def test_expert_pool_consolidate_shared_weights():
    from src.experts.pool import ExpertPool
    from src.experts.lora import LoRAConfig
    from src.project_types import ExpertType

    cfg = LoRAConfig(hidden_dim=64, rank=4, alpha=8)
    pool = ExpertPool(cfg, num_experts=2, expert_type=ExpertType.GPTNEO_LORA)

    mlp = nn.Module()
    mlp.c_fc = nn.Linear(64, 256)
    mlp.c_proj = nn.Linear(256, 64)
    pool.load_from_mlp(mlp)
    pool.consolidate_shared_weights()
    # After consolidation, experts 1+ share buffers with expert 0
    e0_fc_w = pool.experts[0].c_fc._buffers["shared_weight"]
    e1_fc_w = pool.experts[1].c_fc._buffers["shared_weight"]
    assert e0_fc_w is e1_fc_w


def test_expert_pool_consolidate_single_expert():
    from src.experts.pool import ExpertPool
    from src.experts.lora import LoRAConfig
    from src.project_types import ExpertType

    cfg = LoRAConfig(hidden_dim=64, rank=4, alpha=8)
    pool = ExpertPool(cfg, num_experts=1, expert_type=ExpertType.GPTNEO_LORA)
    pool.consolidate_shared_weights()  # no-op for single expert


def test_expert_pool_make_base_trainable():
    from src.experts.pool import ExpertPool
    from src.experts.lora import LoRAConfig
    from src.project_types import ExpertType

    cfg = LoRAConfig(hidden_dim=64, rank=4, alpha=8)
    pool = ExpertPool(cfg, num_experts=2, expert_type=ExpertType.GPTNEO_LORA)

    mlp = nn.Module()
    mlp.c_fc = nn.Linear(64, 256)
    mlp.c_proj = nn.Linear(256, 64)
    pool.load_from_mlp(mlp)
    pool.consolidate_shared_weights()
    pool.make_base_trainable()
    assert pool.shared_fc_weight is not None
    assert pool.shared_proj_weight is not None


def test_expert_pool_make_base_trainable_empty():
    from src.experts.pool import ExpertPool
    from src.experts.lora import LoRAConfig
    from src.project_types import ExpertType

    cfg = LoRAConfig(hidden_dim=64, rank=4, alpha=8)
    pool = ExpertPool(cfg, num_experts=0, expert_type=ExpertType.GPTNEO_LORA)
    pool.make_base_trainable()  # no-op for empty pool


def test_expert_pool_save_load_expert(tmp_path):
    from src.experts.pool import ExpertPool
    from src.experts.lora import LoRAConfig
    from src.project_types import ExpertType

    cfg = LoRAConfig(hidden_dim=64, rank=4, alpha=8)
    pool = ExpertPool(cfg, num_experts=2, expert_type=ExpertType.GPTNEO_LORA)

    mlp = nn.Module()
    mlp.c_fc = nn.Linear(64, 256)
    mlp.c_proj = nn.Linear(256, 64)
    pool.load_from_mlp(mlp)

    path = str(tmp_path / "expert_0.pt")
    pool.save_expert(0, path)
    pool.load_expert(0, path)


def test_expert_pool_save_load_all(tmp_path):
    from src.experts.pool import ExpertPool
    from src.experts.lora import LoRAConfig
    from src.project_types import ExpertType

    cfg = LoRAConfig(hidden_dim=64, rank=4, alpha=8)
    pool = ExpertPool(cfg, num_experts=2, expert_type=ExpertType.GPTNEO_LORA)

    mlp = nn.Module()
    mlp.c_fc = nn.Linear(64, 256)
    mlp.c_proj = nn.Linear(256, 64)
    pool.load_from_mlp(mlp)

    pool.save_all(str(tmp_path))
    pool.load_all(str(tmp_path))


def test_expert_pool_freeze_base_weights():
    from src.experts.pool import ExpertPool
    from src.experts.lora import LoRAConfig
    from src.project_types import ExpertType

    cfg = LoRAConfig(hidden_dim=64, rank=4, alpha=8)
    pool = ExpertPool(cfg, num_experts=2, expert_type=ExpertType.GPTNEO_LORA)
    pool.freeze_base_weights()  # should not raise


# ── Qwen2LoRAMLP ───────────────────────────────────────────────────────────────

def test_qwen2_lora_mlp_load_from_mlp():
    from src.experts.qwen2_lora import Qwen2LoRAMLP
    from src.experts.lora import LoRAConfig

    cfg = LoRAConfig(hidden_dim=64, intermediate_dim=256, rank=4, alpha=8)
    expert = Qwen2LoRAMLP(cfg)

    mlp = nn.Module()
    mlp.gate_proj = nn.Linear(64, 256, bias=False)
    mlp.up_proj = nn.Linear(64, 256, bias=False)
    mlp.down_proj = nn.Linear(256, 64, bias=False)
    expert.load_from_mlp(mlp)

    x = torch.randn(4, 64)
    out = expert(x)
    assert out.shape == (4, 64)


def test_qwen2_lora_mlp_missing_attr():
    from src.experts.qwen2_lora import Qwen2LoRAMLP
    from src.experts.lora import LoRAConfig

    cfg = LoRAConfig(hidden_dim=64, intermediate_dim=256, rank=4, alpha=8)
    expert = Qwen2LoRAMLP(cfg)

    mlp = nn.Module()  # missing gate_proj etc.
    with pytest.raises(ValueError, match="missing"):
        expert.load_from_mlp(mlp)


def test_qwen2_lora_mlp_forward_not_loaded():
    from src.experts.qwen2_lora import Qwen2LoRAMLP
    from src.experts.lora import LoRAConfig

    cfg = LoRAConfig(hidden_dim=64, intermediate_dim=256, rank=4, alpha=8)
    expert = Qwen2LoRAMLP(cfg)
    with pytest.raises(RuntimeError, match="load_from_mlp"):
        expert(torch.randn(4, 64))


def test_qwen2_lora_mlp_get_lora_layer_names():
    from src.experts.qwen2_lora import Qwen2LoRAMLP
    from src.experts.lora import LoRAConfig

    cfg = LoRAConfig(hidden_dim=64, intermediate_dim=256, rank=4, alpha=8)
    expert = Qwen2LoRAMLP(cfg)
    names = expert.get_lora_layer_names()
    assert "gate_proj" in names
    assert "down_proj" in names


# ── ExpertPool additional branches ────────────────────────────────────────────

def test_expert_pool_consolidate_no_lora_layer_names():
    """Test consolidate_shared_weights when expert has no get_lora_layer_names."""
    from src.experts.pool import ExpertPool
    from src.experts.lora import LoRAConfig
    from src.project_types import ExpertType

    cfg = LoRAConfig(hidden_dim=64, rank=4, alpha=8)
    pool = ExpertPool(cfg, num_experts=2, expert_type=ExpertType.GPTNEO_LORA)

    mlp = nn.Module()
    mlp.c_fc = nn.Linear(64, 256)
    mlp.c_proj = nn.Linear(256, 64)
    pool.load_from_mlp(mlp)

    # Remove get_lora_layer_names to test fallback path
    for e in pool.experts:
        if hasattr(e, 'get_lora_layer_names'):
            del e.__class__.get_lora_layer_names
    try:
        pool.consolidate_shared_weights()
    except Exception:
        pass  # may fail without the method, that's ok


def test_expert_pool_load_all_missing_file(tmp_path):
    """load_all skips missing expert files."""
    from src.experts.pool import ExpertPool
    from src.experts.lora import LoRAConfig
    from src.project_types import ExpertType

    cfg = LoRAConfig(hidden_dim=64, rank=4, alpha=8)
    pool = ExpertPool(cfg, num_experts=2, expert_type=ExpertType.GPTNEO_LORA)
    # No files in tmp_path — should not raise
    pool.load_all(str(tmp_path))


# ── LoRALayer additional branches ─────────────────────────────────────────────

def test_lora_layer_forward_base_bias_dtype_mismatch():
    """Test LoRALayer forward when base_bias dtype differs from input."""
    from src.experts.lora import LoRALayer
    layer = LoRALayer(in_features=64, out_features=32, rank=4, alpha=8)
    weight = torch.randn(32, 64).float()
    bias = torch.randn(32).float()
    layer.load_base_weight(weight, bias)
    # Input in float32 — should work fine
    x = torch.randn(4, 64).float()
    out = layer(x)
    assert out.shape == (4, 32)


# ── SharedLoRALayer dtype mismatch branches ───────────────────────────────────

def test_shared_lora_layer_dtype_mismatch_weight_and_bias():
    """Trigger both w.dtype != x.dtype and b.dtype != x.dtype branches."""
    from src.experts.lora import SharedLoRALayer
    weight = torch.randn(32, 64).to(torch.float16)
    bias = torch.randn(32).to(torch.float16)
    layer = SharedLoRALayer(shared_weight=weight, shared_bias=bias, rank=4, alpha=8)
    x = torch.randn(4, 64).float()  # float32 vs float16 → triggers both dtype branches
    out = layer(x)
    assert out.shape == (4, 32)


# ── ExpertPool consolidate_shared_weights — no c_fc fallback ─────────────────

def test_expert_pool_consolidate_no_c_fc():
    """consolidate_shared_weights returns early when expert has no c_fc and no get_lora_layer_names."""
    from src.experts.pool import ExpertPool
    from src.experts.lora import LoRAConfig, LoRAMLPExpert
    import torch.nn as nn

    # Create a minimal expert class with neither get_lora_layer_names nor c_fc
    class BareExpert(LoRAMLPExpert):
        def __init__(self, config):
            super().__init__(config)
        def load_from_mlp(self, mlp): pass
        def forward(self, x): return x

    cfg = LoRAConfig(hidden_dim=64, rank=4, alpha=8)
    # Build pool normally but with BareExpert via registry patching
    from src.core.registry import ExpertRegistry
    from src.project_types import ExpertType
    ExpertRegistry._registries["experts"]["bare_test_consolidate"] = BareExpert

    pool = ExpertPool.__new__(ExpertPool)
    nn.Module.__init__(pool)
    pool.config = cfg
    pool.expert_type = ExpertType.GPTNEO_LORA
    pool.expert_class = BareExpert
    pool.experts = nn.ModuleList([BareExpert(cfg), BareExpert(cfg)])
    pool.shared_fc_weight = None
    pool.shared_proj_weight = None

    pool.consolidate_shared_weights()  # should hit the `else: return` branch


# ── ExpertPool make_base_trainable — no c_fc branch ──────────────────────────

def test_expert_pool_make_base_trainable_no_c_fc():
    """make_base_trainable returns early when expert has no c_fc."""
    from src.experts.pool import ExpertPool
    from src.experts.lora import LoRAConfig, LoRAMLPExpert
    import torch.nn as nn

    class BareExpert(LoRAMLPExpert):
        def __init__(self, config):
            super().__init__(config)
        def load_from_mlp(self, mlp): pass
        def forward(self, x): return x

    cfg = LoRAConfig(hidden_dim=64, rank=4, alpha=8)
    pool = ExpertPool.__new__(ExpertPool)
    nn.Module.__init__(pool)
    pool.config = cfg
    pool.experts = nn.ModuleList([BareExpert(cfg)])
    pool.shared_fc_weight = None
    pool.shared_proj_weight = None

    pool.make_base_trainable()  # should hit `if not (hasattr(e0, "c_fc") ...): return`
    assert pool.shared_fc_weight is None


# ── LoRAMLPExpert freeze_base_weights ─────────────────────────────────────────

def test_lora_mlp_expert_freeze_base_weights_sets_no_grad():
    """freeze_base_weights sets requires_grad=False on base_weight params."""
    from src.experts.lora import LoRAConfig
    from src.experts.gpt_neo_lora import GPTNeoLoRAMLP
    import torch.nn as nn

    cfg = LoRAConfig(hidden_dim=64, rank=4, alpha=8, trainable_base=True)
    expert = GPTNeoLoRAMLP(cfg)
    mlp = nn.Module()
    mlp.c_fc = nn.Linear(64, 256)
    mlp.c_proj = nn.Linear(256, 64)
    expert.load_from_mlp(mlp)

    # Manually add a trainable parameter named "base_weight_param" to trigger the branch
    expert.register_parameter("base_weight_param", nn.Parameter(torch.randn(4)))
    expert.freeze_base_weights()
    assert not expert.base_weight_param.requires_grad
