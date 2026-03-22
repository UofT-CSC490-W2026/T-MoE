import torch
import torch.nn as nn
import torch.optim as optim

from src.experts.lora import LoRAConfig
from src.layers.lora_moe import LoRAMoELayer
from src.routers.metabolic import MetabolicRouter
from src.configs.router import MetabolicRouterConfig


class MockMLP(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()
        self.c_fc = nn.Linear(hidden_dim, intermediate_dim)
        self.act = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(intermediate_dim, hidden_dim)

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


def test_training_integration():
    hidden_dim, intermediate_dim = 64, 256

    # Backbone MLP (will be frozen)
    backbone = MockMLP(hidden_dim, intermediate_dim)

    # Configs
    lora_cfg = LoRAConfig(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        rank=4,
        alpha=16,
        dropout=0.1,
    )
    # tau_specialization=0.5 → threshold = 0.5/4 = 12.5%; top-2/4 routing gives ~25%
    # usage per expert → always above threshold → fatigue accumulates deterministically.
    router_cfg = MetabolicRouterConfig(
        hidden_dim=hidden_dim, num_experts=4, top_k=2, tau_specialization=0.5
    )

    # Build MoE layer using from_pretrained_mlp to trigger load_from_mlp
    moe = LoRAMoELayer.from_pretrained_mlp(
        mlp=backbone,
        router=MetabolicRouter(router_cfg),
        lora_config=lora_cfg,
        num_experts=4,
    )

    # ── verify freeze ──
    # Base weights are stored as buffers (register_buffer) inside SharedLoRALayer,
    # NOT as parameters. They don't appear in moe.parameters() and receive no gradients.
    expert_0 = moe.expert_pool[0]
    assert not expert_0.c_fc.shared_weight.requires_grad, (
        "Expert base weights should be frozen (buffer)"
    )
    assert any(p.requires_grad for p in moe.router.parameters()), (
        "Router should be trainable"
    )
    assert any(p.requires_grad for p in moe.expert_pool.parameters()), (
        "Experts should be trainable"
    )

    # ── one training step ──
    opt = optim.AdamW((p for p in moe.parameters() if p.requires_grad), lr=1e-3)
    x = torch.randn(8, 16, hidden_dim)

    out, metrics = moe(x, return_metrics=True)
    loss = out.mean()

    opt.zero_grad()
    loss.backward()

    assert expert_0.c_fc.shared_weight.grad is None, (
        "Base weight buffer should receive no gradients"
    )
    assert any(
        p.grad is not None for p in moe.router.parameters() if p.requires_grad
    ), "Router should receive gradients"

    opt.step()

    # ── router.step() applies deferred fatigue and increments counter ──
    fatigue_before = moe.router.fatigue.clone()
    moe.router.step()
    assert moe.router.num_steps.item() == 1
    assert not moe.router._usage_pending

    # Fatigue should have changed (tokens were routed during forward)
    assert not torch.allclose(moe.router.fatigue, fatigue_before), (
        "Fatigue should update after router.step()"
    )

    # ── multi-step: fatigue accumulates and recovers ──
    moe.router.gamma_recovery = 0.0  # disable recovery to verify accumulation
    moe.router.beta_cost = 1.0

    fatigue_step1 = moe.router.fatigue.clone()
    out = moe(x)
    out.mean().backward()
    opt.step()
    moe.router.step()

    # At least one expert should have changed fatigue
    assert not torch.allclose(moe.router.fatigue, fatigue_step1)


def test_gradient_accumulation_defers_fatigue():
    """Fatigue should only update once per optimizer.step(), not per forward pass."""
    hidden_dim = 64
    cfg = MetabolicRouterConfig(hidden_dim=hidden_dim, num_experts=4, top_k=2)
    lora_cfg = LoRAConfig(hidden_dim=hidden_dim, rank=4, alpha=16)
    moe = LoRAMoELayer.from_pretrained_mlp(
        mlp=MockMLP(hidden_dim, 256),
        router=MetabolicRouter(cfg),
        lora_config=lora_cfg,
        num_experts=4,
    )
    moe.train()

    x = torch.randn(4, 8, hidden_dim)
    # Two forward passes (simulating grad accumulation)
    moe(x)
    moe(x)

    # Usage should be pending (accumulated across both forwards)
    assert moe.router._usage_pending
    assert moe.router.num_steps.item() == 0  # step() not called yet

    moe.router.step()
    assert moe.router.num_steps.item() == 1  # single step despite two forwards
