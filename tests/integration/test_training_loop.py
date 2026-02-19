import torch
import torch.nn as nn
import torch.optim as optim

from src.experts.lora import LoRAConfig
from src.layers.lora_moe import LoRAMoELayer
from src.routers.metabolic import MetabolicRouter
from configs.router import MetabolicRouterConfig


def test_training_integration():
    hidden_dim, intermediate_dim = 64, 256

    # Backbone MLP (will be frozen)
    backbone = nn.Sequential(
        nn.Linear(hidden_dim, intermediate_dim),
        nn.GELU(),
        nn.Linear(intermediate_dim, hidden_dim),
    )

    # Configs
    lora_cfg = LoRAConfig(
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        rank=4,
        alpha=16,
        dropout=0.1,
    )
    router_cfg = MetabolicRouterConfig(hidden_dim=hidden_dim, num_experts=4, top_k=2)

    # Build MoE layer
    moe = LoRAMoELayer(
        base_layer=backbone,
        router=MetabolicRouter(router_cfg),
        lora_config=lora_cfg,
        num_experts=4,
    )

    # ── verify freeze ──
    assert not backbone[0].weight.requires_grad, "Backbone should be frozen"
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

    assert backbone[0].weight.grad is None, "Backbone should receive no gradients"
    assert any(
        p.grad is not None for p in moe.router.parameters() if p.requires_grad
    ), "Router should receive gradients"

    opt.step()
