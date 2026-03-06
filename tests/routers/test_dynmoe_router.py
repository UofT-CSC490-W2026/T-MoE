import torch

from src.configs.router import DynMoERouterConfig
from src.routers.dynmoe import DynMoERouter


def test_dynmoe_router_shapes(device):
    config = DynMoERouterConfig(hidden_dim=32, num_experts=6, top_k=3)
    router = DynMoERouter(config).to(device)

    x = torch.randn(2, 4, config.hidden_dim, device=device)
    weights, indices, metrics = router(x, return_metrics=True)

    assert weights.shape == (2, 4, config.top_k)
    assert indices.shape == (2, 4, config.top_k)
    assert (weights.sum(dim=-1) - 1.0).abs().max().item() < 1e-5
    assert metrics is not None


def test_dynmoe_router_threshold_masks(device):
    config = DynMoERouterConfig(
        hidden_dim=4, num_experts=4, top_k=2, gate_threshold=0.9
    )
    router = DynMoERouter(config).to(device)
    router.eval()

    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.weight.copy_(torch.eye(4, device=device))

    # logits -> sigmoid => only large positive logit should pass threshold
    x = torch.tensor([[[10.0, -2.0, -2.0, -2.0]]], device=device)
    weights, indices, _ = router(x, return_metrics=False)

    # First expert should be selected with weight ~1
    assert indices[0, 0, 0].item() == 0
    assert weights[0, 0, 0].item() > 0.9


def test_dynmoe_router_aux_loss_optional(device):
    config = DynMoERouterConfig(
        hidden_dim=8, num_experts=4, top_k=2, use_aux_loss=True, aux_loss_coef=0.01
    )
    router = DynMoERouter(config).to(device)
    router.train()

    x = torch.randn(2, 3, config.hidden_dim, device=device)
    router(x, return_metrics=False)
    aux_loss = router.compute_aux_loss()

    assert torch.isfinite(aux_loss).all()
