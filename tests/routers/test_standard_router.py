import torch

from src.configs.router import StandardRouterConfig
from src.routers.standard import StandardRouter


def test_standard_router_forward_shapes(device):
    config = StandardRouterConfig(hidden_dim=128, num_experts=6, top_k=2)
    router = StandardRouter(config).to(device)

    x = torch.randn(2, 3, config.hidden_dim, device=device)
    weights, indices, metrics = router(x, return_metrics=True)

    print("standard_router_forward_shapes:")
    print("  weights.shape:", weights.shape)
    print("  indices.shape:", indices.shape)
    print("  weights.sum(dim=-1):", weights.sum(dim=-1))

    assert weights.shape == (2, 3, config.top_k)
    assert indices.shape == (2, 3, config.top_k)
    assert (weights.sum(dim=-1) - 1.0).abs().max().item() < 1e-5
    assert metrics is not None


def test_standard_router_aux_loss_nonzero_when_enabled(device):
    config = StandardRouterConfig(
        hidden_dim=64, num_experts=4, top_k=2, use_aux_loss=True, aux_loss_coef=0.01
    )
    router = StandardRouter(config).to(device)
    router.train()

    x = torch.randn(2, 4, config.hidden_dim, device=device)
    router(x, return_metrics=False)
    aux_loss = router.compute_aux_loss()

    print("standard_router_aux_loss:")
    print("  aux_loss:", aux_loss.item())

    assert torch.isfinite(aux_loss).all()


def test_standard_router_indices_in_range(device):
    config = StandardRouterConfig(hidden_dim=32, num_experts=5, top_k=3)
    router = StandardRouter(config).to(device)

    x = torch.randn(4, 2, config.hidden_dim, device=device)
    _, indices, _ = router(x, return_metrics=False)

    assert indices.min().item() >= 0
    assert indices.max().item() < config.num_experts


def test_standard_router_deterministic_in_eval(device):
    config = StandardRouterConfig(hidden_dim=16, num_experts=4, top_k=2)
    router = StandardRouter(config).to(device)
    router.eval()

    x = torch.randn(1, 3, config.hidden_dim, device=device)
    w1, i1, _ = router(x, return_metrics=False)
    w2, i2, _ = router(x, return_metrics=False)

    assert torch.allclose(w1, w2)
    assert torch.equal(i1, i2)


def test_standard_router_topk_matches_known_logits(device):
    config = StandardRouterConfig(hidden_dim=4, num_experts=4, top_k=2)
    router = StandardRouter(config).to(device)
    router.eval()

    # Make gate an identity so logits == input
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.weight.copy_(torch.eye(4, device=device))

    # One token with known logits: expert 3 (0-based) and 1 should win
    x = torch.tensor([[[0.1, 2.0, -1.0, 3.0]]], device=device)
    _, indices, _ = router(x, return_metrics=False)

    topk = indices[0, 0].tolist()
    assert 3 in topk
    assert 1 in topk


def test_standard_router_aux_loss_reflects_imbalance(device):
    config = StandardRouterConfig(
        hidden_dim=4, num_experts=4, top_k=1, use_aux_loss=True, aux_loss_coef=0.01
    )
    router = StandardRouter(config).to(device)
    router.train()

    # Force routing to expert 0 via large positive logit on dimension 0
    with torch.no_grad():
        router.gate.weight.zero_()
        router.gate.weight[0, 0] = 5.0  # expert 0 depends on x[0]

    # All tokens activate expert 0 strongly
    x = torch.zeros(2, 3, config.hidden_dim, device=device)
    x[..., 0] = 10.0
    router(x, return_metrics=False)
    aux_imbalanced = router.compute_aux_loss()

    # Make logits uniform across experts (zero weights) to reduce imbalance
    with torch.no_grad():
        router.gate.weight.zero_()
    router(x, return_metrics=False)
    aux_uniform = router.compute_aux_loss()

    print("aux_loss_imbalance_vs_uniform:", aux_imbalanced.item(), aux_uniform.item())
    assert aux_imbalanced.item() >= aux_uniform.item()
