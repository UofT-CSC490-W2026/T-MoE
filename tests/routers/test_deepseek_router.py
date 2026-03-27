import torch
from src.configs.router import DeepSeekRouterConfig
from src.routers.deepseek import DeepSeekRouter


def test_deepseek_router_forward(device):
    config = DeepSeekRouterConfig(
        hidden_dim=32, num_experts=4, top_k=2, bias_update_rate=0.01
    )

    router = DeepSeekRouter(config).to(device)

    x = torch.randn(2, 4, config.hidden_dim, device=device)

    weights, indices, metrics = router(x, return_metrics=True)

    N = 2 * 4

    assert weights.shape == (N, config.num_experts)

    assert indices is None

    assert router.bias.shape == (4,)

    router.step()

    assert not torch.allclose(router.bias, torch.zeros_like(router.bias))
