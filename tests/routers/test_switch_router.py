import torch

from src.configs.router import SwitchRouterConfig
from src.routers.standard import SwitchRouter


def test_switch_router_top1(device):
    config = SwitchRouterConfig(hidden_dim=32, num_experts=5)
    router = SwitchRouter(config).to(device)

    x = torch.randn(2, 3, config.hidden_dim, device=device)
    weights, indices, _ = router(x, return_metrics=False)

    assert weights.shape[-1] == 1
    assert indices.shape[-1] == 1
