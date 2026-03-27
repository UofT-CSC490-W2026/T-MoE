import torch

from src.configs.router import ExpertChoiceRouterConfig

from src.routers.expert_choice import ExpertChoiceRouter


def test_expert_choice_router_forward(device):
    config = ExpertChoiceRouterConfig(hidden_dim=32, num_experts=4, top_k=2)

    router = ExpertChoiceRouter(config).to(device)

    x = torch.randn(2, 4, config.hidden_dim, device=device)

    weights, indices, metrics = router(x, return_metrics=True)

    N = 2 * 4

    assert weights.shape == (N, config.num_experts)

    assert indices is None

    assert "token_drop_rate" in metrics
