import torch

from src.configs.router import TopKRouterConfig

from src.routers.standard import TopKRouter


def test_topk_router_aux_loss_zero(device):

    config = TopKRouterConfig(hidden_dim=64, num_experts=4, top_k=2)

    router = TopKRouter(config).to(device)

    router.train()

    x = torch.randn(2, 4, config.hidden_dim, device=device)

    router(x, return_metrics=False)

    aux_loss = router.compute_aux_loss()

    print("topk_router_aux_loss:")

    print("  aux_loss:", aux_loss.item())

    assert aux_loss.item() == 0.0
