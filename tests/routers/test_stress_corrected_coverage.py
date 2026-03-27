from __future__ import annotations

from unittest.mock import patch

import torch

from src.configs.router import StressCorrectedRouterConfig

from src.routers.stress_corrected import StressCorrectedRouter


def _make_router(**kwargs):

    defaults = dict(
        hidden_dim=16, num_experts=4, top_k=2, temperature=1.0, noise_std=0.0
    )

    defaults.update(kwargs)

    cfg = StressCorrectedRouterConfig(**defaults)

    return StressCorrectedRouter(cfg)


def test_sync_pending_counts_no_dist_module():

    router = _make_router()

    with patch.dict("sys.modules", {"torch.distributed": None}):
        router._sync_pending_counts_distributed()


def test_sync_pending_counts_not_initialized():

    router = _make_router()

    with patch("torch.distributed.is_initialized", return_value=False):
        router._sync_pending_counts_distributed()


def test_sync_pending_counts_world_size_1():

    router = _make_router()

    with patch("torch.distributed.is_initialized", return_value=True):
        with patch("torch.distributed.get_world_size", return_value=1):
            router._sync_pending_counts_distributed()


def test_sync_pending_counts_distributed():

    router = _make_router()

    with patch("torch.distributed.is_initialized", return_value=True):
        with patch("torch.distributed.get_world_size", return_value=2):
            with patch("torch.distributed.all_reduce") as mock_ar:
                router._sync_pending_counts_distributed()

    assert mock_ar.call_count == 2


def test_sync_ema_load_no_dist():

    router = _make_router()

    with patch.dict("sys.modules", {"torch.distributed": None}):
        router._sync_ema_load_distributed()


def test_sync_ema_load_not_initialized():

    router = _make_router()

    with patch("torch.distributed.is_initialized", return_value=False):
        router._sync_ema_load_distributed()


def test_sync_ema_load_world_size_1():

    router = _make_router()

    with patch("torch.distributed.is_initialized", return_value=True):
        with patch("torch.distributed.get_world_size", return_value=1):
            router._sync_ema_load_distributed()


def test_sync_ema_load_distributed():

    router = _make_router()

    with patch("torch.distributed.is_initialized", return_value=True):
        with patch("torch.distributed.get_world_size", return_value=2):
            with patch("torch.distributed.all_reduce") as mock_ar:
                router._sync_ema_load_distributed()

    mock_ar.assert_called_once()


def test_sync_lambda_no_dist():

    router = _make_router()

    with patch.dict("sys.modules", {"torch.distributed": None}):
        router._sync_lambda_distributed()


def test_sync_lambda_not_initialized():

    router = _make_router()

    with patch("torch.distributed.is_initialized", return_value=False):
        router._sync_lambda_distributed()


def test_sync_lambda_world_size_1():

    router = _make_router()

    with patch("torch.distributed.is_initialized", return_value=True):
        with patch("torch.distributed.get_world_size", return_value=1):
            router._sync_lambda_distributed()


def test_sync_lambda_distributed():

    router = _make_router()

    with patch("torch.distributed.is_initialized", return_value=True):
        with patch("torch.distributed.get_world_size", return_value=2):
            with patch("torch.distributed.broadcast") as mock_bc:
                router._sync_lambda_distributed()

    assert mock_bc.call_count == 2


def test_sync_welford_no_dist():

    router = _make_router()

    with patch.dict("sys.modules", {"torch.distributed": None}):
        router._sync_welford_distributed()


def test_sync_welford_not_initialized():

    router = _make_router()

    with patch("torch.distributed.is_initialized", return_value=False):
        router._sync_welford_distributed()


def test_sync_welford_world_size_1():

    router = _make_router()

    with patch("torch.distributed.is_initialized", return_value=True):
        with patch("torch.distributed.get_world_size", return_value=1):
            router._sync_welford_distributed()


def test_sync_welford_distributed():

    router = _make_router(num_experts=4)

    E = 4

    router.welford_n = torch.ones(E) * 10.0

    router.welford_mu = torch.rand(E)

    router.welford_M2 = torch.rand(E)

    def _fake_all_gather(out_list, tensor):

        for t in out_list:
            t.copy_(tensor)

    with patch("torch.distributed.is_initialized", return_value=True):
        with patch("torch.distributed.get_world_size", return_value=2):
            with patch("torch.distributed.all_gather", side_effect=_fake_all_gather):
                router._sync_welford_distributed()

    assert torch.all(router.welford_n >= 10.0)
