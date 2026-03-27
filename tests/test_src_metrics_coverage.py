import torch
import numpy as np
from unittest.mock import patch, MagicMock


def _make_tracker(num_experts=4):
    from src.metrics.router_metrics import RouterMetricsTracker

    router = MagicMock()
    router.num_experts = num_experts
    router.top_k = 2
    return RouterMetricsTracker(router)


def test_compute_usage_dense():
    tracker = _make_tracker()
    weights = torch.ones(8, 4) * 0.25
    usage = tracker._compute_usage(None, weights)
    assert usage.shape == (4,)


def test_compute_usage_sparse():
    tracker = _make_tracker()
    indices = torch.randint(0, 4, (2, 4, 2))
    weights = torch.ones(2, 4, 2) * 0.5
    usage = tracker._compute_usage(indices, weights)
    assert usage.shape == (4,)


def test_compute_expert_entropy():
    tracker = _make_tracker()
    indices = torch.randint(0, 4, (2, 4, 2))
    weights = torch.ones(2, 4, 2) * 0.5
    result = tracker.compute_expert_entropy(indices, weights)
    assert "expert_entropy" in result
    assert "expert_entropy_normalized" in result


def test_compute_fatigue_stats_no_fatigue():
    from src.metrics.router_metrics import RouterMetricsTracker

    router = MagicMock(spec=[])
    router.num_experts = 4
    tracker = RouterMetricsTracker(router)
    result = tracker.compute_fatigue_stats()
    assert result == {}


def test_compute_fatigue_stats_with_fatigue():
    from src.metrics.router_metrics import RouterMetricsTracker

    router = MagicMock()
    router.num_experts = 4
    router.fatigue = torch.tensor([0.1, 0.2, 0.3, 0.4])
    tracker = RouterMetricsTracker(router)
    result = tracker.compute_fatigue_stats()
    assert "fatigue_mean" in result
    assert "fatigue_std" in result
    assert "fatigue_min" in result
    assert "fatigue_max" in result


def test_compute_usage_distribution():
    tracker = _make_tracker()
    indices = torch.randint(0, 4, (2, 4, 2))
    weights = torch.ones(2, 4, 2) * 0.5
    result = tracker.compute_usage_distribution(indices, weights)
    assert "usage_counts" in result
    assert "usage_distribution" in result


def test_compute_gini_coefficient():
    tracker = _make_tracker()
    indices = torch.randint(0, 4, (2, 4, 2))
    weights = torch.ones(2, 4, 2) * 0.5
    gini = tracker.compute_gini_coefficient(indices, weights)
    assert 0.0 <= gini <= 1.0


def test_compute_gini_with_precomputed_usage():
    tracker = _make_tracker()
    usage = torch.tensor([1.0, 0.0, 0.0, 0.0])
    gini = tracker.compute_gini_coefficient(None, torch.zeros(8, 4), usage=usage)
    assert gini > 0.5


def test_compute_effective_experts():
    tracker = _make_tracker()
    indices = torch.randint(0, 4, (2, 4, 2))
    weights = torch.ones(2, 4, 2) * 0.5
    eff = tracker.compute_effective_experts(indices, weights)
    assert eff > 0.0


def test_compute_effective_experts_with_entropy():
    tracker = _make_tracker()
    indices = torch.randint(0, 4, (2, 4, 2))
    weights = torch.ones(2, 4, 2) * 0.5
    eff = tracker.compute_effective_experts(indices, weights, entropy=1.386)
    assert eff > 0.0


def test_compute_confidence_metrics():
    tracker = _make_tracker()
    weights = torch.tensor([[[0.8, 0.2], [0.6, 0.4]]])
    result = tracker.compute_confidence_metrics(weights)
    assert "router_confidence_mean" in result
    assert "router_confidence_std" in result
    assert "top1_dominance" in result


def test_compute_all_metrics_with_num_steps():
    from src.metrics.router_metrics import RouterMetricsTracker

    router = MagicMock()
    router.num_experts = 4
    router.num_steps = torch.tensor(42)
    tracker = RouterMetricsTracker(router)
    indices = torch.randint(0, 4, (2, 4, 2))
    weights = torch.ones(2, 4, 2) * 0.5
    metrics = tracker.compute_all_metrics(indices, weights)
    assert metrics["num_steps"] == 42


def test_compute_all_metrics_with_custom_metrics():
    from src.metrics.router_metrics import RouterMetricsTracker

    router = MagicMock()
    router.num_experts = 4
    router.get_custom_metrics.return_value = {"custom_key": 1.0}
    tracker = RouterMetricsTracker(router)
    indices = torch.randint(0, 4, (2, 4, 2))
    weights = torch.ones(2, 4, 2) * 0.5
    metrics = tracker.compute_all_metrics(indices, weights)
    assert "custom_key" in metrics


def test_log_to_wandb_not_available():
    tracker = _make_tracker()
    with patch("src.metrics.router_metrics.WANDB_AVAILABLE", False):
        tracker.log_to_wandb({}, step=0)


def test_log_to_wandb_no_run():
    tracker = _make_tracker()
    with patch("src.metrics.router_metrics.WANDB_AVAILABLE", True):
        with patch("src.metrics.router_metrics.wandb") as mock_wandb:
            mock_wandb.run = None
            tracker.log_to_wandb({"loss": 1.0}, step=0)


def test_log_to_wandb_with_run():
    tracker = _make_tracker()
    with patch("src.metrics.router_metrics.WANDB_AVAILABLE", True):
        with patch("src.metrics.router_metrics.wandb") as mock_wandb:
            mock_run = MagicMock()
            mock_wandb.run = mock_run
            mock_wandb.Histogram = MagicMock(return_value=MagicMock())
            metrics = {
                "expert_entropy": 1.5,
                "fatigue_per_expert": np.array([0.1, 0.2, 0.3, 0.4]),
                "usage_distribution": np.array([0.25, 0.25, 0.25, 0.25]),
                "stress_per_expert": np.array([0.1, 0.2, 0.3, 0.4]),
                "ema_load_per_expert": np.array([0.25, 0.25, 0.25, 0.25]),
            }
            tracker.log_to_wandb(metrics, step=10, prefix="router")
            mock_wandb.log.assert_called()


def test_global_spec_tracker_update():
    from src.metrics.router_metrics import GlobalSpecializationTracker

    tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
    token_ids = torch.randint(0, 100, (2, 8))
    expert_indices = torch.randint(0, 4, (2, 8, 2))
    tracker.update(token_ids, expert_indices)
    assert tracker.total_tokens > 0


def test_global_spec_tracker_update_none_indices():
    from src.metrics.router_metrics import GlobalSpecializationTracker

    tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
    token_ids = torch.randint(0, 100, (2, 8))
    tracker.update(token_ids, None)
    assert tracker.total_tokens == 0


def test_global_spec_tracker_update_empty_valid():
    from src.metrics.router_metrics import GlobalSpecializationTracker

    tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
    token_ids = torch.full((2, 8), 200)
    expert_indices = torch.randint(0, 4, (2, 8, 2))
    tracker.update(token_ids, expert_indices)
    assert tracker.total_tokens == 0


def test_global_spec_tracker_compute_metrics_empty():
    from src.metrics.router_metrics import GlobalSpecializationTracker

    tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
    result = tracker.compute_metrics()
    assert result == {}


def test_global_spec_tracker_compute_metrics():
    from src.metrics.router_metrics import GlobalSpecializationTracker

    tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
    token_ids = torch.randint(0, 100, (4, 16))
    expert_indices = torch.randint(0, 4, (4, 16, 2))
    tracker.update(token_ids, expert_indices)
    result = tracker.compute_metrics()
    assert "specialization_score" in result
    assert "collapse_score" in result


def test_global_spec_tracker_compute_metrics_uniform():
    from src.metrics.router_metrics import GlobalSpecializationTracker

    tracker = GlobalSpecializationTracker(vocab_size=4, num_experts=4)
    token_ids = torch.arange(4).unsqueeze(1).expand(4, 4).contiguous()
    expert_indices = torch.stack(
        [
            torch.zeros(4, 4, dtype=torch.long),
            torch.ones(4, 4, dtype=torch.long),
            torch.full((4, 4), 2, dtype=torch.long),
            torch.full((4, 4), 3, dtype=torch.long),
        ],
        dim=2,
    )
    expert_indices = torch.randint(0, 4, (4, 4, 2))
    tracker.update(token_ids, expert_indices)
    result = tracker.compute_metrics()
    assert "collapse_score" in result


def test_global_spec_tracker_sync_not_distributed():
    from src.metrics.router_metrics import GlobalSpecializationTracker

    tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
    token_ids = torch.randint(0, 100, (2, 8))
    expert_indices = torch.randint(0, 4, (2, 8, 2))
    tracker.update(token_ids, expert_indices)
    result = tracker.sync_and_compute("cpu", is_distributed=False)
    assert isinstance(result, dict)


def test_global_spec_tracker_update_with_negative_experts():
    from src.metrics.router_metrics import GlobalSpecializationTracker

    tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
    token_ids = torch.randint(0, 100, (2, 4))
    expert_indices = torch.tensor(
        [[[-1, 0], [1, -1], [2, 3], [0, 1]], [[-1, -1], [0, 1], [2, 3], [-1, 0]]]
    )
    tracker.update(token_ids, expert_indices)


def test_global_spec_tracker_compute_metrics_zero_tokens():
    from src.metrics.router_metrics import GlobalSpecializationTracker

    tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
    result = tracker.compute_metrics()
    assert result == {}


def test_global_spec_tracker_compute_metrics_no_active_mask():
    from src.metrics.router_metrics import GlobalSpecializationTracker

    tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
    tracker.total_tokens = 10
    result = tracker.compute_metrics()
    assert result == {}


def test_global_spec_tracker_update_all_padding_experts():
    from src.metrics.router_metrics import GlobalSpecializationTracker

    tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
    token_ids = torch.randint(0, 100, (2, 8))
    expert_indices = torch.full((2, 8, 2), -1, dtype=torch.long)
    tracker.update(token_ids, expert_indices)
    assert tracker.total_tokens == 0


def test_global_spec_tracker_sync_non_distributed():
    from src.metrics.router_metrics import GlobalSpecializationTracker

    tracker = GlobalSpecializationTracker(vocab_size=10, num_experts=4)
    token_ids = torch.arange(10).unsqueeze(0)
    expert_indices = torch.randint(0, 4, (1, 10, 1))
    tracker.update(token_ids, expert_indices)
    result = tracker.sync_and_compute(device="cpu", is_distributed=False)
    assert isinstance(result, dict)
