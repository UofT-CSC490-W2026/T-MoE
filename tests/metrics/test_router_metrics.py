import pytest

import torch

import numpy as np

from src.configs.router import MetabolicRouterConfig

from src.routers.metabolic import MetabolicRouter

from src.metrics.router_metrics import RouterMetricsTracker, GlobalSpecializationTracker


@pytest.fixture
def router():

    cfg = MetabolicRouterConfig(hidden_dim=64, num_experts=4, top_k=2)

    return MetabolicRouter(cfg)


@pytest.fixture
def tracker(router):

    return RouterMetricsTracker(router)


def _uniform_routing(num_experts=4, top_k=2, batch=2, seq=8):

    indices = torch.zeros(batch, seq, top_k, dtype=torch.long)

    for i in range(batch * seq):
        b, s = divmod(i, seq)

        for k in range(top_k):
            indices[b, s, k] = (i * top_k + k) % num_experts

    weights = torch.full((batch, seq, top_k), 1.0 / top_k)

    return indices, weights


def _collapsed_routing(num_experts=4, top_k=2, batch=2, seq=8):

    indices = torch.zeros(batch, seq, top_k, dtype=torch.long)

    weights = torch.full((batch, seq, top_k), 1.0 / top_k)

    return indices, weights


class TestComputeUsage:
    def test_returns_per_expert_tensor(self, tracker):

        indices, weights = _uniform_routing()

        usage = tracker._compute_usage(indices, weights)

        assert usage.shape == (tracker.num_experts,)

        assert usage.dtype == torch.float32

    def test_sums_to_total_weight(self, tracker):

        indices, weights = _uniform_routing()

        usage = tracker._compute_usage(indices, weights)

        expected = weights.sum().item()

        assert abs(usage.sum().item() - expected) < 1e-4

    def test_collapsed_routing_concentrates_on_expert_0(self, tracker):

        indices, weights = _collapsed_routing()

        usage = tracker._compute_usage(indices, weights)

        assert usage[0].item() > 0

        assert usage[1:].sum().item() == 0.0


class TestExpertEntropy:
    def test_returns_expected_keys(self, tracker):

        indices, weights = _uniform_routing()

        result = tracker.compute_expert_entropy(indices, weights)

        assert "expert_entropy" in result

        assert "expert_entropy_normalized" in result

    def test_uniform_routing_has_max_entropy(self, tracker):

        indices, weights = _uniform_routing(num_experts=4)

        result = tracker.compute_expert_entropy(indices, weights)

        assert result["expert_entropy_normalized"] > 0.9

    def test_collapsed_routing_has_zero_entropy(self, tracker):

        indices, weights = _collapsed_routing()

        result = tracker.compute_expert_entropy(indices, weights)

        assert result["expert_entropy_normalized"] < 0.1


class TestGiniCoefficient:
    def test_uniform_routing_near_zero_gini(self, tracker):

        indices, weights = _uniform_routing()

        gini = tracker.compute_gini_coefficient(indices, weights)

        assert gini < 0.2

    def test_collapsed_routing_high_gini(self, tracker):

        indices, weights = _collapsed_routing()

        gini = tracker.compute_gini_coefficient(indices, weights)

        assert gini > 0.5

    def test_gini_in_range(self, tracker):

        indices, weights = _uniform_routing()

        gini = tracker.compute_gini_coefficient(indices, weights)

        assert 0.0 <= gini <= 1.0


class TestEffectiveExperts:
    def test_uniform_routing_near_num_experts(self, tracker):

        indices, weights = _uniform_routing(num_experts=4)

        eff = tracker.compute_effective_experts(indices, weights)

        assert eff > 3.0

    def test_collapsed_routing_near_one(self, tracker):

        indices, weights = _collapsed_routing()

        eff = tracker.compute_effective_experts(indices, weights)

        assert eff < 1.5

    def test_accepts_precomputed_entropy(self, tracker):

        indices, weights = _uniform_routing()

        entropy_val = 1.0

        eff = tracker.compute_effective_experts(indices, weights, entropy=entropy_val)

        assert abs(eff - np.exp(1.0)) < 1e-5


class TestComputeAllMetrics:
    def test_includes_all_expected_keys(self, tracker):

        indices, weights = _uniform_routing()

        metrics = tracker.compute_all_metrics(indices, weights)

        for key in (
            "expert_entropy",
            "expert_entropy_normalized",
            "effective_experts",
            "routing_diversity_gini",
            "router_confidence_mean",
            "router_confidence_std",
            "top1_dominance",
            "usage_counts",
            "usage_distribution",
        ):
            assert key in metrics, f"Missing key: {key}"

    def test_fatigue_keys_present_for_metabolic(self, tracker):

        indices, weights = _uniform_routing()

        metrics = tracker.compute_all_metrics(indices, weights)

        assert "fatigue_mean" in metrics

        assert "fatigue_max" in metrics

    def test_entropy_computed_once(self, tracker):

        indices, weights = _uniform_routing()

        metrics = tracker.compute_all_metrics(indices, weights)

        expected_eff = np.exp(metrics["expert_entropy"])

        assert abs(metrics["effective_experts"] - expected_eff) < 1e-5


class TestGlobalSpecializationTracker:
    def test_update_increments_total_tokens(self):

        tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)

        token_ids = torch.randint(0, 100, (2, 8))

        expert_indices = torch.randint(0, 4, (2, 8, 2))

        tracker.update(token_ids, expert_indices)

        assert tracker.total_tokens == 16

    def test_update_filters_padding_tokens(self):

        tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)

        token_ids = torch.full((2, 8), -1, dtype=torch.long)

        expert_indices = torch.randint(0, 4, (2, 8, 2))

        tracker.update(token_ids, expert_indices)

        assert tracker.total_tokens == 0

    def test_compute_metrics_returns_empty_before_update(self):

        tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)

        assert tracker.compute_metrics() == {}

    def test_compute_metrics_returns_expected_keys(self):

        tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)

        token_ids = torch.randint(0, 100, (4, 16))

        expert_indices = torch.randint(0, 4, (4, 16, 2))

        tracker.update(token_ids, expert_indices)

        metrics = tracker.compute_metrics()

        for key in (
            "specialization_score",
            "collapse_score",
            "marginal_entropy",
            "conditional_entropy",
            "global_tokens_seen",
        ):
            assert key in metrics

    def test_specialization_score_bounded(self):

        tracker = GlobalSpecializationTracker(vocab_size=50, num_experts=4)

        token_ids = torch.randint(0, 50, (8, 32))

        expert_indices = torch.randint(0, 4, (8, 32, 2))

        tracker.update(token_ids, expert_indices)

        m = tracker.compute_metrics()

        assert 0.0 <= m["specialization_score"] <= 1.0

        assert 0.0 <= m["collapse_score"] <= 1.0
