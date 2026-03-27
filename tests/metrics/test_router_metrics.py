"""
Tests for src/metrics/router_metrics.py

Chose this module because it's the most complex metrics code in the repo —
two classes, a bunch of branching logic, information-theoretic formulas,
and several easy-to-miss edge cases around padding, zero usage, and
distributed state management.

Covers:
- GlobalSpecializationTracker: padding tokens/experts, accumulation, empty state,
  specialization score formula, sync non-destructiveness, global_tokens_seen
- RouterMetricsTracker: dense vs sparse usage, negative indices, Gini correctness,
  entropy stability, effective experts identity, confidence metrics, fatigue
  presence/absence, usage distribution normalization, single-expert boundary,
  Gini cache, compute_all_metrics consistency
"""

import math
import pytest
import torch
import numpy as np

from src.configs.router import MetabolicRouterConfig, StressCorrectedRouterConfig
from src.routers.metabolic import MetabolicRouter
from src.routers.stress_corrected import StressCorrectedRouter
from src.metrics.router_metrics import RouterMetricsTracker, GlobalSpecializationTracker


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def metabolic_router():
    cfg = MetabolicRouterConfig(hidden_dim=32, num_experts=4, top_k=2)
    return MetabolicRouter(cfg)


@pytest.fixture
def spar_router():
    cfg = StressCorrectedRouterConfig(hidden_dim=32, num_experts=4, top_k=2)
    return StressCorrectedRouter(cfg)


@pytest.fixture
def tracker(metabolic_router):
    return RouterMetricsTracker(metabolic_router)


def _make_uniform_routing(num_experts=4, top_k=2, batch=2, seq=8):
    # Cycles indices evenly across all experts — near-max entropy
    total = batch * seq
    indices = torch.zeros(batch, seq, top_k, dtype=torch.long)
    for i in range(total):
        b, s = divmod(i, seq)
        for k in range(top_k):
            indices[b, s, k] = (i * top_k + k) % num_experts
    weights = torch.full((batch, seq, top_k), 1.0 / top_k)
    return indices, weights


def _make_collapsed_routing(num_experts=4, top_k=2, batch=2, seq=8):
    # Everything goes to expert 0
    indices = torch.zeros(batch, seq, top_k, dtype=torch.long)
    weights = torch.full((batch, seq, top_k), 1.0 / top_k)
    return indices, weights


# ---------------------------------------------------------------------------
# _compute_usage — dense path (indices=None)
# ---------------------------------------------------------------------------

class TestComputeUsageDensePath:
    # Expert-choice routers pass weights as [N, E] with no indices.
    # This path just sums columns — easy to break with a wrong dim or missing branch.

    def test_dense_path_returns_correct_shape(self, tracker):
        weights = torch.rand(16, 4)
        usage = tracker._compute_usage(None, weights)
        assert usage.shape == (4,)

    def test_dense_path_sums_columns(self, tracker):
        # Each expert should get exactly its column sum
        weights = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                 [0.0, 2.0, 0.0, 0.0],
                                 [0.0, 0.0, 3.0, 0.0],
                                 [0.0, 0.0, 0.0, 4.0]])
        usage = tracker._compute_usage(None, weights)
        assert torch.allclose(usage, torch.tensor([1.0, 2.0, 3.0, 4.0]), atol=1e-6)

    def test_dense_path_output_is_float32(self, tracker):
        # float16 input should still give float32 usage — needed for stable entropy
        weights = torch.rand(8, 4, dtype=torch.float16)
        usage = tracker._compute_usage(None, weights)
        assert usage.dtype == torch.float32


# ---------------------------------------------------------------------------
# _compute_usage — negative indices (-1 from adaptive-k)
# ---------------------------------------------------------------------------

class TestComputeUsageNegativeIndices:
    # adaptive-k can produce -1 indices when fewer than top_k experts are selected.
    # Without clamp(min=0), -1 wraps to the last expert and silently inflates it.

    def test_negative_indices_do_not_inflate_last_expert(self, tracker):
        # All -1 with zero weights — nothing should accumulate anywhere
        indices = torch.full((2, 4, 2), -1, dtype=torch.long)
        weights = torch.zeros(2, 4, 2)
        usage = tracker._compute_usage(indices, weights)
        assert usage.sum().item() == pytest.approx(0.0, abs=1e-7)

    def test_mixed_valid_and_negative_indices(self, tracker):
        # Expert 0 gets real weight; the -1 slot has weight 0 so it's a no-op
        indices = torch.tensor([[[0, -1]]], dtype=torch.long)
        weights = torch.tensor([[[0.8, 0.0]]])
        usage = tracker._compute_usage(indices, weights)
        assert usage[0].item() == pytest.approx(0.8, abs=1e-6)
        assert usage[1:].sum().item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Gini coefficient — mathematical correctness
# ---------------------------------------------------------------------------

class TestGiniCoefficientCorrectness:
    # The Gini formula has a specific closed form — "looks right" isn't enough.
    # An off-by-one in the (n+1)/n term gives plausible but wrong values.

    def test_perfect_balance_gives_zero_gini(self, tracker):
        usage = torch.ones(4) / 4.0
        indices = torch.zeros(1, 1, 1, dtype=torch.long)
        gini = tracker.compute_gini_coefficient(indices, None, usage=usage)
        assert gini == pytest.approx(0.0, abs=1e-5)

    def test_single_expert_monopoly_gives_near_one_gini(self, tracker):
        # For N=4, one expert taking everything → Gini = (N-1)/N = 0.75
        usage = torch.tensor([1.0, 0.0, 0.0, 0.0])
        indices = torch.zeros(1, 1, 1, dtype=torch.long)
        gini = tracker.compute_gini_coefficient(indices, None, usage=usage)
        assert gini == pytest.approx(0.75, abs=1e-4)

    def test_gini_is_scale_invariant(self, tracker):
        # Multiplying all usage by 100 shouldn't change the Gini
        usage_base = torch.tensor([0.4, 0.3, 0.2, 0.1])
        usage_scaled = usage_base * 100.0
        indices = torch.zeros(1, 1, 1, dtype=torch.long)
        gini_base = tracker.compute_gini_coefficient(indices, None, usage=usage_base)
        gini_scaled = tracker.compute_gini_coefficient(indices, None, usage=usage_scaled)
        assert gini_base == pytest.approx(gini_scaled, abs=1e-5)

    def test_gini_always_in_unit_interval(self, tracker):
        torch.manual_seed(42)
        for _ in range(20):
            usage = torch.rand(4).abs()
            indices = torch.zeros(1, 1, 1, dtype=torch.long)
            gini = tracker.compute_gini_coefficient(indices, None, usage=usage)
            assert 0.0 <= gini <= 1.0 + 1e-6, f"Gini out of range: {gini}"


# ---------------------------------------------------------------------------
# Entropy — numerical stability with near-zero usage
# ---------------------------------------------------------------------------

class TestEntropyNumericalStability:
    # Early in training some experts get almost no tokens.
    # A NaN entropy here would corrupt all downstream metrics and WandB.

    def test_near_zero_usage_no_nan(self, tracker):
        usage = torch.tensor([0.5, 0.3, 0.2, 1e-9])
        indices = torch.zeros(1, 1, 1, dtype=torch.long)
        result = tracker.compute_expert_entropy(indices, None, usage=usage)
        assert not math.isnan(result["expert_entropy"])
        assert not math.isinf(result["expert_entropy"])

    def test_all_zero_usage_no_nan(self, tracker):
        # Degenerate: no tokens routed anywhere — should still not blow up
        usage = torch.zeros(4)
        indices = torch.zeros(1, 1, 1, dtype=torch.long)
        result = tracker.compute_expert_entropy(indices, None, usage=usage)
        assert not math.isnan(result["expert_entropy"])

    def test_normalized_entropy_bounded_by_one(self, tracker):
        indices, weights = _make_uniform_routing(num_experts=4)
        result = tracker.compute_expert_entropy(indices, weights)
        assert 0.0 <= result["expert_entropy_normalized"] <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Effective experts — must equal exp(entropy)
# ---------------------------------------------------------------------------

class TestEffectiveExpertsInvariant:
    # effective_experts = exp(entropy) is a mathematical identity.
    # If the wrong entropy (e.g. normalized) is used, the metric is silently wrong.

    def test_effective_experts_equals_exp_entropy(self, tracker):
        indices, weights = _make_uniform_routing(num_experts=4)
        entropy = tracker.compute_expert_entropy(indices, weights)["expert_entropy"]
        eff = tracker.compute_effective_experts(indices, weights)
        assert eff == pytest.approx(math.exp(entropy), rel=1e-5)

    def test_effective_experts_precomputed_entropy_used(self, tracker):
        # Passing entropy=2.0 should give exp(2.0), not recompute from weights
        indices, weights = _make_uniform_routing()
        eff = tracker.compute_effective_experts(indices, weights, entropy=2.0)
        assert eff == pytest.approx(math.exp(2.0), rel=1e-6)

    def test_effective_experts_range_uniform(self, tracker):
        indices, weights = _make_uniform_routing(num_experts=4)
        assert tracker.compute_effective_experts(indices, weights) > 3.5

    def test_effective_experts_range_collapsed(self, tracker):
        indices, weights = _make_collapsed_routing(num_experts=4)
        assert tracker.compute_effective_experts(indices, weights) < 1.5


# ---------------------------------------------------------------------------
# Confidence metrics — top_k=1 degenerate case
# ---------------------------------------------------------------------------

class TestConfidenceMetricsTopK1:
    # With top_k=1 each token has one expert with weight 1.0.
    # top1_dominance divides max_weight by weight_sum — both are 1.0 so result is 1.0.
    # A missing clamp_min on weight_sum would give NaN here.

    def test_top1_dominance_is_one_for_top_k_1(self, tracker):
        weights = torch.ones(2, 8, 1)
        result = tracker.compute_confidence_metrics(weights)
        assert result["top1_dominance"] == pytest.approx(1.0, abs=1e-6)

    def test_confidence_mean_equals_one_for_top_k_1(self, tracker):
        weights = torch.ones(2, 8, 1)
        result = tracker.compute_confidence_metrics(weights)
        assert result["router_confidence_mean"] == pytest.approx(1.0, abs=1e-6)

    def test_confidence_std_zero_for_uniform_weights(self, tracker):
        # All tokens identical → std should be 0
        weights = torch.full((4, 8, 2), 0.5)
        result = tracker.compute_confidence_metrics(weights)
        assert result["router_confidence_std"] == pytest.approx(0.0, abs=1e-5)

    def test_top1_dominance_less_than_one_for_equal_top2(self, tracker):
        # Equal split across 2 experts → top1 gets 50% of weight
        weights = torch.full((2, 4, 2), 0.5)
        result = tracker.compute_confidence_metrics(weights)
        assert result["top1_dominance"] == pytest.approx(0.5, abs=1e-5)


# ---------------------------------------------------------------------------
# compute_all_metrics — usage computed once and shared
# ---------------------------------------------------------------------------

class TestComputeAllMetricsUsageReuse:
    # compute_all_metrics passes the same usage tensor to every sub-metric.
    # If any sub-function ignores the usage= kwarg and recomputes, results
    # can diverge due to floating-point ordering.

    def test_entropy_consistent_with_standalone(self, tracker):
        indices, weights = _make_uniform_routing()
        all_metrics = tracker.compute_all_metrics(indices, weights)
        usage = tracker._compute_usage(indices, weights)
        standalone = tracker.compute_expert_entropy(indices, weights, usage=usage)
        assert all_metrics["expert_entropy"] == pytest.approx(
            standalone["expert_entropy"], rel=1e-6
        )

    def test_gini_consistent_with_standalone(self, tracker):
        indices, weights = _make_uniform_routing()
        all_metrics = tracker.compute_all_metrics(indices, weights)
        usage = tracker._compute_usage(indices, weights)
        standalone_gini = tracker.compute_gini_coefficient(indices, weights, usage=usage)
        assert all_metrics["routing_diversity_gini"] == pytest.approx(
            standalone_gini, rel=1e-6
        )

    def test_effective_experts_uses_entropy_from_all_metrics(self, tracker):
        # effective_experts = exp(expert_entropy) must hold inside compute_all_metrics
        indices, weights = _make_uniform_routing()
        metrics = tracker.compute_all_metrics(indices, weights)
        assert metrics["effective_experts"] == pytest.approx(
            math.exp(metrics["expert_entropy"]), rel=1e-5
        )


# ---------------------------------------------------------------------------
# compute_all_metrics — router-specific conditional keys
# ---------------------------------------------------------------------------

class TestComputeAllMetricsRouterSpecificKeys:
    # fatigue, num_steps, and custom metrics are gated on hasattr checks.
    # A buffer rename would silently drop them — these tests catch that.

    def test_fatigue_keys_present_for_metabolic_router(self, tracker):
        indices, weights = _make_uniform_routing()
        metrics = tracker.compute_all_metrics(indices, weights)
        for key in ("fatigue_mean", "fatigue_std", "fatigue_min", "fatigue_max"):
            assert key in metrics, f"Missing fatigue key: {key}"

    def test_fatigue_keys_absent_for_spar_router(self, spar_router):
        # SPAR has no fatigue buffer — these keys must not appear
        spar_tracker = RouterMetricsTracker(spar_router)
        indices, weights = _make_uniform_routing()
        metrics = spar_tracker.compute_all_metrics(indices, weights)
        assert "fatigue_mean" not in metrics

    def test_num_steps_present_for_spar_router(self, spar_router):
        spar_tracker = RouterMetricsTracker(spar_router)
        indices, weights = _make_uniform_routing()
        metrics = spar_tracker.compute_all_metrics(indices, weights)
        assert "num_steps" in metrics
        assert metrics["num_steps"] == 0

    def test_custom_metrics_hook_called_for_spar_router(self, spar_router):
        # SPAR implements get_custom_metrics — lambda_val and ema_load_mean should show up
        spar_tracker = RouterMetricsTracker(spar_router)
        indices, weights = _make_uniform_routing()
        metrics = spar_tracker.compute_all_metrics(indices, weights)
        assert "lambda_val" in metrics
        assert "ema_load_mean" in metrics


# ---------------------------------------------------------------------------
# GlobalSpecializationTracker — padding token filtering
# ---------------------------------------------------------------------------

class TestGlobalSpecializationTrackerPaddingFiltering:
    # Padding tokens (id < 0 or >= vocab_size) must be silently dropped.
    # Including them corrupts H(E|T) since they're not real vocabulary items.

    def test_tokens_below_zero_are_filtered(self):
        tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
        token_ids = torch.full((2, 8), -1, dtype=torch.long)
        expert_indices = torch.randint(0, 4, (2, 8, 2))
        tracker.update(token_ids, expert_indices)
        assert tracker.total_tokens == 0
        assert tracker.usage_counts.sum().item() == 0

    def test_tokens_at_vocab_boundary_are_filtered(self):
        # token_id == vocab_size is out of range — valid range is [0, vocab_size-1]
        tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
        token_ids = torch.full((1, 4), 100, dtype=torch.long)
        expert_indices = torch.randint(0, 4, (1, 4, 2))
        tracker.update(token_ids, expert_indices)
        assert tracker.total_tokens == 0

    def test_mixed_valid_and_padding_tokens(self):
        tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
        token_ids = torch.cat([
            torch.randint(0, 100, (1, 4)),
            torch.full((1, 4), -1, dtype=torch.long),
        ], dim=0)
        expert_indices = torch.randint(0, 4, (2, 4, 2))
        tracker.update(token_ids, expert_indices)
        assert tracker.total_tokens == 4


# ---------------------------------------------------------------------------
# GlobalSpecializationTracker — padding expert filtering (-1 in adaptive-k)
# ---------------------------------------------------------------------------

class TestGlobalSpecializationTrackerPaddingExperts:
    # adaptive-k can produce -1 expert indices when fewer than top_k experts are chosen.
    # Without the expert_mask filter, -1 reaches bincount and either crashes or corrupts.

    def test_all_negative_expert_indices_no_crash(self):
        # All expert indices are -1 — the expert_mask early-return fires before
        # total_tokens is incremented, so both stay at 0.
        tracker = GlobalSpecializationTracker(vocab_size=50, num_experts=4)
        token_ids = torch.randint(0, 50, (2, 4))
        expert_indices = torch.full((2, 4, 2), -1, dtype=torch.long)
        tracker.update(token_ids, expert_indices)
        assert tracker.usage_counts.sum().item() == 0
        assert tracker.total_tokens == 0

    def test_mixed_valid_and_negative_expert_indices(self):
        # top_k=2: expert 0 is valid, -1 is padding — only expert 0 gets a count
        tracker = GlobalSpecializationTracker(vocab_size=50, num_experts=4)
        token_ids = torch.zeros(1, 1, dtype=torch.long)
        expert_indices = torch.tensor([[[0, -1]]])
        tracker.update(token_ids, expert_indices)
        assert tracker.usage_counts[0, 0].item() == 1
        assert tracker.usage_counts[0, 1:].sum().item() == 0


# ---------------------------------------------------------------------------
# GlobalSpecializationTracker — accumulation across multiple updates
# ---------------------------------------------------------------------------

class TestGlobalSpecializationTrackerAccumulation:
    # update() is called once per batch — counts must add up, not reset.
    # H(E|T) needs many batches to be meaningful, so this is critical.

    def test_multiple_updates_accumulate_total_tokens(self):
        tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
        token_ids = torch.randint(0, 100, (2, 8))
        expert_indices = torch.randint(0, 4, (2, 8, 2))
        tracker.update(token_ids, expert_indices)
        tracker.update(token_ids, expert_indices)
        assert tracker.total_tokens == 32  # 2 calls × 16 tokens

    def test_multiple_updates_accumulate_usage_counts(self):
        tracker = GlobalSpecializationTracker(vocab_size=10, num_experts=2)
        token_ids = torch.zeros(1, 1, dtype=torch.long)
        expert_indices = torch.zeros(1, 1, 1, dtype=torch.long)
        for _ in range(5):
            tracker.update(token_ids, expert_indices)
        assert tracker.usage_counts[0, 0].item() == 5

    def test_none_expert_indices_skips_update(self):
        # Dense routers return None for expert_indices — should be a no-op
        tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
        token_ids = torch.randint(0, 100, (2, 8))
        tracker.update(token_ids, None)
        assert tracker.total_tokens == 0
        assert tracker.usage_counts.sum().item() == 0


# ---------------------------------------------------------------------------
# GlobalSpecializationTracker.compute_metrics — specialization score
# ---------------------------------------------------------------------------

class TestSpecializationScoreFormula:
    # specialization_score = 1 - H(E|T) / H(E)
    # Swapping numerator/denominator or using the wrong axis gives a plausible
    # but wrong metric that would mislead experiment analysis.

    def test_perfect_specialization_score_near_one(self):
        # Token i always goes to expert i → H(E|T) ≈ 0 → score ≈ 1
        tracker = GlobalSpecializationTracker(vocab_size=4, num_experts=4)
        for token in range(4):
            token_ids = torch.full((1, 10), token, dtype=torch.long)
            expert_indices = torch.full((1, 10, 1), token, dtype=torch.long)
            tracker.update(token_ids, expert_indices)
        metrics = tracker.compute_metrics()
        assert metrics["specialization_score"] > 0.8

    def test_random_routing_specialization_score_near_zero(self):
        # Random routing → H(E|T) ≈ H(E) → score ≈ 0
        torch.manual_seed(0)
        tracker = GlobalSpecializationTracker(vocab_size=20, num_experts=4)
        for _ in range(50):
            token_ids = torch.randint(0, 20, (4, 16))
            expert_indices = torch.randint(0, 4, (4, 16, 2))
            tracker.update(token_ids, expert_indices)
        metrics = tracker.compute_metrics()
        assert metrics["specialization_score"] < 0.3

    def test_degenerate_marginal_entropy_zero_branch(self):
        # All tokens go to expert 0 → H(E) ≈ 0 → degenerate branch fires.
        # Known bug: the branch assigns Python floats (0.0, 1.0) but the return
        # statement calls .item() on them, which raises AttributeError.
        # This test pins that behavior so the bug doesn't go unnoticed.
        tracker = GlobalSpecializationTracker(vocab_size=10, num_experts=4)
        token_ids = torch.randint(0, 10, (2, 8))
        expert_indices = torch.zeros(2, 8, 1, dtype=torch.long)
        tracker.update(token_ids, expert_indices)
        with pytest.raises(AttributeError, match="'float' object has no attribute 'item'"):
            tracker.compute_metrics()

    def test_specialization_and_collapse_scores_bounded(self):
        torch.manual_seed(7)
        tracker = GlobalSpecializationTracker(vocab_size=50, num_experts=8)
        token_ids = torch.randint(0, 50, (4, 32))
        expert_indices = torch.randint(0, 8, (4, 32, 2))
        tracker.update(token_ids, expert_indices)
        metrics = tracker.compute_metrics()
        assert 0.0 <= metrics["specialization_score"] <= 1.0
        assert 0.0 <= metrics["collapse_score"] <= 1.0


# ---------------------------------------------------------------------------
# GlobalSpecializationTracker.sync_and_compute — must not modify local state
# ---------------------------------------------------------------------------

class TestSyncAndComputeNonDestructive:
    # sync_and_compute temporarily swaps in synced data, computes, then restores.
    # If it's destructive, the next compute_metrics call sees a partial histogram.

    def test_non_distributed_sync_does_not_modify_state(self):
        tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
        token_ids = torch.randint(0, 100, (2, 8))
        expert_indices = torch.randint(0, 4, (2, 8, 2))
        tracker.update(token_ids, expert_indices)

        counts_before = tracker.usage_counts.clone()
        tokens_before = tracker.total_tokens

        tracker.sync_and_compute(device="cpu", is_distributed=False)

        assert torch.equal(tracker.usage_counts, counts_before)
        assert tracker.total_tokens == tokens_before

    def test_non_distributed_sync_returns_same_as_compute_metrics(self):
        tracker = GlobalSpecializationTracker(vocab_size=50, num_experts=4)
        token_ids = torch.randint(0, 50, (4, 16))
        expert_indices = torch.randint(0, 4, (4, 16, 2))
        tracker.update(token_ids, expert_indices)

        result_sync = tracker.sync_and_compute(device="cpu", is_distributed=False)
        result_direct = tracker.compute_metrics()

        for key in result_direct:
            assert result_sync[key] == pytest.approx(result_direct[key], rel=1e-6), (
                f"Mismatch for key '{key}'"
            )


# ---------------------------------------------------------------------------
# compute_fatigue_stats — presence/absence depending on router type
# ---------------------------------------------------------------------------

class TestFatigueStatsAbsenceAndPresence:
    # compute_fatigue_stats uses hasattr(router, 'fatigue').
    # A wrong attribute name would raise AttributeError on non-metabolic routers.

    def test_fatigue_stats_empty_for_router_without_fatigue(self, spar_router):
        spar_tracker = RouterMetricsTracker(spar_router)
        assert spar_tracker.compute_fatigue_stats() == {}

    def test_fatigue_stats_all_keys_for_metabolic(self, tracker):
        result = tracker.compute_fatigue_stats()
        expected = {"fatigue_mean", "fatigue_std", "fatigue_min", "fatigue_max",
                    "fatigue_per_expert"}
        assert expected.issubset(result.keys())

    def test_fatigue_stats_values_are_finite(self, tracker):
        result = tracker.compute_fatigue_stats()
        for key in ("fatigue_mean", "fatigue_std", "fatigue_min", "fatigue_max"):
            assert math.isfinite(result[key]), f"{key} is not finite at init"

    def test_fatigue_per_expert_is_numpy_array(self, tracker):
        # WandB histogram logging expects numpy, not a tensor
        result = tracker.compute_fatigue_stats()
        assert isinstance(result["fatigue_per_expert"], np.ndarray)
        assert result["fatigue_per_expert"].shape == (4,)


# ---------------------------------------------------------------------------
# compute_usage_distribution — normalization and output types
# ---------------------------------------------------------------------------

class TestUsageDistributionNormalization:
    # usage_distribution must sum to 1 and be numpy (for WandB).
    # Dividing by counts.sum() without epsilon gives NaN when usage is all zeros.

    def test_usage_distribution_sums_to_one(self, tracker):
        indices, weights = _make_uniform_routing()
        result = tracker.compute_usage_distribution(indices, weights)
        assert result["usage_distribution"].sum() == pytest.approx(1.0, abs=1e-5)

    def test_usage_distribution_is_numpy(self, tracker):
        indices, weights = _make_uniform_routing()
        result = tracker.compute_usage_distribution(indices, weights)
        assert isinstance(result["usage_distribution"], np.ndarray)
        assert isinstance(result["usage_counts"], np.ndarray)

    def test_usage_distribution_no_nan_for_zero_usage(self, tracker):
        # All-zero usage → epsilon normalization → all-zero distribution, no NaN
        usage = torch.zeros(4)
        indices = torch.zeros(1, 1, 1, dtype=torch.long)
        result = tracker.compute_usage_distribution(indices, None, usage=usage)
        assert not np.any(np.isnan(result["usage_distribution"]))

    def test_usage_counts_shape_matches_num_experts(self, tracker):
        indices, weights = _make_uniform_routing(num_experts=4)
        result = tracker.compute_usage_distribution(indices, weights)
        assert result["usage_counts"].shape == (4,)


# ---------------------------------------------------------------------------
# Single-expert router (num_experts=1) — boundary condition
# ---------------------------------------------------------------------------

class TestSingleExpertDegenerate:
    # num_experts=1 exercises the N=1 case in the Gini formula: (n+1)/n = 2.
    # If the numerator is also 2, Gini = 0 (correct). A bug gives negative Gini.

    def test_single_expert_gini_is_zero(self):
        cfg = MetabolicRouterConfig(hidden_dim=16, num_experts=1, top_k=1)
        router = MetabolicRouter(cfg)
        single_tracker = RouterMetricsTracker(router)
        indices = torch.zeros(2, 4, 1, dtype=torch.long)
        weights = torch.ones(2, 4, 1)
        assert single_tracker.compute_gini_coefficient(indices, weights) == pytest.approx(0.0, abs=1e-5)

    def test_single_expert_entropy_is_zero(self):
        cfg = MetabolicRouterConfig(hidden_dim=16, num_experts=1, top_k=1)
        router = MetabolicRouter(cfg)
        single_tracker = RouterMetricsTracker(router)
        indices = torch.zeros(2, 4, 1, dtype=torch.long)
        weights = torch.ones(2, 4, 1)
        result = single_tracker.compute_expert_entropy(indices, weights)
        assert result["expert_entropy"] == pytest.approx(0.0, abs=1e-5)

    def test_single_expert_effective_experts_is_one(self):
        cfg = MetabolicRouterConfig(hidden_dim=16, num_experts=1, top_k=1)
        router = MetabolicRouter(cfg)
        single_tracker = RouterMetricsTracker(router)
        indices = torch.zeros(2, 4, 1, dtype=torch.long)
        weights = torch.ones(2, 4, 1)
        assert single_tracker.compute_effective_experts(indices, weights) == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# compute_metrics — empty state before any update
# ---------------------------------------------------------------------------

class TestComputeMetricsEmptyState:
    # Rank 0 might call compute_metrics at step 0 before any data arrives.
    # Must return {} cleanly — an exception here would crash training.

    def test_empty_tracker_returns_empty_dict(self):
        tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
        assert tracker.compute_metrics() == {}

    def test_empty_tracker_no_crash(self):
        tracker = GlobalSpecializationTracker(vocab_size=1000, num_experts=8)
        try:
            tracker.compute_metrics()
        except Exception as e:
            pytest.fail(f"compute_metrics raised {type(e).__name__} on empty tracker: {e}")

    def test_tracker_after_all_padding_returns_empty(self):
        # All padding → total_tokens stays 0 → same as never calling update
        tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
        token_ids = torch.full((4, 16), -1, dtype=torch.long)
        expert_indices = torch.randint(0, 4, (4, 16, 2))
        tracker.update(token_ids, expert_indices)
        assert tracker.compute_metrics() == {}


# ---------------------------------------------------------------------------
# Gini index cache — device consistency
# ---------------------------------------------------------------------------

class TestGiniIndexDeviceCache:
    # The gini_index tensor is cached on first call to avoid re-allocating each time.
    # A stale or missing cache would cause device mismatch errors on GPU.

    def test_gini_index_cache_populated_after_first_call(self, tracker):
        indices, weights = _make_uniform_routing()
        tracker.compute_gini_coefficient(indices, weights)
        assert hasattr(tracker, "_gini_index_cache")
        assert hasattr(tracker, "_gini_index_device")

    def test_gini_index_cache_correct_values(self, tracker):
        # Should be [1, 2, 3, 4] for num_experts=4
        indices, weights = _make_uniform_routing(num_experts=4)
        tracker.compute_gini_coefficient(indices, weights)
        expected = torch.arange(1, 5, dtype=torch.float32)
        assert torch.allclose(tracker._gini_index_cache.cpu(), expected)

    def test_gini_result_consistent_across_calls(self, tracker):
        # Cache must not corrupt state between calls
        indices, weights = _make_uniform_routing()
        gini1 = tracker.compute_gini_coefficient(indices, weights)
        gini2 = tracker.compute_gini_coefficient(indices, weights)
        assert gini1 == pytest.approx(gini2, abs=1e-8)


# ---------------------------------------------------------------------------
# global_tokens_seen — counts tokens, not token×top_k assignments
# ---------------------------------------------------------------------------

class TestGlobalTokensSeen:
    # global_tokens_seen is used to gate "have we seen enough data yet".
    # Counting top_k assignments instead of tokens would make it fire too early.

    def test_global_tokens_seen_matches_total_tokens(self):
        tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
        token_ids = torch.randint(0, 100, (3, 10))  # 30 tokens
        expert_indices = torch.randint(0, 4, (3, 10, 2))  # top_k=2
        tracker.update(token_ids, expert_indices)
        metrics = tracker.compute_metrics()
        # Should be 30, not 60 (30 tokens × top_k=2)
        assert metrics["global_tokens_seen"] == pytest.approx(30.0, abs=1e-6)

    def test_global_tokens_seen_accumulates_across_updates(self):
        tracker = GlobalSpecializationTracker(vocab_size=100, num_experts=4)
        token_ids = torch.randint(0, 100, (2, 8))
        expert_indices = torch.randint(0, 4, (2, 8, 2))
        tracker.update(token_ids, expert_indices)
        tracker.update(token_ids, expert_indices)
        metrics = tracker.compute_metrics()
        assert metrics["global_tokens_seen"] == pytest.approx(32.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Stress test — large batch and GPT-2 vocab scale
# ---------------------------------------------------------------------------

class TestLargeBatchStability:
    # Numerical issues are more likely to surface at scale.
    # Also checks that vocab_size × num_experts doesn't overflow in bincount.

    def test_large_batch_no_nan_in_all_metrics(self):
        cfg = MetabolicRouterConfig(hidden_dim=64, num_experts=8, top_k=2)
        router = MetabolicRouter(cfg)
        large_tracker = RouterMetricsTracker(router)

        torch.manual_seed(99)
        indices = torch.randint(0, 8, (8, 64, 2))
        weights = torch.rand(8, 64, 2)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        metrics = large_tracker.compute_all_metrics(indices, weights)
        for key, val in metrics.items():
            if isinstance(val, float):
                assert not math.isnan(val), f"NaN in '{key}'"
                assert not math.isinf(val), f"Inf in '{key}'"

    def test_global_tracker_large_vocab_no_crash(self):
        # GPT-2 vocab (50257) × 8 experts — realistic scale
        tracker = GlobalSpecializationTracker(vocab_size=50257, num_experts=8)
        token_ids = torch.randint(0, 50257, (4, 128))
        expert_indices = torch.randint(0, 8, (4, 128, 2))
        try:
            tracker.update(token_ids, expert_indices)
            metrics = tracker.compute_metrics()
        except Exception as e:
            pytest.fail(f"Raised {type(e).__name__} at GPT-2 scale: {e}")
        assert "specialization_score" in metrics
