from __future__ import annotations

from collections import defaultdict
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from evals.routing_analysis import (
    _RoutingHookState,
    analyze_expert_token_distributions,
    compute_specialization_score,
)


class _StubTokenizer:
    model_max_length = 128

    def __call__(
        self,
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        add_special_tokens=False,
    ):
        ids = [ord(c) % 50 for c in text[:max_length]]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(tid % 128) for tid in token_ids)


def _make_token_log(
    layer_expert_tokens: Dict[int, Dict[int, List[int]]],
) -> Dict[int, Dict[int, List[int]]]:
    result = defaultdict(lambda: defaultdict(list))
    for layer, experts in layer_expert_tokens.items():
        for expert, tokens in experts.items():
            result[layer][expert].extend(tokens)
    return result


class TestRoutingHookState:
    def test_initial_state_is_empty(self):
        state = _RoutingHookState()
        assert state.token_log == {}
        assert state._handles == []

    def test_clear_resets_token_log(self):
        state = _RoutingHookState()
        state.token_log[0][1].extend([10, 20, 30])
        state.clear()
        assert state.token_log[0][1] == []

    def test_remove_hooks_clears_handles(self):
        state = _RoutingHookState()
        mock_handle = MagicMock()
        state._handles.append(mock_handle)
        state.remove_hooks()
        mock_handle.remove.assert_called_once()
        assert state._handles == []

    def test_remove_hooks_idempotent_on_empty(self):
        state = _RoutingHookState()
        state.remove_hooks()
        assert state._handles == []


class TestAnalyzeExpertTokenDistributions:
    def setup_method(self):
        self.tokenizer = _StubTokenizer()

    def test_returns_correct_structure(self):
        token_log = _make_token_log({10: {0: [1, 2, 3], 1: [4, 5, 6, 7]}})
        result = analyze_expert_token_distributions(token_log, self.tokenizer, top_n=5)
        assert 10 in result
        assert 0 in result[10]
        assert 1 in result[10]

    def test_total_tokens_correct(self):
        token_log = _make_token_log({10: {0: [1, 1, 1, 2, 2], 1: [3, 4]}})
        result = analyze_expert_token_distributions(token_log, self.tokenizer)
        assert result[10][0]["total_tokens"] == 5
        assert result[10][1]["total_tokens"] == 2

    def test_unique_tokens_correct(self):
        token_log = _make_token_log({10: {0: [1, 1, 1, 2, 2]}})
        result = analyze_expert_token_distributions(token_log, self.tokenizer)
        assert result[10][0]["unique_tokens"] == 2

    def test_type_token_ratio_uniform(self):
        token_log = _make_token_log({10: {0: [1, 2, 3, 4, 5]}})
        result = analyze_expert_token_distributions(token_log, self.tokenizer)
        assert result[10][0]["type_token_ratio"] == pytest.approx(1.0, abs=1e-4)

    def test_type_token_ratio_single_token(self):
        token_log = _make_token_log({10: {0: [7, 7, 7, 7]}})
        result = analyze_expert_token_distributions(token_log, self.tokenizer)
        assert result[10][0]["type_token_ratio"] == pytest.approx(0.25, abs=1e-4)

    def test_top_tokens_sorted_by_frequency(self):
        token_log = _make_token_log({10: {0: [5, 5, 5, 5, 3, 3, 1]}})
        result = analyze_expert_token_distributions(token_log, self.tokenizer, top_n=3)
        top = result[10][0]["top_tokens"]
        assert top[0]["token_id"] == 5
        assert top[0]["count"] == 4
        assert top[1]["token_id"] == 3
        assert top[1]["count"] == 2

    def test_top_n_respected(self):
        token_log = _make_token_log({10: {0: list(range(20))}})
        result = analyze_expert_token_distributions(token_log, self.tokenizer, top_n=5)
        assert len(result[10][0]["top_tokens"]) == 5

    def test_empty_expert_handled(self):
        token_log = _make_token_log({10: {0: []}})
        result = analyze_expert_token_distributions(token_log, self.tokenizer)
        stats = result[10][0]
        assert stats["total_tokens"] == 0
        assert stats["unique_tokens"] == 0
        assert stats["type_token_ratio"] == 0.0
        assert stats["top_tokens"] == []

    def test_token_freq_sums_to_one(self):
        token_log = _make_token_log({10: {0: [1, 1, 2, 3]}})
        result = analyze_expert_token_distributions(token_log, self.tokenizer)
        total_freq = sum(t["freq"] for t in result[10][0]["top_tokens"])
        assert total_freq == pytest.approx(1.0, abs=1e-4)

    def test_multiple_layers_independent(self):
        token_log = _make_token_log({2: {0: [1, 1, 1]}, 10: {0: [2, 2]}})
        result = analyze_expert_token_distributions(token_log, self.tokenizer)
        assert result[2][0]["total_tokens"] == 3
        assert result[10][0]["total_tokens"] == 2


class TestComputeSpecializationScore:
    def test_high_ttr_gives_low_score(self):
        assert compute_specialization_score({"type_token_ratio": 1.0}) == pytest.approx(
            0.0, abs=1e-4
        )

    def test_low_ttr_gives_high_score(self):
        assert compute_specialization_score({"type_token_ratio": 0.1}) == pytest.approx(
            0.9, abs=1e-4
        )

    def test_mid_ttr(self):
        assert compute_specialization_score({"type_token_ratio": 0.4}) == pytest.approx(
            0.6, abs=1e-4
        )

    def test_missing_ttr_defaults_to_zero_score(self):
        assert compute_specialization_score({}) == pytest.approx(0.0, abs=1e-4)

    def test_score_is_complement_of_ttr(self):
        for ttr in [0.0, 0.25, 0.5, 0.75, 1.0]:
            score = compute_specialization_score({"type_token_ratio": ttr})
            assert score == pytest.approx(1.0 - ttr, abs=1e-4)


class TestInverseCorrelationProperty:
    def test_specialist_has_higher_score_than_generalist(self):
        tokenizer = _StubTokenizer()
        specialist_log = _make_token_log({10: {0: [7] * 100}})
        generalist_log = _make_token_log({10: {1: list(range(50)) * 2}})

        specialist_stats = analyze_expert_token_distributions(
            specialist_log, tokenizer
        )[10][0]
        generalist_stats = analyze_expert_token_distributions(
            generalist_log, tokenizer
        )[10][1]

        assert compute_specialization_score(
            specialist_stats
        ) > compute_specialization_score(generalist_stats)

    def test_specialization_monotone_with_repetition(self):
        tokenizer = _StubTokenizer()
        scores = []
        for n_unique in [1, 5, 10, 20, 50]:
            tokens = (list(range(n_unique)) * (100 // n_unique))[:100]
            log = _make_token_log({10: {0: tokens}})
            stats = analyze_expert_token_distributions(log, tokenizer)[10][0]
            scores.append(compute_specialization_score(stats))

        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score not monotone: scores[{i}]={scores[i]:.4f} < scores[{i + 1}]={scores[i + 1]:.4f}"
            )


class _StubRouter(nn.Module):
    def __init__(self, num_experts: int = 4, top_k: int = 1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, hidden):
        B, T, _ = hidden.shape
        indices = torch.arange(T, device=hidden.device).unsqueeze(0).expand(B, -1)
        indices = (indices % self.num_experts).unsqueeze(-1)  # [B, T, 1]
        weights = torch.ones(B, T, self.top_k, device=hidden.device)
        return weights, indices


class _StubLoRAMoELayer(nn.Module):
    def __init__(self, num_experts: int = 4):
        super().__init__()
        self.router = _StubRouter(num_experts=num_experts)
        self._last_routing_weights = None

    def forward(self, hidden):
        # Replicate what LoRAMoELayer does: set _last_routing_weights before returning
        # so the routing hook can read dispatch without re-running the router.
        B, T, _ = hidden.shape
        _, indices = self.router(hidden)  # indices: [B, T, 1]
        weights = torch.zeros(B * T, self.router.num_experts, device=hidden.device)
        flat_indices = indices.view(B * T, -1)
        weights.scatter_(1, flat_indices, 1.0)
        self._last_routing_weights = weights.detach()
        return hidden


class _StubMoEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.ModuleDict(
            {
                "h": nn.ModuleList(
                    [
                        nn.Identity(),
                        nn.Identity(),
                        _StubLoRAMoELayer(num_experts=4),  # layer 2
                    ]
                )
            }
        )

    def forward(self, input_ids):
        hidden = torch.zeros(*input_ids.shape, 16)
        self.backbone["h"][2](hidden)
        return torch.zeros(*input_ids.shape, 50), None


class TestRunRoutingAnalysisIntegration:
    def _make_config(self):
        return {"model": {"model_key": "gpt-neo-125m"}}

    @patch("evals.routing_analysis.load_model_for_eval")
    @patch("evals.routing_analysis.model_lookup")
    @patch("evals.routing_analysis.AutoTokenizer")
    def test_returns_payload_with_expected_keys(
        self, mock_auto_tok, mock_model_lookup, mock_load
    ):
        from evals.routing_analysis import run_routing_analysis

        stub_model = _StubMoEModel()
        stub_model.eval()
        mock_load.return_value = (stub_model, {"step": 100})
        mock_model_lookup.return_value = {"hf_name": "EleutherAI/gpt-neo-125m"}
        mock_auto_tok.from_pretrained.return_value = _StubTokenizer()

        payload = run_routing_analysis(
            config=self._make_config(),
            checkpoint_path="/fake/ckpt.pt",
            model=stub_model,
            checkpoint_info={"step": 100},
            texts=["hello world foo bar baz"] * 5,
            n_samples=5,
            max_length=16,
            top_n_tokens=10,
            device="cpu",
        )

        assert payload["task"] == "routing_analysis"
        assert "results" in payload
        assert "summary" in payload["metadata"]
        assert "full_analysis" in payload["metadata"]

    @patch("evals.routing_analysis.load_model_for_eval")
    @patch("evals.routing_analysis.model_lookup")
    @patch("evals.routing_analysis.AutoTokenizer")
    def test_samples_processed_matches_input(
        self, mock_auto_tok, mock_model_lookup, mock_load
    ):
        from evals.routing_analysis import run_routing_analysis

        stub_model = _StubMoEModel()
        stub_model.eval()
        mock_load.return_value = (stub_model, {})
        mock_model_lookup.return_value = {"hf_name": "EleutherAI/gpt-neo-125m"}
        mock_auto_tok.from_pretrained.return_value = _StubTokenizer()

        payload = run_routing_analysis(
            config=self._make_config(),
            checkpoint_path="/fake/ckpt.pt",
            model=stub_model,
            checkpoint_info={},
            texts=["abc def ghi"] * 8,
            device="cpu",
        )

        assert payload["results"]["samples_processed"] == pytest.approx(8.0)

    @patch("evals.routing_analysis.load_model_for_eval")
    @patch("evals.routing_analysis.model_lookup")
    @patch("evals.routing_analysis.AutoTokenizer")
    def test_empty_texts_skipped(self, mock_auto_tok, mock_model_lookup, mock_load):
        from evals.routing_analysis import run_routing_analysis

        stub_model = _StubMoEModel()
        stub_model.eval()
        mock_load.return_value = (stub_model, {})
        mock_model_lookup.return_value = {"hf_name": "EleutherAI/gpt-neo-125m"}
        mock_auto_tok.from_pretrained.return_value = _StubTokenizer()

        payload = run_routing_analysis(
            config=self._make_config(),
            checkpoint_path="/fake/ckpt.pt",
            model=stub_model,
            checkpoint_info={},
            texts=["hello world", "", "   ", "foo bar"],
            device="cpu",
        )

        assert payload["results"]["samples_processed"] == pytest.approx(2.0)

    @patch("evals.routing_analysis.load_model_for_eval")
    @patch("evals.routing_analysis.model_lookup")
    @patch("evals.routing_analysis.AutoTokenizer")
    def test_summary_rows_have_required_fields(
        self, mock_auto_tok, mock_model_lookup, mock_load
    ):
        from evals.routing_analysis import run_routing_analysis

        stub_model = _StubMoEModel()
        stub_model.eval()
        mock_load.return_value = (stub_model, {})
        mock_model_lookup.return_value = {"hf_name": "EleutherAI/gpt-neo-125m"}
        mock_auto_tok.from_pretrained.return_value = _StubTokenizer()

        payload = run_routing_analysis(
            config=self._make_config(),
            checkpoint_path="/fake/ckpt.pt",
            model=stub_model,
            checkpoint_info={},
            texts=["hello world test"] * 3,
            device="cpu",
        )

        required = {
            "layer",
            "expert",
            "total_tokens",
            "unique_tokens",
            "type_token_ratio",
            "specialization_score",
            "top_5_tokens",
        }
        for row in payload["metadata"]["summary"]:
            assert required.issubset(row.keys()), (
                f"Missing keys: {required - row.keys()}"
            )

    @patch("evals.routing_analysis.load_model_for_eval")
    @patch("evals.routing_analysis.model_lookup")
    @patch("evals.routing_analysis.AutoTokenizer")
    def test_specialization_score_in_valid_range(
        self, mock_auto_tok, mock_model_lookup, mock_load
    ):
        from evals.routing_analysis import run_routing_analysis

        stub_model = _StubMoEModel()
        stub_model.eval()
        mock_load.return_value = (stub_model, {})
        mock_model_lookup.return_value = {"hf_name": "EleutherAI/gpt-neo-125m"}
        mock_auto_tok.from_pretrained.return_value = _StubTokenizer()

        payload = run_routing_analysis(
            config=self._make_config(),
            checkpoint_path="/fake/ckpt.pt",
            model=stub_model,
            checkpoint_info={},
            texts=["the quick brown fox jumps"] * 10,
            device="cpu",
        )

        for row in payload["metadata"]["summary"]:
            score = row["specialization_score"]
            assert 0.0 <= score <= 1.0, (
                f"Specialization score {score} out of [0,1] for "
                f"layer={row['layer']} expert={row['expert']}"
            )
