import math

from evals.efficiency import run_efficiency_eval, summarize_timing_measurements


class _DummyBackbone:
    def __init__(self):
        self.config = type("Config", (), {"vocab_size": 32})()


class _DummyModel:
    def __init__(self):
        self.backbone = _DummyBackbone()
        self.vocab_size = 32


def test_summarize_timing_measurements_computes_expected_stats():
    summary = summarize_timing_measurements(
        [1.0, 2.0],
        batch_size=2,
        seq_len=4,
    )

    assert math.isclose(summary["throughput_tokens_per_sec_mean"], 6.0, rel_tol=1e-6)
    assert math.isclose(summary["throughput_tokens_per_sec_std"], 2.0, rel_tol=1e-6)
    assert math.isclose(summary["latency_ms_per_token_p50"], 187.5, rel_tol=1e-6)
    assert math.isclose(summary["latency_ms_per_token_p95"], 250.0, rel_tol=1e-6)


def test_run_efficiency_eval_adds_reference_ratio(monkeypatch, tmp_path):
    call_log = []

    monkeypatch.setattr(
        "evals.efficiency.load_model_for_eval",
        lambda **kwargs: (_DummyModel(), {"step": 10, "metrics": {}}),
    )

    def fake_profile(model, **kwargs):
        call_log.append(kwargs)
        if len(call_log) == 1:
            return {
                "batch_1": {
                    "throughput_tokens_per_sec_mean": 100.0,
                    "throughput_tokens_per_sec_std": 5.0,
                    "latency_ms_per_token_p50": 10.0,
                    "latency_ms_per_token_p95": 12.0,
                },
                "batch_32": {
                    "throughput_tokens_per_sec_mean": 1200.0,
                    "throughput_tokens_per_sec_std": 20.0,
                    "latency_ms_per_token_p50": 0.9,
                    "latency_ms_per_token_p95": 1.0,
                },
                "peak_memory_bytes": 123456,
            }
        return {
            "batch_1": {
                "throughput_tokens_per_sec_mean": 110.0,
                "throughput_tokens_per_sec_std": 3.0,
                "latency_ms_per_token_p50": 8.0,
                "latency_ms_per_token_p95": 9.0,
            },
            "batch_32": {
                "throughput_tokens_per_sec_mean": 1400.0,
                "throughput_tokens_per_sec_std": 30.0,
                "latency_ms_per_token_p50": 0.6,
                "latency_ms_per_token_p95": 0.7,
            },
            "peak_memory_bytes": 111111,
        }

    monkeypatch.setattr("evals.efficiency._profile_loaded_model", fake_profile)

    payload = run_efficiency_eval(
        config={"experiment_name": "demo"},
        checkpoint_path=tmp_path / "checkpoint_step_10.pt",
        output_path=tmp_path / "efficiency.json",
        device="cpu",
        reference_checkpoint_path=tmp_path / "checkpoint_step_20.pt",
    )

    assert payload["task"] == "efficiency"
    assert payload["results"]["batch_1_throughput_tokens_per_sec_mean"] == 100.0
    assert math.isclose(payload["results"]["router_overhead_ratio_batch_1"], 1.25, rel_tol=1e-6)
    assert math.isclose(payload["results"]["router_overhead_ratio_batch_32"], 1.5, rel_tol=1e-6)
    assert payload["metadata"]["reference_checkpoint_path"].endswith("checkpoint_step_20.pt")
