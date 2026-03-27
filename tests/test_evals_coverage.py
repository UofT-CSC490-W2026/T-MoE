"""Tests for evals/ coverage gaps — perplexity, efficiency, lm_harness, loading."""
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock
from pathlib import Path


# ── evals/loading ──────────────────────────────────────────────────────────────

def test_cfg_select_dict():
    from evals.loading import _cfg_select
    config = {"model": {"model_key": "gpt-neo-125m"}}
    assert _cfg_select(config, "model.model_key") == "gpt-neo-125m"
    assert _cfg_select(config, "model.missing", "default") == "default"


def test_cfg_select_missing_key():
    from evals.loading import _cfg_select
    config = {"a": {"b": 1}}
    assert _cfg_select(config, "a.c", "fallback") == "fallback"


def test_require_present():
    from evals.loading import _require
    config = {"model": {"model_key": "gpt-neo-125m"}}
    assert _require(config, "model.model_key") == "gpt-neo-125m"


def test_require_missing():
    from evals.loading import _require
    config = {}
    with pytest.raises(ValueError, match="Missing required"):
        _require(config, "model.model_key")


def test_as_list_none():
    from evals.loading import _as_list
    assert _as_list(None) == []


def test_as_list_int():
    from evals.loading import _as_list
    assert _as_list(5) == [5]


def test_as_list_list():
    from evals.loading import _as_list
    assert _as_list([1, 2, 3]) == [1, 2, 3]


def test_as_list_tuple():
    from evals.loading import _as_list
    assert _as_list((1, 2)) == [1, 2]


def test_router_kwargs():
    from evals.loading import _router_kwargs
    config = {
        "router": {
            "type": "standard",
            "num_experts": 4,
            "top_k": 2,
            "temperature": 1.0,
        }
    }
    router_type, num_experts, top_k, kwargs = _router_kwargs(config)
    assert router_type == "standard"
    assert num_experts == 4
    assert top_k == 2


def test_build_model_from_config():
    from evals.loading import build_model_from_config
    config = {
        "model": {"model_key": "gpt-neo-125m", "freeze_backbone": True, "moe_layer_indices": [-1]},
        "router": {"type": "standard", "num_experts": 4, "top_k": 2},
        "expert": {"type": "gpt_neo_lora", "count": 4, "lora": {"rank": 4, "alpha": 8, "dropout": 0.0, "init_scale": 0.01}},
    }
    mock_model = MagicMock()
    mock_model.num_layers = 12
    mock_model.hidden_dim = 768
    mock_mlp = nn.Module()
    mock_mlp.c_fc = nn.Linear(768, 3072)
    mock_mlp.c_proj = nn.Linear(3072, 768)
    mock_model.get_mlp_at.return_value = mock_mlp
    mock_model.inject_moe_layers = MagicMock()

    with patch("evals.loading.ModelRegistry") as MockReg:
        MockReg.get.return_value = MagicMock(return_value=mock_model)
        with patch("evals.loading.model_lookup", return_value={
            "model_type": "gpt_neo", "variant": "125m", "hidden_dim": 768,
            "intermediate_dim": 3072
        }):
            model = build_model_from_config(config, device="cpu")


def test_load_model_for_eval_not_found(tmp_path):
    from evals.loading import load_model_for_eval
    config = {"model": {"model_key": "gpt-neo-125m"}}
    with pytest.raises(FileNotFoundError):
        load_model_for_eval(config, tmp_path / "nonexistent.pt")


# ── evals/perplexity ───────────────────────────────────────────────────────────

def test_cfg_select_perplexity():
    from evals.perplexity import _cfg_select
    config = {"model": {"model_key": "gpt-neo-125m"}}
    assert _cfg_select(config, "model.model_key") == "gpt-neo-125m"
    assert _cfg_select(config, "missing.key", "default") == "default"


def test_dtype_name():
    from evals.perplexity import _dtype_name
    assert _dtype_name(torch.bfloat16) == "bfloat16"
    assert _dtype_name(torch.float32) == "float32"


def test_autocast_context_cpu():
    from evals.perplexity import _autocast_context
    ctx = _autocast_context("cpu", torch.float32)
    with ctx:
        pass  # should not raise


def test_infer_eval_context_length():
    from evals.perplexity import infer_eval_context_length
    model = MagicMock()
    model.backbone.config.max_position_embeddings = 2048
    config = {}
    length = infer_eval_context_length(model, config)
    assert length == 2048


def test_infer_eval_context_length_tokenizer_limit():
    from evals.perplexity import infer_eval_context_length
    model = MagicMock()
    model.backbone.config.max_position_embeddings = 2048
    tokenizer = MagicMock()
    tokenizer.model_max_length = 512
    config = {}
    length = infer_eval_context_length(model, config, tokenizer)
    assert length == 512


def test_infer_eval_context_length_fallback():
    from evals.perplexity import infer_eval_context_length
    model = MagicMock()
    del model.backbone
    config = {}
    length = infer_eval_context_length(model, config)
    assert length >= 2


def test_summarize_language_model_metrics():
    from evals.perplexity import summarize_language_model_metrics
    result = summarize_language_model_metrics(total_nll=100.0, total_tokens=100)
    assert "ppl" in result
    assert result["ppl"] == pytest.approx(2.718, rel=0.01)


def test_summarize_language_model_metrics_with_bpb():
    from evals.perplexity import summarize_language_model_metrics
    result = summarize_language_model_metrics(total_nll=100.0, total_tokens=100, total_bytes=800)
    assert "bpb" in result


def test_summarize_language_model_metrics_zero_tokens():
    from evals.perplexity import summarize_language_model_metrics
    with pytest.raises(ValueError, match="total_tokens"):
        summarize_language_model_metrics(total_nll=100.0, total_tokens=0)


def test_summarize_language_model_metrics_zero_bytes():
    from evals.perplexity import summarize_language_model_metrics
    with pytest.raises(ValueError, match="total_bytes"):
        summarize_language_model_metrics(total_nll=100.0, total_tokens=100, total_bytes=0)


def test_document_windows_short():
    from evals.perplexity import _document_windows
    input_ids = torch.randint(0, 100, (1, 1))  # too short
    windows = list(_document_windows(0, input_ids, stride=512, max_length=1024))
    assert len(windows) == 0


def test_document_windows_normal():
    from evals.perplexity import _document_windows
    input_ids = torch.randint(0, 100, (1, 100))
    windows = list(_document_windows(0, input_ids, stride=50, max_length=100))
    assert len(windows) > 0


def test_compute_document_nll_invalid_shape():
    from evals.perplexity import compute_document_nll
    model = MagicMock()
    with pytest.raises(ValueError, match="shape"):
        compute_document_nll(model, torch.randn(2, 10), stride=5, max_length=10, device="cpu")


def test_compute_document_nll_invalid_stride():
    from evals.perplexity import compute_document_nll
    model = MagicMock()
    with pytest.raises(ValueError, match="stride"):
        compute_document_nll(model, torch.randint(0, 100, (1, 10)), stride=0, max_length=10, device="cpu")


def test_compute_document_nll_invalid_max_length():
    from evals.perplexity import compute_document_nll
    model = MagicMock()
    with pytest.raises(ValueError, match="max_length"):
        compute_document_nll(model, torch.randint(0, 100, (1, 10)), stride=5, max_length=1, device="cpu")


def test_compute_document_nll_short_seq():
    from evals.perplexity import compute_document_nll
    model = MagicMock()
    input_ids = torch.randint(0, 100, (1, 1))
    nll, tokens = compute_document_nll(model, input_ids, stride=5, max_length=10, device="cpu")
    assert nll == 0.0
    assert tokens == 0


def test_compute_document_nll_normal():
    from evals.perplexity import compute_document_nll

    def fake_model(input_ids):
        B, L = input_ids.shape
        return (torch.randn(B, L, 100),)

    input_ids = torch.randint(0, 100, (1, 20))
    nll, tokens = compute_document_nll(fake_model, input_ids, stride=10, max_length=20, device="cpu")
    assert tokens > 0


def test_run_batched_forward_same_length():
    from evals.perplexity import _run_batched_forward, _Window
    mock_model = MagicMock()
    logits = torch.randn(2, 10, 100)
    mock_model.return_value = (logits,)
    windows = [
        _Window(0, torch.randint(0, 100, (1, 10)), torch.ones(9, dtype=torch.bool)),
        _Window(1, torch.randint(0, 100, (1, 10)), torch.ones(9, dtype=torch.bool)),
    ]
    results = _run_batched_forward(mock_model, windows, "cpu", torch.float32)
    assert len(results) == 2


def test_run_batched_forward_different_lengths():
    from evals.perplexity import _run_batched_forward, _Window
    mock_model = MagicMock()
    logits = torch.randn(2, 15, 100)
    mock_model.return_value = MagicMock(logits=logits)
    windows = [
        _Window(0, torch.randint(0, 100, (1, 10)), torch.ones(9, dtype=torch.bool)),
        _Window(1, torch.randint(0, 100, (1, 15)), torch.ones(14, dtype=torch.bool)),
    ]
    results = _run_batched_forward(mock_model, windows, "cpu", torch.float32)
    assert len(results) == 2


# ── evals/efficiency ───────────────────────────────────────────────────────────

def test_dtype_name_efficiency():
    from evals.efficiency import _dtype_name
    assert _dtype_name(torch.float32) == "float32"


def test_autocast_context_efficiency_cpu():
    from evals.efficiency import _autocast_context
    ctx = _autocast_context("cpu", torch.float32)
    with ctx:
        pass


def test_device_synchronize_cpu():
    from evals.efficiency import _device_synchronize
    _device_synchronize("cpu")  # no-op for CPU


def test_summarize_timing_measurements():
    from evals.efficiency import summarize_timing_measurements
    durations = [0.1, 0.12, 0.09, 0.11]
    result = summarize_timing_measurements(durations, batch_size=4, seq_len=512)
    assert "throughput_tokens_per_sec_mean" in result
    assert "latency_ms_per_token_p50" in result
    assert "latency_ms_per_token_p95" in result


def test_summarize_timing_measurements_single():
    from evals.efficiency import summarize_timing_measurements
    result = summarize_timing_measurements([0.1], batch_size=1, seq_len=100)
    assert result["throughput_tokens_per_sec_std"] == 0.0


def test_summarize_timing_measurements_empty():
    from evals.efficiency import summarize_timing_measurements
    with pytest.raises(ValueError, match="empty"):
        summarize_timing_measurements([], batch_size=1, seq_len=100)


def test_compute_overhead_ratios():
    from evals.efficiency import _compute_overhead_ratios
    current = {"batch_1": {"latency_ms_per_token_p50": 2.0}}
    reference = {"batch_1": {"latency_ms_per_token_p50": 1.0}}
    ratios = _compute_overhead_ratios(current, reference, [1])
    assert ratios["router_overhead_ratio_batch_1"] == pytest.approx(2.0)


def test_compute_overhead_ratios_zero_reference():
    from evals.efficiency import _compute_overhead_ratios
    current = {"batch_1": {"latency_ms_per_token_p50": 2.0}}
    reference = {"batch_1": {"latency_ms_per_token_p50": 0.0}}
    ratios = _compute_overhead_ratios(current, reference, [1])
    assert "router_overhead_ratio_batch_1" not in ratios


def test_profile_loaded_model():
    from evals.efficiency import _profile_loaded_model
    mock_model = MagicMock()
    mock_model.vocab_size = 100
    mock_model.return_value = (torch.randn(1, 10, 100),)
    result = _profile_loaded_model(
        mock_model, device="cpu", seq_len=10, batch_sizes=[1],
        warmup_iters=1, benchmark_iters=2, autocast_dtype=torch.float32
    )
    assert "batch_1" in result
    assert result["peak_memory_bytes"] is None  # CPU


def test_run_efficiency_eval():
    from evals.efficiency import run_efficiency_eval
    mock_model = MagicMock()
    mock_model.vocab_size = 100
    mock_model.return_value = (torch.randn(2, 10, 100),)
    mock_payload = {"task": "efficiency", "results": {}}

    with patch("evals.efficiency.build_results_payload", return_value=mock_payload):
        result = run_efficiency_eval(
            config={}, checkpoint_path="ckpt.pt",
            model=mock_model, checkpoint_info={},
            device="cpu", batch_sizes=[1], seq_len=10,
            warmup_iters=1, benchmark_iters=2,
            autocast_dtype=torch.float32
        )
    assert result == mock_payload


def test_run_efficiency_eval_with_output(tmp_path):
    from evals.efficiency import run_efficiency_eval
    mock_model = MagicMock()
    mock_model.vocab_size = 100
    mock_model.return_value = (torch.randn(1, 10, 100),)
    mock_payload = {"task": "efficiency", "results": {}}

    with patch("evals.efficiency.build_results_payload", return_value=mock_payload):
        with patch("evals.efficiency.write_results_json") as mock_write:
            run_efficiency_eval(
                config={}, checkpoint_path="ckpt.pt",
                model=mock_model, checkpoint_info={},
                output_path=str(tmp_path / "out.json"),
                device="cpu", batch_sizes=[1], seq_len=10,
                warmup_iters=1, benchmark_iters=2,
                autocast_dtype=torch.float32
            )
            mock_write.assert_called_once()


# ── evals/lm_harness_runner ────────────────────────────────────────────────────

def test_extract_primary_metric():
    from evals.lm_harness_runner import _extract_primary_metric
    raw = {"results": {"hellaswag": {"acc_norm,none": 0.75}}}
    result = _extract_primary_metric(raw, "hellaswag")
    assert result == pytest.approx(0.75)


def test_extract_primary_metric_fallback_key():
    from evals.lm_harness_runner import _extract_primary_metric
    raw = {"results": {"hellaswag": {"acc_norm": 0.80}}}
    result = _extract_primary_metric(raw, "hellaswag")
    assert result == pytest.approx(0.80)


def test_extract_primary_metric_missing():
    from evals.lm_harness_runner import _extract_primary_metric
    raw = {"results": {"hellaswag": {"unknown_metric": 0.5}}}
    with pytest.raises(KeyError):
        _extract_primary_metric(raw, "hellaswag")


def test_collect_mmlu_breakdown():
    from evals.lm_harness_runner import _collect_mmlu_breakdown
    raw = {
        "results": {
            "mmlu_math": {"acc,none": 0.6},
            "mmlu_history": {"acc": 0.7},
            "mmlu": {"acc,none": 0.65},  # should be excluded
            "hellaswag": {"acc_norm,none": 0.8},  # should be excluded
        }
    }
    breakdown = _collect_mmlu_breakdown(raw)
    assert "mmlu_math" in breakdown
    assert "mmlu_history" in breakdown
    assert "mmlu" not in breakdown
    assert "hellaswag" not in breakdown


def test_resolve_batch_sizes_scalar():
    from evals.lm_harness_runner import _resolve_batch_sizes
    zs, fs = _resolve_batch_sizes(4)
    assert zs == 4
    assert fs == 4


def test_resolve_batch_sizes_mapping():
    from evals.lm_harness_runner import _resolve_batch_sizes
    zs, fs = _resolve_batch_sizes({"zero_shot": 8, "five_shot": 2})
    assert zs == 8
    assert fs == 2


def test_resolve_batch_sizes_mapping_default():
    from evals.lm_harness_runner import _resolve_batch_sizes
    zs, fs = _resolve_batch_sizes({"default": 4})
    assert zs == 4
    assert fs == 4


def test_run_lm_harness_eval():
    from evals.lm_harness_runner import run_lm_harness_eval
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([MagicMock(dtype=torch.float32)])
    mock_model.backbone = MagicMock()

    mock_tokenizer = MagicMock()
    mock_harness_model = MagicMock()
    mock_eval_result = {"results": {
        "hellaswag": {"acc_norm,none": 0.75},
        "piqa": {"acc,none": 0.80},
        "winogrande": {"acc,none": 0.70},
        "arc_easy": {"acc_norm,none": 0.65},
        "arc_challenge": {"acc_norm,none": 0.55},
        "mmlu": {"acc,none": 0.60},
    }}

    with patch("evals.lm_harness_runner._load_tokenizer_for_model", return_value=mock_tokenizer):
        with patch("evals.lm_harness_runner._build_harness_model", return_value=mock_harness_model):
            with patch("evals.lm_harness_runner._simple_evaluate", return_value=mock_eval_result):
                with patch("evals.lm_harness_runner.build_results_payload", return_value={"task": "lm_harness"}):
                    result = run_lm_harness_eval(
                        config={}, checkpoint_path="ckpt.pt",
                        model=mock_model, checkpoint_info={},
                        device="cpu"
                    )
    assert result["task"] == "lm_harness"


def test_run_lm_harness_eval_different_batch_sizes():
    from evals.lm_harness_runner import run_lm_harness_eval
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([MagicMock(dtype=torch.float32)])
    mock_model.backbone = MagicMock()
    mock_tokenizer = MagicMock()
    mock_harness_model = MagicMock()
    mock_eval_result = {"results": {
        "hellaswag": {"acc_norm,none": 0.75},
        "piqa": {"acc,none": 0.80},
        "winogrande": {"acc,none": 0.70},
        "arc_easy": {"acc_norm,none": 0.65},
        "arc_challenge": {"acc_norm,none": 0.55},
        "mmlu": {"acc,none": 0.60},
    }}

    with patch("evals.lm_harness_runner._load_tokenizer_for_model", return_value=mock_tokenizer):
        with patch("evals.lm_harness_runner._build_harness_model", return_value=mock_harness_model):
            with patch("evals.lm_harness_runner._simple_evaluate", return_value=mock_eval_result):
                with patch("evals.lm_harness_runner.build_results_payload", return_value={"task": "lm_harness"}):
                    result = run_lm_harness_eval(
                        config={}, checkpoint_path="ckpt.pt",
                        model=mock_model, checkpoint_info={},
                        device="cpu",
                        batch_size={"zero_shot": 8, "five_shot": 2}
                    )


def test_run_lm_harness_eval_no_tasks():
    from evals.lm_harness_runner import run_lm_harness_eval
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([MagicMock(dtype=torch.float32)])
    mock_tokenizer = MagicMock()

    with patch("evals.lm_harness_runner._load_tokenizer_for_model", return_value=mock_tokenizer):
        with patch("evals.lm_harness_runner.build_results_payload", return_value={"task": "lm_harness"}):
            result = run_lm_harness_eval(
                config={}, checkpoint_path="ckpt.pt",
                model=mock_model, checkpoint_info={},
                device="cpu",
                zero_shot_tasks=[],
                five_shot_tasks=[]
            )


def test_run_lm_harness_eval_with_output(tmp_path):
    from evals.lm_harness_runner import run_lm_harness_eval
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([MagicMock(dtype=torch.float32)])
    mock_tokenizer = MagicMock()

    with patch("evals.lm_harness_runner._load_tokenizer_for_model", return_value=mock_tokenizer):
        with patch("evals.lm_harness_runner.build_results_payload", return_value={"task": "lm_harness"}):
            with patch("evals.lm_harness_runner.write_results_json") as mock_write:
                run_lm_harness_eval(
                    config={}, checkpoint_path="ckpt.pt",
                    model=mock_model, checkpoint_info={},
                    output_path=str(tmp_path / "out.json"),
                    device="cpu",
                    zero_shot_tasks=[],
                    five_shot_tasks=[]
                )
                mock_write.assert_called_once()


# ── evals/results_schema additional branches ──────────────────────────────────

def test_to_plain_python_various_types():
    from evals.results_schema import _to_plain_python
    import numpy as np
    assert _to_plain_python(None) is None
    assert _to_plain_python(True) is True
    assert _to_plain_python(1.5) == 1.5
    assert _to_plain_python("hello") == "hello"
    assert _to_plain_python([1, 2]) == [1, 2]
    assert _to_plain_python({"a": 1}) == {"a": 1}
    assert _to_plain_python(np.float32(1.5)) == pytest.approx(1.5, rel=0.01)
    assert _to_plain_python(torch.tensor(2.0)) == pytest.approx(2.0)
    assert _to_plain_python(Path("/tmp/test")) == "/tmp/test"


def test_get_git_commit():
    from evals.results_schema import get_git_commit
    result = get_git_commit()
    assert isinstance(result, str)


def test_get_git_commit_failure():
    from evals.results_schema import get_git_commit
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = get_git_commit()
    assert result == "unknown"


def test_infer_checkpoint_step_from_info():
    from evals.results_schema import infer_checkpoint_step
    result = infer_checkpoint_step("ckpt.pt", {"step": 42})
    assert result == 42


def test_infer_checkpoint_step_from_name():
    from evals.results_schema import infer_checkpoint_step
    result = infer_checkpoint_step("checkpoint_step_100.pt")
    assert result == 100


def test_infer_checkpoint_step_unknown():
    from evals.results_schema import infer_checkpoint_step
    result = infer_checkpoint_step("best_model.pt")
    assert result is None


def test_flatten_scalars():
    from evals.results_schema import flatten_scalars
    data = {"a": 1.0, "b": {"c": 2.0, "d": "text"}, "e": [1, 2]}
    result = flatten_scalars(data)
    assert result["a"] == 1.0
    assert result["b/c"] == 2.0
    assert result["b/d"] == "text"
    assert "e" not in result  # lists are not scalars


def test_eval_wandb_project_from_config():
    from evals.results_schema import _eval_wandb_project
    config = {"logging": {"project": "my_project"}}
    assert _eval_wandb_project(config) == "my_project"


def test_eval_wandb_project_from_env(monkeypatch):
    from evals.results_schema import _eval_wandb_project
    monkeypatch.setenv("WANDB_PROJECT", "env_project")
    assert _eval_wandb_project({}) == "env_project"


def test_eval_wandb_project_default():
    from evals.results_schema import _eval_wandb_project
    with patch.dict("os.environ", {}, clear=True):
        result = _eval_wandb_project({})
    assert result == "tmoe"


def test_eval_wandb_entity_from_config():
    from evals.results_schema import _eval_wandb_entity
    config = {"logging": {"entity": "my_entity"}}
    assert _eval_wandb_entity(config) == "my_entity"


def test_eval_wandb_entity_none():
    from evals.results_schema import _eval_wandb_entity
    with patch.dict("os.environ", {}, clear=True):
        result = _eval_wandb_entity({})
    assert result is None


def test_eval_wandb_mode_from_config():
    from evals.results_schema import _eval_wandb_mode
    config = {"logging": {"mode": "offline"}}
    assert _eval_wandb_mode(config) == "offline"


def test_eval_wandb_mode_from_env(monkeypatch):
    from evals.results_schema import _eval_wandb_mode
    monkeypatch.setenv("WANDB_MODE", "offline")
    assert _eval_wandb_mode({}) == "offline"


def test_eval_wandb_mode_default():
    from evals.results_schema import _eval_wandb_mode
    with patch.dict("os.environ", {}, clear=True):
        result = _eval_wandb_mode({})
    assert result == "online"


def test_eval_run_name():
    from evals.results_schema import _eval_run_name
    payload = {"experiment_name": "test_exp", "task": "perplexity"}
    result = _eval_run_name(payload, {})
    assert "test_exp" in result
    assert "perplexity" in result


def test_eval_run_id():
    from evals.results_schema import _eval_run_id
    payload = {"experiment_name": "test_exp", "task": "perplexity"}
    result = _eval_run_id(payload, {})
    assert result.startswith("eval-v")


def test_slugify():
    from evals.results_schema import _slugify
    assert _slugify("hello world!") == "hello-world"
    assert _slugify("") == "experiment"


def test_wandb_history_payload():
    from evals.results_schema import _wandb_history_payload
    payload = {"task": "perplexity", "results": {"ppl": 15.2}}
    result = _wandb_history_payload(payload)
    assert "eval/perplexity/ppl" in result


def test_wandb_summary_payload():
    from evals.results_schema import _wandb_summary_payload
    payload = {
        "task": "perplexity",
        "checkpoint_step": 100,
        "git_commit": "abc123",
        "metadata": {"stride": 512},
    }
    result = _wandb_summary_payload(payload)
    assert result["eval/latest_checkpoint_step"] == 100
    assert result["eval/git_commit"] == "abc123"


def test_build_mmlu_table_empty():
    from evals.results_schema import _build_mmlu_table
    payload = {"metadata": {}}
    result = _build_mmlu_table(payload)
    assert result is None


def test_build_mmlu_table_with_data():
    from evals.results_schema import _build_mmlu_table
    with patch("evals.results_schema.WANDB_AVAILABLE", True):
        with patch("evals.results_schema.wandb") as mock_wandb:
            mock_table = MagicMock()
            mock_wandb.Table.return_value = mock_table
            payload = {"metadata": {"mmlu_subjects": {"math": 0.6, "history": 0.7}}}
            result = _build_mmlu_table(payload)
            assert result is mock_table


def test_log_results_to_wandb_not_available():
    from evals.results_schema import log_results_to_wandb
    with patch("evals.results_schema.WANDB_AVAILABLE", False):
        result = log_results_to_wandb({})
    assert result is False


def test_log_results_to_wandb_disabled_config():
    from evals.results_schema import log_results_to_wandb
    with patch("evals.results_schema.WANDB_AVAILABLE", True):
        result = log_results_to_wandb({}, config={"logging": {"enabled": False}})
    assert result is False


def test_log_results_to_wandb_disabled_mode():
    from evals.results_schema import log_results_to_wandb
    with patch("evals.results_schema.WANDB_AVAILABLE", True):
        with patch("evals.results_schema._eval_wandb_mode", return_value="disabled"):
            result = log_results_to_wandb({})
    assert result is False


def test_log_results_to_wandb_init_fails():
    from evals.results_schema import log_results_to_wandb
    with patch("evals.results_schema.WANDB_AVAILABLE", True):
        with patch("evals.results_schema.wandb") as mock_wandb:
            mock_wandb.run = None
            mock_wandb.init.side_effect = Exception("wandb error")
            result = log_results_to_wandb({"task": "perplexity", "results": {}})
    assert result is False


def test_log_results_to_wandb_success():
    from evals.results_schema import log_results_to_wandb
    with patch("evals.results_schema.WANDB_AVAILABLE", True):
        with patch("evals.results_schema.wandb") as mock_wandb:
            mock_run = MagicMock()
            mock_wandb.init.return_value = mock_run
            mock_wandb.run = mock_run
            mock_wandb.Settings = None
            payload = {
                "task": "perplexity",
                "results": {"ppl": 15.2},
                "experiment_name": "test",
                "checkpoint_step": 100,
                "metadata": {},
            }
            result = log_results_to_wandb(payload)
    assert result is True


def test_write_results_json(tmp_path):
    from evals.results_schema import write_results_json
    payload = {"task": "perplexity", "results": {"ppl": 15.2}}
    out = tmp_path / "results.json"
    write_results_json(payload, out)
    assert out.exists()
    import json
    data = json.loads(out.read_text())
    assert data["task"] == "perplexity"


# ── evals/efficiency additional branches ──────────────────────────────────────

def test_run_efficiency_eval_with_reference(tmp_path):
    from evals.efficiency import run_efficiency_eval
    mock_model = MagicMock()
    mock_model.vocab_size = 100
    mock_model.return_value = (torch.randn(1, 10, 100),)
    mock_payload = {"task": "efficiency", "results": {}}

    with patch("evals.efficiency.build_results_payload", return_value=mock_payload):
        with patch("evals.loading.load_model_for_eval", return_value=(mock_model, {})):
            with patch("evals.efficiency.load_model_for_eval", return_value=(mock_model, {})):
                result = run_efficiency_eval(
                    config={}, checkpoint_path="ckpt.pt",
                    model=mock_model, checkpoint_info={},
                    device="cpu", batch_sizes=[1], seq_len=10,
                    warmup_iters=1, benchmark_iters=2,
                    autocast_dtype=torch.float32,
                    reference_checkpoint_path="ref_ckpt.pt",
                    reference_config={},
                )
    assert result == mock_payload


# ── evals/lm_harness additional branches ──────────────────────────────────────

def test_simple_evaluate_wrapper():
    """Test _simple_evaluate sets up logging filters correctly."""
    from evals.lm_harness_runner import _simple_evaluate
    mock_lm_eval = MagicMock()
    mock_lm_eval.evaluator.simple_evaluate.return_value = {"results": {}}
    with patch.dict("sys.modules", {"lm_eval": mock_lm_eval, "lm_eval.evaluator": mock_lm_eval.evaluator}):
        with patch("datasets.disable_progress_bars"):
            with patch("datasets.enable_progress_bars"):
                result = _simple_evaluate(model=MagicMock(), tasks=[], num_fewshot=0,
                                           batch_size=1, device="cpu", limit=None, log_samples=False)
    assert result == {"results": {}}


# ── evals/loading.py missing lines ────────────────────────────────────────────

def test_cfg_select_omegaconf_path():
    """_cfg_select uses OmegaConf.select when config is an OmegaConf object."""
    from evals.loading import _cfg_select
    from omegaconf import OmegaConf
    config = OmegaConf.create({"model": {"model_key": "gpt-neo-125m"}})
    result = _cfg_select(config, "model.model_key")
    assert result == "gpt-neo-125m"


def test_as_list_omegaconf():
    """_as_list handles OmegaConf list values."""
    from evals.loading import _as_list
    from omegaconf import OmegaConf
    value = OmegaConf.create([1, 2, 3])
    result = _as_list(value)
    assert result == [1, 2, 3]


def test_build_moe_layers_expert_count_mismatch():
    """_build_moe_layers raises ValueError when expert.count != router.num_experts."""
    from evals.loading import _build_moe_layers
    config = {
        "model": {"model_key": "gpt-neo-125m", "moe_layer_indices": [-1]},
        "router": {"type": "standard", "num_experts": 4, "top_k": 2},
        "expert": {"type": "gpt_neo_lora", "count": 8},  # mismatch: 8 != 4
    }
    mock_model = MagicMock()
    mock_model.num_layers = 12
    with patch("evals.loading.model_lookup", return_value={
        "model_type": "gpt_neo", "variant": "125m", "hidden_dim": 768,
        "intermediate_dim": 3072
    }):
        with pytest.raises(ValueError, match="expert.count"):
            _build_moe_layers(mock_model, config)


def test_build_moe_layers_invalid_layer_index():
    """_build_moe_layers raises ValueError for out-of-range layer index."""
    from evals.loading import _build_moe_layers
    config = {
        "model": {"model_key": "gpt-neo-125m", "moe_layer_indices": [99]},
        "router": {"type": "standard", "num_experts": 4, "top_k": 2},
        "expert": {"type": "gpt_neo_lora", "count": 4},
    }
    mock_model = MagicMock()
    mock_model.num_layers = 12
    with patch("evals.loading.model_lookup", return_value={
        "model_type": "gpt_neo", "variant": "125m", "hidden_dim": 768,
        "intermediate_dim": 3072
    }):
        with pytest.raises(ValueError, match="Invalid model.moe_layer_indices"):
            _build_moe_layers(mock_model, config)


# ── evals/results_schema.py missing lines ─────────────────────────────────────

def test_to_plain_python_omegaconf():
    """_to_plain_python converts OmegaConf config to plain Python."""
    from evals.results_schema import _to_plain_python
    from omegaconf import OmegaConf
    config = OmegaConf.create({"key": "value", "num": 42})
    result = _to_plain_python(config)
    assert result == {"key": "value", "num": 42}


def test_to_plain_python_path():
    from evals.results_schema import _to_plain_python
    from pathlib import Path
    result = _to_plain_python(Path("/some/path"))
    assert result == "/some/path"


def test_to_plain_python_tensor_tolist():
    """_to_plain_python calls .tolist() on tensors."""
    import torch
    from evals.results_schema import _to_plain_python
    t = torch.tensor([1.0, 2.0, 3.0])
    result = _to_plain_python(t)
    assert result == [1.0, 2.0, 3.0]


def test_to_plain_python_scalar_item():
    """_to_plain_python calls .item() on scalar tensors."""
    import torch
    from evals.results_schema import _to_plain_python
    t = torch.tensor(42.0)
    result = _to_plain_python(t)
    assert result == pytest.approx(42.0)


def test_to_plain_python_fallback_str():
    """_to_plain_python falls back to str() for unknown types."""
    from evals.results_schema import _to_plain_python

    class Weird:
        def __str__(self): return "weird_value"

    result = _to_plain_python(Weird())
    assert result == "weird_value"


def test_infer_checkpoint_step_from_info_v2():
    from evals.results_schema import infer_checkpoint_step
    result = infer_checkpoint_step("checkpoint_step_100.pt", {"step": 200})
    assert result == 200


def test_infer_checkpoint_step_from_name_v2():
    from evals.results_schema import infer_checkpoint_step
    result = infer_checkpoint_step("checkpoint_step_100.pt", None)
    assert result == 100


def test_infer_checkpoint_step_best_model():
    from evals.results_schema import infer_checkpoint_step
    result = infer_checkpoint_step("best_model.pt", None)
    assert result is None


def test_cfg_select_results_schema_omegaconf():
    """_cfg_select in results_schema uses OmegaConf.select for OmegaConf configs."""
    from evals.results_schema import _cfg_select
    from omegaconf import OmegaConf
    config = OmegaConf.create({"logging": {"project": "my_project"}})
    result = _cfg_select(config, "logging.project")
    assert result == "my_project"


def test_cfg_select_results_schema_missing_key():
    """_cfg_select returns default when key is missing in plain dict."""
    from evals.results_schema import _cfg_select
    result = _cfg_select({"a": 1}, "b.c", "fallback")
    assert result == "fallback"


def test_log_results_to_wandb_not_available_v2():
    from evals.results_schema import log_results_to_wandb
    with patch("evals.results_schema.WANDB_AVAILABLE", False):
        result = log_results_to_wandb({"task": "perplexity"})
    assert result is False


def test_log_results_to_wandb_logging_disabled():
    from evals.results_schema import log_results_to_wandb
    with patch("evals.results_schema.WANDB_AVAILABLE", True):
        result = log_results_to_wandb(
            {"task": "perplexity"},
            config={"logging": {"enabled": False}}
        )
    assert result is False


def test_log_results_to_wandb_mode_disabled():
    from evals.results_schema import log_results_to_wandb
    with patch("evals.results_schema.WANDB_AVAILABLE", True):
        result = log_results_to_wandb(
            {"task": "perplexity"},
            config={"logging": {"mode": "disabled"}}
        )
    assert result is False


# ── evals/results_schema.py additional coverage ───────────────────────────────

def test_to_plain_python_tolist_raises():
    """_to_plain_python falls back to .item() when .tolist() raises TypeError."""
    from evals.results_schema import _to_plain_python

    class BadToList:
        def tolist(self):
            raise TypeError("bad")
        def item(self):
            return 99.0

    result = _to_plain_python(BadToList())
    assert result == pytest.approx(99.0)


def test_to_plain_python_item_raises():
    """_to_plain_python falls back to str() when both .tolist() and .item() raise."""
    from evals.results_schema import _to_plain_python

    class BadBoth:
        def tolist(self):
            raise TypeError("bad")
        def item(self):
            raise TypeError("bad")
        def __str__(self):
            return "fallback_str"

    result = _to_plain_python(BadBoth())
    assert result == "fallback_str"


def test_infer_checkpoint_step_value_error():
    """infer_checkpoint_step returns None when step part is not an integer."""
    from evals.results_schema import infer_checkpoint_step
    result = infer_checkpoint_step("checkpoint_step_abc.pt", None)
    assert result is None


def test_log_results_to_wandb_logging_enabled_false_v2():
    """log_results_to_wandb returns False when logging.enabled is explicitly False."""
    from evals.results_schema import log_results_to_wandb
    with patch("evals.results_schema.WANDB_AVAILABLE", True):
        result = log_results_to_wandb(
            {"task": "perplexity"},
            config={"logging": {"enabled": False}}
        )
    assert result is False


def test_log_results_to_wandb_with_checkpoint_step():
    """log_results_to_wandb logs with step when checkpoint_step is present."""
    from evals.results_schema import log_results_to_wandb
    with patch("evals.results_schema.WANDB_AVAILABLE", True):
        with patch("evals.results_schema.wandb") as mock_wandb:
            mock_run = MagicMock()
            mock_wandb.init.return_value = mock_run
            mock_wandb.run = mock_run
            mock_wandb.Settings = None
            payload = {
                "task": "perplexity",
                "checkpoint_step": 100,
                "results": {"ppl": 15.0},
                "metadata": {},
                "experiment_name": "test_exp",
            }
            result = log_results_to_wandb(payload, config={})
    assert result is True


def test_log_results_to_wandb_wandb_init_fails():
    """log_results_to_wandb returns False when wandb.init raises."""
    from evals.results_schema import log_results_to_wandb
    with patch("evals.results_schema.WANDB_AVAILABLE", True):
        with patch("evals.results_schema.wandb") as mock_wandb:
            mock_wandb.init.side_effect = Exception("wandb error")
            mock_wandb.Settings = None
            result = log_results_to_wandb({"task": "perplexity"}, config={})
    assert result is False


def test_log_results_to_wandb_run_none():
    """log_results_to_wandb returns False when wandb.init returns None."""
    from evals.results_schema import log_results_to_wandb
    with patch("evals.results_schema.WANDB_AVAILABLE", True):
        with patch("evals.results_schema.wandb") as mock_wandb:
            mock_wandb.init.return_value = None
            mock_wandb.Settings = None
            result = log_results_to_wandb({"task": "perplexity"}, config={})
    assert result is False


# ── evals/loading.py OmegaConf router config path ────────────────────────────

def test_router_kwargs_omegaconf():
    """_router_kwargs handles OmegaConf router config."""
    from evals.loading import _router_kwargs
    from omegaconf import OmegaConf
    config = OmegaConf.create({
        "router": {
            "type": "standard",
            "num_experts": 4,
            "top_k": 2,
            "temperature": 1.0,
        }
    })
    router_type, num_experts, top_k, kwargs = _router_kwargs(config)
    assert router_type == "standard"
    assert num_experts == 4
    assert top_k == 2
