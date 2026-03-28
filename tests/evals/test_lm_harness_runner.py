from pathlib import Path
from evals.lm_harness_runner import run_lm_harness_eval


class _DummyBackbone:
    def __init__(self):
        self.device = "cpu"


class _DummyModel:
    def __init__(self):
        self.backbone = _DummyBackbone()

    def parameters(self):
        yield type("Param", (), {"dtype": "torch.float32"})()


def test_run_lm_harness_eval_merges_zero_and_five_shot_results(monkeypatch, tmp_path):
    calls = []
    build_calls = []
    monkeypatch.setattr(
        "evals.lm_harness_runner.load_model_for_eval",
        lambda **kwargs: (_DummyModel(), {"step": 42, "metrics": {}}),
    )
    monkeypatch.setattr(
        "evals.lm_harness_runner._load_tokenizer_for_model",
        lambda config: object(),
    )

    def fake_build_harness_model(model, tokenizer, device, batch_size):
        build_calls.append(batch_size)
        return f"wrapped_model_{batch_size}"

    monkeypatch.setattr(
        "evals.lm_harness_runner._build_harness_model",
        fake_build_harness_model,
    )

    def fake_simple_evaluate(**kwargs):
        calls.append(kwargs)
        task = kwargs["tasks"][0]
        if task == "mmlu":
            return {
                "results": {
                    "mmlu": {"acc,none": 0.55},
                    "mmlu_abstract_algebra": {"acc,none": 0.40},
                    "mmlu_anatomy": {"acc,none": 0.70},
                }
            }
        return {
            "results": {
                "hellaswag": {"acc_norm,none": 0.31},
                "piqa": {"acc,none": 0.62},
                "winogrande": {"acc,none": 0.58},
                "arc_easy": {"acc_norm,none": 0.71},
                "arc_challenge": {"acc_norm,none": 0.39},
            }
        }

    monkeypatch.setattr(
        "evals.lm_harness_runner._simple_evaluate", fake_simple_evaluate
    )
    payload = run_lm_harness_eval(
        config={"experiment_name": "demo"},
        checkpoint_path=tmp_path / "checkpoint_step_42.pt",
        output_path=tmp_path / "lm_harness.json",
        device="cpu",
        batch_size={"zero_shot": 2, "five_shot": 1},
        limit=10,
    )
    assert payload["task"] == "lm_harness"
    assert payload["results"] == {
        "hellaswag": 0.31,
        "piqa": 0.62,
        "winogrande": 0.58,
        "arc_easy": 0.71,
        "arc_challenge": 0.39,
        "mmlu": 0.55,
    }
    assert payload["metadata"]["mmlu_subjects"] == {
        "mmlu_abstract_algebra": 0.40,
        "mmlu_anatomy": 0.70,
    }
    assert payload["metadata"]["batch_size"] == {"zero_shot": 2, "five_shot": 1}
    assert build_calls == [2, 1]
    assert calls[0]["num_fewshot"] == 0
    assert calls[1]["num_fewshot"] == 5
    assert calls[0]["model"] == "wrapped_model_2"
    assert calls[0]["batch_size"] == 2
    assert calls[1]["model"] == "wrapped_model_1"
    assert calls[1]["batch_size"] == 1
    assert Path(tmp_path / "lm_harness.json").exists()


def test_run_lm_harness_eval_allows_empty_five_shot_tasks(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "evals.lm_harness_runner.load_model_for_eval",
        lambda **kwargs: (_DummyModel(), {"step": 42, "metrics": {}}),
    )
    monkeypatch.setattr(
        "evals.lm_harness_runner._load_tokenizer_for_model",
        lambda config: object(),
    )
    monkeypatch.setattr(
        "evals.lm_harness_runner._build_harness_model",
        lambda model, tokenizer, device, batch_size: "wrapped_model",
    )

    def fake_simple_evaluate(**kwargs):
        calls.append(kwargs)
        return {"results": {"piqa": {"acc,none": 0.62}}}

    monkeypatch.setattr(
        "evals.lm_harness_runner._simple_evaluate", fake_simple_evaluate
    )
    payload = run_lm_harness_eval(
        config={"experiment_name": "demo"},
        checkpoint_path=tmp_path / "checkpoint_step_42.pt",
        output_path=tmp_path / "lm_harness_smoke.json",
        device="cpu",
        batch_size=1,
        limit=1,
        zero_shot_tasks=("piqa",),
        five_shot_tasks=(),
    )
    assert payload["results"] == {"piqa": 0.62}
    assert payload["metadata"]["mmlu_subjects"] == {}
    assert len(calls) == 1
    assert calls[0]["tasks"] == ["piqa"]
    assert Path(tmp_path / "lm_harness_smoke.json").exists()
