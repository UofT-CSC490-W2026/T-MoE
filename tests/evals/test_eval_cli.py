from pathlib import Path

import pytest

from scripts.eval import main, run_task


def test_run_task_dispatches_perplexity(monkeypatch, tmp_path):
    captured = {}

    def fake_load_experiment_config(config_path_or_name, overrides=None):
        captured["config_arg"] = config_path_or_name
        captured["overrides"] = overrides
        return {"experiment_name": "demo"}

    def fake_run_perplexity_eval(**kwargs):
        captured["kwargs"] = kwargs
        return {"task": "perplexity", "results": {}}

    monkeypatch.setattr("scripts.eval.load_experiment_config", fake_load_experiment_config)
    monkeypatch.setattr("scripts.eval.run_perplexity_eval", fake_run_perplexity_eval)
    monkeypatch.setattr("scripts.eval.log_results_to_wandb", lambda payload, config: True)

    result = main(
        [
            "--task",
            "perplexity",
            "--checkpoint",
            str(tmp_path / "checkpoint_step_100.pt"),
            "--config",
            "experiments/smoketest.yaml",
            "--output-dir",
            str(tmp_path / "custom_eval"),
            "--device",
            "cpu",
            "--stride",
            "128",
            "--max-documents",
            "4",
            "training.lr=1e-4",
        ]
    )

    assert result["task"] == "perplexity"
    assert captured["config_arg"] == "experiments/smoketest.yaml"
    assert captured["overrides"] == ["training.lr=1e-4"]
    assert captured["kwargs"]["output_path"] == Path(tmp_path / "custom_eval" / "perplexity.json")
    assert captured["kwargs"]["device"] == "cpu"
    assert captured["kwargs"]["stride"] == 128
    assert captured["kwargs"]["max_documents"] == 4


def test_run_task_uses_default_eval_dir(monkeypatch):
    def fake_load_experiment_config(config_path_or_name, overrides=None):
        return {"experiment_name": "demo_exp"}

    captured = {}

    def fake_run_perplexity_eval(**kwargs):
        captured["output_path"] = kwargs["output_path"]
        return {"ok": True}

    monkeypatch.setattr("scripts.eval.load_experiment_config", fake_load_experiment_config)
    monkeypatch.setattr("scripts.eval.run_perplexity_eval", fake_run_perplexity_eval)
    monkeypatch.setattr("scripts.eval.log_results_to_wandb", lambda payload, config: True)

    main(
        [
            "--task",
            "perplexity",
            "--checkpoint",
            "outputs/demo_exp/checkpoint_step_100.pt",
            "--config",
            "demo_exp",
        ]
    )

    assert captured["output_path"] == Path("outputs/demo_exp/eval/perplexity.json")


def test_run_task_rejects_unimplemented_tasks(monkeypatch):
    monkeypatch.setattr(
        "scripts.eval.load_experiment_config",
        lambda *args, **kwargs: {"experiment_name": "demo"},
    )

    args = type(
        "Args",
        (),
        {
            "task": "lm_harness",
            "checkpoint": "outputs/demo/checkpoint_step_100.pt",
            "all_checkpoints": False,
            "config": "demo",
            "output_dir": None,
            "device": "cpu",
            "stride": 512,
            "max_documents": None,
            "batch_size": 1,
            "limit": None,
            "overrides": [],
        },
    )()

    monkeypatch.setattr(
        "scripts.eval.run_lm_harness_eval",
        lambda **kwargs: {"task": "lm_harness"},
    )
    monkeypatch.setattr("scripts.eval.log_results_to_wandb", lambda payload, config: True)

    assert run_task(args)["task"] == "lm_harness"


def test_run_task_dispatches_lm_harness(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.eval.load_experiment_config",
        lambda *args, **kwargs: {"experiment_name": "demo"},
    )

    captured = {}

    def fake_run_lm_harness_eval(**kwargs):
        captured.update(kwargs)
        return {"task": "lm_harness"}

    monkeypatch.setattr("scripts.eval.run_lm_harness_eval", fake_run_lm_harness_eval)
    monkeypatch.setattr("scripts.eval.log_results_to_wandb", lambda payload, config: True)

    result = main(
        [
            "--task",
            "lm_harness",
            "--checkpoint",
            str(tmp_path / "checkpoint_step_100.pt"),
            "--config",
            "demo",
            "--device",
            "cpu",
            "--batch-size",
            "4",
            "--limit",
            "16",
        ]
    )

    assert result["task"] == "lm_harness"
    assert captured["output_path"] == Path("outputs/demo/eval/lm_harness.json")
    assert captured["device"] == "cpu"
    assert captured["batch_size"] == "4"
    assert captured["limit"] == 16.0


def test_run_task_dispatches_efficiency(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.eval.load_experiment_config",
        lambda *args, **kwargs: {"experiment_name": "demo"},
    )

    captured = {}

    def fake_run_efficiency_eval(**kwargs):
        captured.update(kwargs)
        return {"task": "efficiency"}

    monkeypatch.setattr("scripts.eval.run_efficiency_eval", fake_run_efficiency_eval)
    monkeypatch.setattr("scripts.eval.log_results_to_wandb", lambda payload, config: True)

    result = main(
        [
            "--task",
            "efficiency",
            "--checkpoint",
            str(tmp_path / "checkpoint_step_100.pt"),
            "--config",
            "demo",
            "--device",
            "cpu",
            "--seq-len",
            "256",
            "--warmup-iters",
            "2",
            "--benchmark-iters",
            "5",
            "--reference-checkpoint",
            str(tmp_path / "checkpoint_step_200.pt"),
        ]
    )

    assert result["task"] == "efficiency"
    assert captured["output_path"] == Path("outputs/demo/eval/efficiency.json")
    assert captured["device"] == "cpu"
    assert captured["seq_len"] == 256
    assert captured["warmup_iters"] == 2
    assert captured["benchmark_iters"] == 5
    assert str(captured["reference_checkpoint_path"]).endswith("checkpoint_step_200.pt")


def test_run_task_logs_eval_payload_to_wandb(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "scripts.eval.load_experiment_config",
        lambda *args, **kwargs: {"experiment_name": "demo"},
    )

    payload = {
        "task": "perplexity",
        "experiment_name": "demo",
        "checkpoint_step": 100,
        "results": {"wikitext103_ppl": 12.3},
        "metadata": {},
    }
    monkeypatch.setattr("scripts.eval.run_perplexity_eval", lambda **kwargs: payload)

    captured = {}

    def fake_log_results_to_wandb(payload, config):
        captured["payload"] = payload
        captured["config"] = config
        return True

    monkeypatch.setattr("scripts.eval.log_results_to_wandb", fake_log_results_to_wandb)

    result = main(
        [
            "--task",
            "perplexity",
            "--checkpoint",
            str(tmp_path / "checkpoint_step_100.pt"),
            "--config",
            "demo",
        ]
    )

    assert result is payload
    assert captured["payload"] is payload
    assert captured["config"] == {"experiment_name": "demo"}


def test_run_task_sweeps_all_checkpoints_into_history_outputs(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    first = checkpoint_dir / "checkpoint_step_100.pt"
    second = checkpoint_dir / "checkpoint_step_200.pt"
    first.write_text("stub", encoding="utf-8")
    second.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(
        "scripts.eval.load_experiment_config",
        lambda *args, **kwargs: {"experiment_name": "demo"},
    )

    captured = []

    def fake_run_perplexity_eval(**kwargs):
        captured.append(kwargs)
        return {
            "task": "perplexity",
            "experiment_name": "demo",
            "checkpoint_step": int(kwargs["checkpoint_path"].stem.split("_")[-1]),
            "results": {},
            "metadata": {},
        }

    monkeypatch.setattr("scripts.eval.run_perplexity_eval", fake_run_perplexity_eval)
    monkeypatch.setattr("scripts.eval.log_results_to_wandb", lambda payload, config: True)

    result = main(
        [
            "--task",
            "perplexity",
            "--checkpoint",
            str(checkpoint_dir),
            "--all-checkpoints",
            "--config",
            "demo",
        ]
    )

    assert [payload["checkpoint_step"] for payload in result] == [100, 200]
    assert captured[0]["output_path"] == Path(
        "outputs/demo/eval/history/checkpoint_step_100/perplexity.json"
    )
    assert captured[1]["output_path"] == Path(
        "outputs/demo/eval/history/checkpoint_step_200/perplexity.json"
    )
