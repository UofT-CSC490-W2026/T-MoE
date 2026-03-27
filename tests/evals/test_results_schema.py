import json

from evals.results_schema import (
    build_results_payload,
    flatten_scalars,
    get_git_commit,
    infer_checkpoint_step,
    log_results_to_wandb,
    write_results_json,
)


def test_infer_checkpoint_step_prefers_checkpoint_info():
    step = infer_checkpoint_step(
        "outputs/exp/checkpoint_step_1000.pt",
        checkpoint_info={"step": 42},
    )

    assert step == 42


def test_infer_checkpoint_step_falls_back_to_filename():
    step = infer_checkpoint_step("outputs/exp/checkpoint_step_1000.pt")

    assert step == 1000


def test_build_results_payload_matches_required_shape(monkeypatch):
    monkeypatch.setattr(
        "evals.results_schema.get_git_commit",
        lambda cwd=None: "deadbeef",
    )

    payload = build_results_payload(
        task="perplexity",
        checkpoint_path="outputs/demo/checkpoint_step_5000.pt",
        config={"experiment_name": "demo", "training": {"lr": 1e-4}},
        results={"wikitext103_bpb": 1.234, "wikitext103_ppl": 12.34},
        metadata={"dtype": "bfloat16", "stride": 512, "device": "cuda:0"},
        checkpoint_info={"step": 5000, "metrics": {"loss": 1.23}},
        eval_timestamp="2026-03-09T14:32:00Z",
    )

    assert payload["experiment_name"] == "demo"

    assert payload["checkpoint_step"] == 5000

    assert payload["checkpoint_path"].endswith("checkpoint_step_5000.pt")

    assert payload["eval_timestamp"] == "2026-03-09T14:32:00Z"

    assert payload["git_commit"] == "deadbeef"

    assert payload["task"] == "perplexity"

    assert payload["config"]["training"]["lr"] == 1e-4

    assert payload["results"]["wikitext103_bpb"] == 1.234

    assert payload["metadata"]["stride"] == 512


def test_write_results_json_creates_parent_dirs(tmp_path):
    output_path = tmp_path / "outputs" / "demo" / "eval" / "perplexity.json"

    payload = {
        "experiment_name": "demo",
        "checkpoint_step": 100,
        "checkpoint_path": "outputs/demo/checkpoint_step_100.pt",
        "eval_timestamp": "2026-03-09T14:32:00Z",
        "git_commit": "deadbeef",
        "task": "perplexity",
        "config": {},
        "results": {"wikitext103_bpb": 1.23},
        "metadata": {"device": "cuda:0"},
    }

    written_path = write_results_json(payload, output_path)

    assert written_path == output_path

    with output_path.open(encoding="utf-8") as handle:
        saved = json.load(handle)

    assert saved["results"]["wikitext103_bpb"] == 1.23


def test_flatten_scalars_keeps_only_scalar_entries():
    flattened = flatten_scalars(
        {
            "results": {"wikitext103_bpb": 1.23, "tags": ["ignored"]},
            "metadata": {"device": "cuda:0", "stride": 512},
            "nested": {"deep": {"flag": True}},
        }
    )

    assert flattened == {
        "results/wikitext103_bpb": 1.23,
        "metadata/device": "cuda:0",
        "metadata/stride": 512,
        "nested/deep/flag": True,
    }


def test_get_git_commit_returns_unknown_when_git_fails(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("evals.results_schema.subprocess.run", fake_run)

    assert get_git_commit() == "unknown"


def test_write_results_json_stringifies_unknown_leaf_types(tmp_path):
    class _WeirdLeaf:
        def __str__(self):
            return "float32"

    output_path = tmp_path / "results.json"

    payload = {
        "experiment_name": "demo",
        "checkpoint_step": 1,
        "checkpoint_path": "outputs/demo/checkpoint_step_1.pt",
        "eval_timestamp": "2026-03-09T14:32:00Z",
        "git_commit": "deadbeef",
        "task": "lm_harness",
        "config": {},
        "results": {},
        "metadata": {"raw_results": {"dtype": _WeirdLeaf()}},
    }

    write_results_json(payload, output_path)

    with output_path.open(encoding="utf-8") as handle:
        saved = json.load(handle)

    assert saved["metadata"]["raw_results"]["dtype"] == "float32"


def test_log_results_to_wandb_logs_scalars_and_mmlu_table(monkeypatch):
    class _FakeTable:
        def __init__(self, columns):
            self.columns = columns

            self.rows = []

        def add_data(self, *row):
            self.rows.append(row)

    class _FakeRun:
        def __init__(self):
            self.logged = []

            self.finished = False

            self.summary = {}

        def log(self, data, step=None):
            self.logged.append((data, step))

        def finish(self):
            self.finished = True

    class _FakeWandb:
        Table = _FakeTable

        def __init__(self):
            self.init_kwargs = None

            self.run = _FakeRun()

        def init(self, **kwargs):
            self.init_kwargs = kwargs

            return self.run

    fake_wandb = _FakeWandb()

    monkeypatch.delenv("WANDB_PROJECT", raising=False)

    monkeypatch.delenv("WANDB_ENTITY", raising=False)

    monkeypatch.setattr("evals.results_schema.WANDB_AVAILABLE", True)

    monkeypatch.setattr("evals.results_schema.wandb", fake_wandb)

    payload = {
        "experiment_name": "demo",
        "checkpoint_step": 42,
        "checkpoint_path": "outputs/demo/checkpoint_step_42.pt",
        "git_commit": "deadbeef",
        "task": "lm_harness",
        "config": {},
        "results": {"piqa": 0.62, "mmlu": 0.55},
        "metadata": {
            "device": "cuda:0",
            "mmlu_subjects": {
                "mmlu_anatomy": 0.70,
                "mmlu_abstract_algebra": 0.40,
            },
            "raw_results": {"ignored": True},
        },
    }

    logged = log_results_to_wandb(payload, config={"experiment_name": "demo"})

    assert logged is True

    assert fake_wandb.init_kwargs["project"] == "tmoe"

    assert fake_wandb.init_kwargs["name"] == "eval/demo/lm_harness"

    assert fake_wandb.init_kwargs["resume"] == "allow"

    assert fake_wandb.init_kwargs["id"].startswith("eval-v5-demo-lm-harness-")

    assert fake_wandb.run.logged[0] == (
        {
            "eval/lm_harness/piqa": 0.62,
            "eval/lm_harness/mmlu": 0.55,
        },
        42,
    )

    assert fake_wandb.run.logged[1][0]["eval/lm_harness/mmlu_subjects"].rows == [
        ("mmlu_abstract_algebra", 0.40),
        ("mmlu_anatomy", 0.70),
    ]

    assert fake_wandb.run.logged[1][1] == 42

    assert fake_wandb.run.summary["eval/lm_harness/meta/device"] == "cuda:0"

    assert fake_wandb.run.summary["eval/latest_checkpoint_step"] == 42

    assert fake_wandb.run.finished is True


def test_log_results_to_wandb_uses_wandb_env_defaults(monkeypatch):
    class _FakeRun:
        def __init__(self):
            self.summary = {}

        def log(self, data, step=None):
            return None

        def finish(self):
            return None

    class _FakeWandb:
        Table = object

        def __init__(self):
            self.init_kwargs = None

        def init(self, **kwargs):
            self.init_kwargs = kwargs

            return _FakeRun()

    fake_wandb = _FakeWandb()

    monkeypatch.setenv("WANDB_PROJECT", "T-MoE")

    monkeypatch.setenv("WANDB_ENTITY", "uoft")

    monkeypatch.setattr("evals.results_schema.WANDB_AVAILABLE", True)

    monkeypatch.setattr("evals.results_schema.wandb", fake_wandb)

    logged = log_results_to_wandb(
        {
            "experiment_name": "demo",
            "task": "perplexity",
            "results": {"wikitext103_ppl": 12.3},
            "metadata": {},
            "config": {},
        },
        config={"experiment_name": "demo"},
    )

    assert logged is True

    assert fake_wandb.init_kwargs["project"] == "T-MoE"

    assert fake_wandb.init_kwargs["entity"] == "uoft"

    assert fake_wandb.init_kwargs["mode"] == "online"


def test_log_results_to_wandb_overrides_disabled_env_to_online(monkeypatch):
    class _FakeRun:
        def __init__(self):
            self.summary = {}

        def log(self, data, step=None):
            return None

        def finish(self):
            return None

    class _FakeWandb:
        Table = object

        def __init__(self):
            self.init_kwargs = None

        def init(self, **kwargs):
            self.init_kwargs = kwargs

            return _FakeRun()

    fake_wandb = _FakeWandb()

    monkeypatch.setenv("WANDB_MODE", "disabled")

    monkeypatch.setattr("evals.results_schema.WANDB_AVAILABLE", True)

    monkeypatch.setattr("evals.results_schema.wandb", fake_wandb)

    logged = log_results_to_wandb(
        {
            "experiment_name": "demo",
            "task": "perplexity",
            "results": {"wikitext103_ppl": 12.3},
            "metadata": {},
            "config": {},
        },
        config={"experiment_name": "demo"},
    )

    assert logged is True

    assert fake_wandb.init_kwargs["mode"] == "online"


def test_log_results_to_wandb_skips_when_logging_disabled(monkeypatch):
    monkeypatch.setattr("evals.results_schema.WANDB_AVAILABLE", True)

    monkeypatch.setattr("evals.results_schema.wandb", object())

    logged = log_results_to_wandb(
        {"task": "perplexity", "results": {}, "metadata": {}, "config": {}},
        config={"logging": {"enabled": False}},
    )

    assert logged is False
