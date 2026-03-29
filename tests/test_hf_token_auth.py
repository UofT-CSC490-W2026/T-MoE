from __future__ import annotations

import os
from unittest.mock import MagicMock, patch


class TestHfEnv:
    def test_token_present_is_injected(self):
        from run_modal_training import _hf_env

        with patch.dict(os.environ, {"HF_TOKEN": "test-token-abc"}, clear=False):
            result = _hf_env({"KEY": "val"})

        assert result["HF_TOKEN"] == "test-token-abc"

    def test_token_absent_dict_unchanged(self):
        from run_modal_training import _hf_env

        env = os.environ.copy()
        env.pop("HF_TOKEN", None)
        with patch.dict(os.environ, env, clear=True):
            base = {"KEY": "val"}
            result = _hf_env(base)

        assert result == base
        assert "HF_TOKEN" not in result

    def test_existing_keys_preserved_when_token_injected(self):
        from run_modal_training import _hf_env

        with patch.dict(os.environ, {"HF_TOKEN": "tok"}, clear=False):
            result = _hf_env({"HF_DATASETS_CACHE": "/cache", "HF_HOME": "/home"})

        assert result["HF_DATASETS_CACHE"] == "/cache"
        assert result["HF_HOME"] == "/home"
        assert result["HF_TOKEN"] == "tok"


def test_stage_data_subprocess_env_includes_hf_token():
    from run_modal_training import _hf_env

    hf_cache = "/vol/hf_cache"
    with patch.dict(os.environ, {"HF_TOKEN": "stage-data-token"}, clear=False):
        env = _hf_env(
            {**os.environ, "HF_DATASETS_CACHE": hf_cache, "HF_HOME": hf_cache}
        )

    assert env.get("HF_TOKEN") == "stage-data-token"
    assert env.get("HF_DATASETS_CACHE") == hf_cache
    assert env.get("HF_HOME") == hf_cache


def test_stage_eval_data_subprocess_env_includes_hf_token():
    from run_modal_training import _hf_env

    hf_cache = "/vol/hf_cache"
    with patch.dict(os.environ, {"HF_TOKEN": "eval-data-token"}, clear=False):
        env = _hf_env(
            {**os.environ, "HF_DATASETS_CACHE": hf_cache, "HF_HOME": hf_cache}
        )

    assert env.get("HF_TOKEN") == "eval-data-token"


def test_stage_eval_perplexity_subprocess_env_includes_hf_token():
    from run_modal_training import _hf_env

    hf_cache = "/vol/hf_cache"
    shards_dir = "/vol/data"
    with patch.dict(os.environ, {"HF_TOKEN": "eval-token"}, clear=False):
        env = _hf_env(
            {
                **os.environ,
                "HF_DATASETS_CACHE": hf_cache,
                "HF_HOME": hf_cache,
                "SHARD_BASE_DIR": shards_dir,
            }
        )

    assert env.get("HF_TOKEN") == "eval-token"
    assert env.get("SHARD_BASE_DIR") == shards_dir


def _make_pipeline_config():
    cfg = MagicMock()
    cfg.aws_region = "us-east-1"
    cfg.environment = "dev"
    cfg.raw_data_bucket = "my-bucket"
    cfg.dataset_name = "wikitext"
    return cfg


def test_submit_batch_job_includes_hf_token_when_set():
    import importlib

    mock_client = MagicMock()
    mock_client.submit_job.return_value = {"jobId": "job-123"}
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_client

    # Patch boto3 in sys.modules so the lazy `import boto3` inside the function picks up our mock.
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        from infra.backends import aws_backend

        importlib.reload(aws_backend)

        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("HF_TOKEN", "WANDB_API_KEY")
        }
        clean_env["HF_TOKEN"] = "batch-hf-token"
        with patch.dict(os.environ, clean_env, clear=True):
            aws_backend._submit_batch_job(_make_pipeline_config(), "test-experiment")

    call_kwargs = mock_client.submit_job.call_args[1]
    environment = call_kwargs["containerOverrides"]["environment"]
    hf_entries = [e for e in environment if e["name"] == "HF_TOKEN"]
    assert len(hf_entries) == 1
    assert hf_entries[0]["value"] == "batch-hf-token"


def test_submit_batch_job_excludes_hf_token_when_absent():
    import importlib

    mock_client = MagicMock()
    mock_client.submit_job.return_value = {"jobId": "job-456"}
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_client

    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        from infra.backends import aws_backend

        importlib.reload(aws_backend)

        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("HF_TOKEN", "WANDB_API_KEY")
        }
        with patch.dict(os.environ, clean_env, clear=True):
            aws_backend._submit_batch_job(_make_pipeline_config(), "test-experiment")

    call_kwargs = mock_client.submit_job.call_args[1]
    environment = call_kwargs["containerOverrides"]["environment"]
    hf_entries = [e for e in environment if e["name"] == "HF_TOKEN"]
    assert len(hf_entries) == 0


def test_get_tokenizer_passes_token_kwarg():
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "pad"
    mock_tokenizer.eos_token = "eos"
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.model_max_length = int(1e30)
    mock_model_info = {"hf_name": "EleutherAI/gpt-neo-125m"}

    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
    ) as mock_fpt:
        with patch("src.configs.model.model_lookup", return_value=mock_model_info):
            with patch.dict(os.environ, {"HF_TOKEN": "get-tok-token"}, clear=False):
                from scripts.prepare_data import get_tokenizer

                get_tokenizer("gpt-neo-125m")

    mock_fpt.assert_called_once()
    _, kwargs = mock_fpt.call_args
    assert kwargs.get("token") == "get-tok-token"


def test_load_tokenizer_for_model_passes_token_kwarg():
    mock_tokenizer = MagicMock()
    mock_model_info = {"hf_name": "EleutherAI/gpt-neo-125m"}
    mock_config = MagicMock()
    mock_config.get = lambda k, default=None: default

    with patch(
        "transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer
    ) as mock_fpt:
        with patch("evals.perplexity.model_lookup", return_value=mock_model_info):
            with patch.dict(os.environ, {"HF_TOKEN": "perplexity-token"}, clear=False):
                from evals.perplexity import _load_tokenizer_for_model

                _load_tokenizer_for_model(mock_config)

    mock_fpt.assert_called_once()
    _, kwargs = mock_fpt.call_args
    assert kwargs.get("token") == "perplexity-token"


def test_load_dataset_texts_passes_token_kwarg():
    mock_dataset = MagicMock()
    mock_dataset.__iter__ = MagicMock(return_value=iter([]))
    mock_dataset.__len__ = MagicMock(return_value=0)

    dataset_spec = {
        "hf_path": "wikitext",
        "hf_name": "wikitext-103-raw-v1",
        "split": "test",
        "text_column": "text",
        "streaming": False,
    }

    with patch("datasets.load_dataset", return_value=mock_dataset) as mock_load_ds:
        with patch.dict(os.environ, {"HF_TOKEN": "dataset-token"}, clear=False):
            from evals.perplexity import _load_dataset_texts

            _load_dataset_texts(dataset_spec)

    mock_load_ds.assert_called_once()
    _, kwargs = mock_load_ds.call_args
    assert kwargs.get("token") == "dataset-token"
