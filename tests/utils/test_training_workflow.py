"""Tests for src/utils/training_workflow.py."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest


def _make_mock_cfg(
    dataset_key="fineweb-edu",
    model_key="qwen2-1.5b",
    exp_name="qwen2_1.5b_stress_v3_fineweb",
):
    cfg = MagicMock()
    cfg.get.side_effect = lambda k, d=None: {"experiment_name": exp_name}.get(k, d)
    return cfg


# ---------------------------------------------------------------------------
# config_name lookup
# ---------------------------------------------------------------------------


def test_execute_training_workflow_config_name_hyphen(tmp_path):
    """config_name with hyphens (YAML stem) must resolve to the correct file,
    even when experiment_name in the YAML uses underscores."""
    from src.utils.training_workflow import execute_training_workflow

    # Create a YAML with the hyphenated filename but underscore experiment_name
    cfg_file = tmp_path / "qwen2_1.5b_stress_v3-fineweb.yaml"
    cfg_file.write_text("experiment_name: qwen2_1.5b_stress_v3_fineweb\n")

    mock_cfg = MagicMock()
    mock_cfg.get.side_effect = lambda k, d=None: {
        "experiment_name": "qwen2_1.5b_stress_v3_fineweb"
    }.get(k, d)

    mock_shard_dir = tmp_path / "shards" / "fineweb-edu" / "vocab151936"
    mock_shard_dir.mkdir(parents=True)
    # Place a dummy shard so prepare_data is skipped
    (mock_shard_dir / "train_shard_0000.bin").write_bytes(b"")

    mock_get_shard_dir = MagicMock(return_value=mock_shard_dir)

    with patch("src.utils.training_workflow.EXPERIMENTS_DIR", tmp_path):
        with patch("src.utils.training_workflow.get_shard_dir", mock_get_shard_dir):
            with patch("src.utils.training_workflow.OmegaConf") as mock_oc:
                mock_oc.select.side_effect = lambda cfg, key, default=None: {
                    "dataset.dataset_key": "fineweb-edu",
                    "model.model_key": "qwen2-1.5b",
                }.get(key, default)
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0)
                    with patch(
                        "src.utils.training_workflow._read_last_checkpoint_metrics",
                        return_value={"loss": 0.4, "best_loss": 0.4},
                    ):
                        with patch("torch.cuda.device_count", return_value=1):
                            with patch("torch.cuda.is_available", return_value=False):
                                output_dir, metrics = execute_training_workflow(
                                    mock_cfg,
                                    str(tmp_path / "cache"),
                                    config_name="qwen2_1.5b_stress_v3-fineweb",
                                )

    # Verify train subprocess was called
    train_calls = [c for c in mock_run.call_args_list if "scripts.train" in str(c)]
    assert train_calls, "train subprocess must be called"


def test_execute_training_workflow_missing_config_name(tmp_path):
    """When no config_name is given and experiment_name doesn't match a file,
    FileNotFoundError must be raised."""
    from src.utils.training_workflow import execute_training_workflow

    mock_cfg = MagicMock()
    mock_cfg.get.side_effect = lambda k, d=None: {
        "experiment_name": "nonexistent_config"
    }.get(k, d)

    with patch("src.utils.training_workflow.EXPERIMENTS_DIR", tmp_path):
        with patch("src.utils.training_workflow.OmegaConf") as mock_oc:
            mock_oc.select.side_effect = lambda cfg, key, default=None: {
                "dataset.dataset_key": "fineweb-edu",
                "model.model_key": "qwen2-1.5b",
            }.get(key, default)
            mock_shard_dir = tmp_path / "shards"
            with patch(
                "src.utils.training_workflow.get_shard_dir",
                return_value=mock_shard_dir,
            ):
                with pytest.raises(FileNotFoundError):
                    execute_training_workflow(mock_cfg, str(tmp_path / "cache"))


def test_execute_training_workflow_hf_token_in_subprocess_env(tmp_path):
    """HF_TOKEN must be present in the subprocess env for both prepare_data and train."""
    from src.utils.training_workflow import execute_training_workflow

    cfg_file = tmp_path / "my-config.yaml"
    cfg_file.write_text("experiment_name: my_config\n")

    mock_cfg = MagicMock()
    mock_cfg.get.side_effect = lambda k, d=None: {"experiment_name": "my_config"}.get(
        k, d
    )

    mock_shard_dir = tmp_path / "shards"
    mock_shard_dir.mkdir(parents=True)
    # No train shard → prepare_data will run

    captured_envs = []

    def capture_run(cmd, **kwargs):
        captured_envs.append(kwargs.get("env", {}))
        return MagicMock(returncode=0)

    with patch("src.utils.training_workflow.EXPERIMENTS_DIR", tmp_path):
        with patch(
            "src.utils.training_workflow.get_shard_dir", return_value=mock_shard_dir
        ):
            with patch("src.utils.training_workflow.OmegaConf") as mock_oc:
                mock_oc.select.side_effect = lambda cfg, key, default=None: {
                    "dataset.dataset_key": "wikitext-2",
                    "model.model_key": "gpt-neo-125m",
                }.get(key, default)
                with patch("subprocess.run", side_effect=capture_run):
                    with patch(
                        "src.utils.training_workflow._read_last_checkpoint_metrics",
                        return_value={"loss": 0.5, "best_loss": 0.5},
                    ):
                        with patch("torch.cuda.device_count", return_value=1):
                            with patch("torch.cuda.is_available", return_value=False):
                                with patch.dict(
                                    os.environ, {"HF_TOKEN": "hf_abc123"}, clear=False
                                ):
                                    execute_training_workflow(
                                        mock_cfg,
                                        str(tmp_path / "cache"),
                                        config_name="my-config",
                                    )

    assert captured_envs, "subprocess.run was never called"
    for env in captured_envs:
        assert env.get("HF_TOKEN") == "hf_abc123", "HF_TOKEN must be in subprocess env"


def test_execute_training_workflow_torchrun_multi_gpu(tmp_path):
    """With multiple GPUs, torchrun with --standalone must be used."""
    from src.utils.training_workflow import execute_training_workflow

    cfg_file = tmp_path / "my-config.yaml"
    cfg_file.write_text("experiment_name: my_config\n")

    mock_cfg = MagicMock()
    mock_cfg.get.side_effect = lambda k, d=None: {"experiment_name": "my_config"}.get(
        k, d
    )

    mock_shard_dir = tmp_path / "shards"
    mock_shard_dir.mkdir(parents=True)
    # Place a dummy shard so prepare_data is skipped
    (mock_shard_dir / "train_shard_0000.bin").write_bytes(b"")

    with patch("src.utils.training_workflow.EXPERIMENTS_DIR", tmp_path):
        with patch(
            "src.utils.training_workflow.get_shard_dir", return_value=mock_shard_dir
        ):
            with patch("src.utils.training_workflow.OmegaConf") as mock_oc:
                mock_oc.select.side_effect = lambda cfg, key, default=None: {
                    "dataset.dataset_key": "wikitext-2",
                    "model.model_key": "gpt-neo-125m",
                }.get(key, default)
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0)
                    with patch(
                        "src.utils.training_workflow._read_last_checkpoint_metrics",
                        return_value={"loss": 0.5, "best_loss": 0.5},
                    ):
                        with patch("torch.cuda.device_count", return_value=4):
                            with patch("torch.cuda.is_available", return_value=True):
                                execute_training_workflow(
                                    mock_cfg,
                                    str(tmp_path / "cache"),
                                    config_name="my-config",
                                )

    train_cmd = mock_run.call_args_list[-1][0][0]
    assert "torchrun" in train_cmd[0], "torchrun must be used for multi-GPU"
    assert "--standalone" in train_cmd, "torchrun must use --standalone flag"
    assert "--nproc_per_node=4" in train_cmd
