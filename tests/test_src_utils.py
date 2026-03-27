"""Tests for src/utils/ modules."""

import pytest
from unittest.mock import patch, MagicMock


# ── config_loader ──────────────────────────────────────────────────────────────


def test_load_experiment_config_by_path(tmp_path):
    from src.utils.config_loader import load_experiment_config

    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text("experiment_name: test\ntraining:\n  lr: 0.001\n")
    cfg = load_experiment_config(str(cfg_path))
    assert cfg.experiment_name == "test"


def test_load_experiment_config_with_overrides(tmp_path):
    from src.utils.config_loader import load_experiment_config

    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text("experiment_name: test\ntraining:\n  lr: 0.001\n")
    cfg = load_experiment_config(str(cfg_path), overrides=["training.lr=0.01"])
    assert cfg.training.lr == 0.01


def test_load_experiment_config_bare_name(tmp_path):
    from src.utils.config_loader import load_experiment_config
    from src.project_types import EXPERIMENTS_DIR

    # Use a real experiment file
    exp_files = list(EXPERIMENTS_DIR.glob("*.yaml"))
    if not exp_files:
        pytest.skip("No experiment files found")  # pragma: no cover
    stem = exp_files[0].stem
    cfg = load_experiment_config(stem)
    assert cfg is not None


def test_load_experiment_config_not_found(tmp_path):
    from src.utils.config_loader import load_experiment_config

    with pytest.raises(SystemExit):
        load_experiment_config(str(tmp_path / "nonexistent.yaml"))


def test_load_experiment_config_sets_experiment_name(tmp_path):
    from src.utils.config_loader import load_experiment_config

    cfg_path = tmp_path / "my_experiment.yaml"
    cfg_path.write_text("training:\n  lr: 0.001\n")
    cfg = load_experiment_config(str(cfg_path))
    assert cfg.experiment_name == "my_experiment"


# ── training_workflow ──────────────────────────────────────────────────────────


def test_read_last_checkpoint_metrics_no_checkpoints(tmp_path):
    from src.utils.training_workflow import _read_last_checkpoint_metrics

    result = _read_last_checkpoint_metrics(tmp_path)
    assert result["loss"] == float("inf")


def test_read_last_checkpoint_metrics_with_checkpoint(tmp_path):
    import json
    from src.utils.training_workflow import _read_last_checkpoint_metrics

    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    meta = {"metrics": {"loss": 0.42}}
    (ckpt_dir / "checkpoint_step_100.json").write_text(json.dumps(meta))
    result = _read_last_checkpoint_metrics(tmp_path)
    assert result["loss"] == pytest.approx(0.42)


def test_execute_training_workflow_config_not_found(tmp_path):
    from src.utils.training_workflow import execute_training_workflow
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"experiment_name": "nonexistent_exp_xyz"})
    with pytest.raises(FileNotFoundError):
        execute_training_workflow(cfg, str(tmp_path))


def test_execute_training_workflow_runs(tmp_path):
    from src.utils.training_workflow import execute_training_workflow
    from omegaconf import OmegaConf
    from src.project_types import EXPERIMENTS_DIR

    exp_files = list(EXPERIMENTS_DIR.glob("*.yaml"))
    if not exp_files:
        pytest.skip("No experiment files found")  # pragma: no cover

    cfg = OmegaConf.create({"experiment_name": exp_files[0].stem})

    mock_shard_path = MagicMock()
    mock_shard_path.glob.return_value = iter([])  # no existing shards

    with patch("src.configs.dataset.get_shard_dir", return_value=mock_shard_path):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            with patch("torch.cuda.is_available", return_value=False):
                with patch("torch.cuda.device_count", return_value=0):
                    output_dir, metrics = execute_training_workflow(cfg, str(tmp_path))
                    assert isinstance(output_dir, str)
