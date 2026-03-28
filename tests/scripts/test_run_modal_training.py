from unittest.mock import patch, MagicMock
import pytest
import sys


@pytest.fixture(autouse=True)
def mock_modal_module():
    mock_modal = MagicMock()
    mock_modal.App = MagicMock(return_value=MagicMock())
    mock_modal.Image = MagicMock()
    mock_modal.Secret = MagicMock()
    mock_modal.Volume = MagicMock()
    mock_modal.Volume.from_name = MagicMock(return_value=MagicMock())
    with patch.dict(sys.modules, {"modal": mock_modal}):
        yield


def test_config_path():
    import run_modal_training

    assert (
        run_modal_training._config_path("experiments/test.yaml")
        == "/app/experiments/test.yaml"
    )
    assert run_modal_training._config_path("/abs/path.yaml") == "/abs/path.yaml"
    assert run_modal_training._config_path("test") == "/app/experiments/test"


def test_override_list():
    import run_modal_training

    assert run_modal_training._override_list("") == []
    assert run_modal_training._override_list("a=1,b=2") == ["a=1", "b=2"]
    assert run_modal_training._override_list("a=1, b=2 ,") == ["a=1", "b=2"]


def test_experiment_output_dir():
    import run_modal_training

    cfg = MagicMock()
    cfg.experiment_name = "test_exp"
    result = run_modal_training._experiment_output_dir(cfg)
    assert "test_exp" in result


def test_checkpoint_sort_key():
    import run_modal_training
    from pathlib import Path

    p1 = Path("checkpoint_step_100.pt")
    p2 = Path("checkpoint_step_200.pt")
    p3 = Path("best_model.pt")
    assert run_modal_training._checkpoint_sort_key(
        p1
    ) < run_modal_training._checkpoint_sort_key(p2)
    assert run_modal_training._checkpoint_sort_key(p3)[0] == 10**18


def test_latest_checkpoint_path(tmp_path):
    import run_modal_training

    (tmp_path / "checkpoint_step_100.pt").touch()
    (tmp_path / "checkpoint_step_200.pt").touch()
    result = run_modal_training._latest_checkpoint_path(tmp_path)
    assert "200" in str(result)


def test_latest_checkpoint_path_empty(tmp_path):
    import run_modal_training

    with pytest.raises(FileNotFoundError):
        run_modal_training._latest_checkpoint_path(tmp_path)


def test_resolve_runtime_path():
    import run_modal_training

    assert run_modal_training._resolve_runtime_path("") == ""
    assert run_modal_training._resolve_runtime_path("/abs/path") == "/abs/path"
    assert "outputs" in run_modal_training._resolve_runtime_path("outputs/test")
    assert (
        run_modal_training._resolve_runtime_path("relative/path")
        == "/app/relative/path"
    )


def test_resolve_eval_tasks():
    import run_modal_training

    assert run_modal_training._resolve_eval_tasks("all") == [
        "perplexity",
        "lm_harness",
        "efficiency",
    ]
    assert run_modal_training._resolve_eval_tasks("perplexity") == ["perplexity"]
    assert run_modal_training._resolve_eval_tasks("perplexity,lm_harness") == [
        "perplexity",
        "lm_harness",
    ]
    assert run_modal_training._resolve_eval_tasks("") == []
    with pytest.raises(ValueError, match="Unsupported"):
        run_modal_training._resolve_eval_tasks("invalid_task")


def test_resolve_eval_checkpoint_best(tmp_path):
    import run_modal_training

    cfg = MagicMock()
    cfg.experiment_name = "test"
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    best = checkpoints_dir / "best_model.pt"
    best.touch()
    with patch.object(
        run_modal_training, "_experiment_output_dir", return_value=str(tmp_path)
    ):
        result = run_modal_training._resolve_eval_checkpoint(cfg, "best", False)
        assert "best_model" in result


def test_resolve_eval_checkpoint_latest(tmp_path):
    import run_modal_training

    cfg = MagicMock()
    cfg.experiment_name = "test"
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    (checkpoints_dir / "checkpoint_step_100.pt").touch()
    (checkpoints_dir / "checkpoint_step_200.pt").touch()
    with patch.object(
        run_modal_training, "_experiment_output_dir", return_value=str(tmp_path)
    ):
        result = run_modal_training._resolve_eval_checkpoint(cfg, "latest", False)
        assert "200" in result


def test_resolve_eval_checkpoint_explicit():
    import run_modal_training

    cfg = MagicMock()
    result = run_modal_training._resolve_eval_checkpoint(
        cfg, "/abs/path/ckpt.pt", False
    )
    assert result == "/abs/path/ckpt.pt"


def test_resolve_eval_checkpoint_all(tmp_path):
    import run_modal_training

    cfg = MagicMock()
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    with patch.object(
        run_modal_training, "_experiment_output_dir", return_value=str(tmp_path)
    ):
        result = run_modal_training._resolve_eval_checkpoint(cfg, "", True)
        assert "checkpoints" in result


def test_load_cfg():
    import run_modal_training
    from omegaconf import OmegaConf

    with patch.object(run_modal_training, "OmegaConf", OmegaConf):
        with patch("omegaconf.OmegaConf.load") as mock_load:
            mock_cfg = MagicMock()
            mock_load.return_value = mock_cfg
            result = run_modal_training._load_cfg("/app/config.yaml", "")
            assert result is mock_cfg


def test_load_cfg_no_overrides():
    import run_modal_training
    from omegaconf import OmegaConf

    with patch.object(run_modal_training, "OmegaConf", OmegaConf):
        with patch("omegaconf.OmegaConf.load") as mock_load:
            mock_cfg = MagicMock()
            mock_load.return_value = mock_cfg
            result = run_modal_training._load_cfg("/app/config.yaml", "")
            assert result is mock_cfg
