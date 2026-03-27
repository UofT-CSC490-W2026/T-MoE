"""Extra coverage for run_modal_training.py — missing lines 121-122, 141, 194."""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture(autouse=True)
def mock_modal():
    mock_modal = MagicMock()
    mock_modal.App = MagicMock(return_value=MagicMock())
    mock_modal.Image = MagicMock()
    mock_modal.Secret = MagicMock()
    mock_modal.Volume = MagicMock()
    mock_modal.Volume.from_name = MagicMock(return_value=MagicMock())
    with patch.dict(sys.modules, {"modal": mock_modal}):
        yield


def test_load_cfg_with_overrides():
    """Covers lines 121-122: the overrides merge path in _load_cfg."""
    import run_modal_training
    from omegaconf import OmegaConf

    with patch.object(run_modal_training, "OmegaConf", OmegaConf):
        with patch("omegaconf.OmegaConf.load") as mock_load:
            base_cfg = OmegaConf.create({"training": {"lr": 1e-4}})
            mock_load.return_value = base_cfg
            result = run_modal_training._load_cfg(
                "/app/config.yaml", "training.lr=5e-5"
            )
    assert abs(result.training.lr - 5e-5) < 1e-10


def test_resolve_runtime_path_outputs():
    """Covers line 141: outputs/ prefix path."""
    import run_modal_training

    result = run_modal_training._resolve_runtime_path("outputs/my_experiment")
    assert result.startswith(run_modal_training.VOLUME_MOUNT)
    assert "outputs/my_experiment" in result


def test_resolve_eval_checkpoint_no_best_fallback_to_latest(tmp_path):
    """Covers line 194: fallback to _latest_checkpoint_path when best_model.pt missing."""
    import run_modal_training

    cfg = MagicMock()
    cfg.experiment_name = "test"
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    (checkpoints_dir / "checkpoint_step_100.pt").touch()

    with patch.object(
        run_modal_training, "_experiment_output_dir", return_value=str(tmp_path)
    ):
        result = run_modal_training._resolve_eval_checkpoint(cfg, "", False)
    assert "100" in result


def test_checkpoint_sort_key_invalid_step():
    """Covers the ValueError branch in _checkpoint_sort_key."""
    import run_modal_training

    p = Path("checkpoint_step_notanumber.pt")
    key = run_modal_training._checkpoint_sort_key(p)
    assert key[0] == 10**18
