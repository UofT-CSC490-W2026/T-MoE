"""Tests for scripts/eval.py — lightweight mocked tests."""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_build_parser():
    from scripts.eval import _build_parser
    parser = _build_parser()
    assert parser is not None


def test_get_eval_param_cli_priority():
    from scripts.eval import _get_eval_param
    config = MagicMock()
    result = _get_eval_param(config, "stride", 256, sentinel=None)
    assert result == 256


def test_get_eval_param_yaml_fallback():
    from scripts.eval import _get_eval_param
    from omegaconf import OmegaConf
    config = OmegaConf.create({"eval": {"stride": 1024}})
    result = _get_eval_param(config, "stride", None, sentinel=None)
    assert result == 1024


def test_get_eval_param_default_fallback():
    from scripts.eval import _get_eval_param
    from omegaconf import OmegaConf
    config = OmegaConf.create({})
    result = _get_eval_param(config, "stride", None, sentinel=None)
    assert result == 512  # hardcoded default


def test_default_output_dir():
    from scripts.eval import _default_output_dir
    config = {"experiment_name": "test_exp"}
    result = _default_output_dir(config)
    assert "test_exp" in str(result)


def test_checkpoint_sort_key():
    from scripts.eval import _checkpoint_sort_key
    p1 = Path("checkpoint_step_100.pt")
    p2 = Path("checkpoint_step_200.pt")
    p3 = Path("best_model.pt")
    assert _checkpoint_sort_key(p1) < _checkpoint_sort_key(p2)
    assert _checkpoint_sort_key(p3)[0] == 10**18


def test_resolve_checkpoint_paths_single(tmp_path):
    from scripts.eval import _resolve_checkpoint_paths
    ckpt = tmp_path / "checkpoint_step_100.pt"
    ckpt.touch()
    result = _resolve_checkpoint_paths(str(ckpt), False)
    assert len(result) == 1
    assert result[0] == ckpt


def test_resolve_checkpoint_paths_directory(tmp_path):
    from scripts.eval import _resolve_checkpoint_paths
    (tmp_path / "checkpoint_step_100.pt").touch()
    (tmp_path / "checkpoint_step_200.pt").touch()
    result = _resolve_checkpoint_paths(str(tmp_path), False)
    assert len(result) == 2


def test_resolve_checkpoint_paths_empty_dir(tmp_path):
    from scripts.eval import _resolve_checkpoint_paths
    with pytest.raises(FileNotFoundError):
        _resolve_checkpoint_paths(str(tmp_path), False)


def test_resolve_output_path_single(tmp_path):
    from scripts.eval import _resolve_output_path
    config = {"experiment_name": "test"}
    result = _resolve_output_path("perplexity", config, str(tmp_path))
    assert result.name == "perplexity.json"


def test_resolve_output_path_multiple(tmp_path):
    from scripts.eval import _resolve_output_path
    config = {"experiment_name": "test"}
    ckpt = Path("checkpoint_step_100.pt")
    result = _resolve_output_path("perplexity", config, str(tmp_path),
                                   checkpoint_path=ckpt, multiple_checkpoints=True)
    assert "history" in str(result)


def test_resolve_output_path_multiple_no_checkpoint():
    from scripts.eval import _resolve_output_path
    config = {"experiment_name": "test"}
    with pytest.raises(ValueError):
        _resolve_output_path("perplexity", config, None,
                              checkpoint_path=None, multiple_checkpoints=True)


def test_get_dist_info_no_env(monkeypatch):
    from scripts.eval import _get_dist_info
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    rank, world_size = _get_dist_info()
    assert rank == 0
    assert world_size == 1


def test_get_dist_info_with_env(monkeypatch):
    from scripts.eval import _get_dist_info
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")
    rank, world_size = _get_dist_info()
    assert rank == 1
    assert world_size == 4


def test_init_distributed_single():
    from scripts.eval import _init_distributed
    _init_distributed(0, 1)  # no-op for world_size=1


def test_load_experiment_config(tmp_path):
    from scripts.eval import load_experiment_config
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text("experiment_name: test\n")
    cfg = load_experiment_config(str(cfg_path))
    assert cfg.experiment_name == "test"


# load_model_for_eval is imported locally inside run_task, so patch at source
def _run_eval_main(tmp_path, task, extra_args=None, mock_payload=None):
    """Helper to run scripts.eval.main with mocked model loading."""
    from scripts.eval import main
    from omegaconf import OmegaConf
    ckpt = tmp_path / "checkpoint_step_100.pt"
    ckpt.touch()
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text("experiment_name: test\n")

    if mock_payload is None:
        mock_payload = {"task": task, "results": {}}

    # Use a real OmegaConf config so _get_eval_param doesn't fail
    real_cfg = OmegaConf.create({"experiment_name": "test"})

    argv = ["--task", task, "--checkpoint", str(ckpt), "--config", str(cfg_path)]
    if extra_args:
        argv.extend(extra_args)

    patches = {
        "perplexity": "scripts.eval.run_perplexity_eval",
        "lm_harness": "scripts.eval.run_lm_harness_eval",
        "efficiency": "scripts.eval.run_efficiency_eval",
    }
    with patch("scripts.eval.load_experiment_config", return_value=real_cfg):
        with patch("evals.loading.load_model_for_eval", return_value=(MagicMock(), {})):
            with patch(patches[task], return_value=mock_payload) as mock_fn:
                with patch("scripts.eval.log_results_to_wandb"):
                    result = main(argv)
    return result, mock_fn


def test_main_perplexity(tmp_path):
    result, _ = _run_eval_main(tmp_path, "perplexity")
    assert result["task"] == "perplexity"


def test_main_lm_harness(tmp_path):
    result, _ = _run_eval_main(tmp_path, "lm_harness")
    assert result["task"] == "lm_harness"


def test_main_efficiency(tmp_path):
    result, _ = _run_eval_main(tmp_path, "efficiency")
    assert result["task"] == "efficiency"


def test_main_multiple_checkpoints(tmp_path):
    from scripts.eval import main
    from omegaconf import OmegaConf
    (tmp_path / "checkpoint_step_100.pt").touch()
    (tmp_path / "checkpoint_step_200.pt").touch()
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text("experiment_name: test\n")
    mock_payload = {"task": "perplexity", "results": {}}
    real_cfg = OmegaConf.create({"experiment_name": "test"})

    with patch("scripts.eval.load_experiment_config", return_value=real_cfg):
        with patch("evals.loading.load_model_for_eval", return_value=(MagicMock(), {})):
            with patch("scripts.eval.run_perplexity_eval", return_value=mock_payload):
                with patch("scripts.eval.log_results_to_wandb"):
                    result = main([
                        "--task", "perplexity",
                        "--checkpoint", str(tmp_path),
                        "--config", str(cfg_path),
                    ])
    assert isinstance(result, list)
    assert len(result) == 2


def test_main_with_reference_config(tmp_path):
    from scripts.eval import main
    from omegaconf import OmegaConf
    ckpt = tmp_path / "checkpoint_step_100.pt"
    ckpt.touch()
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text("experiment_name: test\n")
    ref_cfg_path = tmp_path / "ref.yaml"
    ref_cfg_path.write_text("experiment_name: ref\n")
    mock_payload = {"task": "efficiency", "results": {}}
    real_cfg = OmegaConf.create({"experiment_name": "test"})

    with patch("scripts.eval.load_experiment_config", return_value=real_cfg):
        with patch("evals.loading.load_model_for_eval", return_value=(MagicMock(), {})):
            with patch("scripts.eval.run_efficiency_eval", return_value=mock_payload):
                with patch("scripts.eval.log_results_to_wandb"):
                    main([
                        "--task", "efficiency",
                        "--checkpoint", str(ckpt),
                        "--config", str(cfg_path),
                        "--reference-config", str(ref_cfg_path),
                    ])


def test_main_lm_harness_batch_size_fallback(tmp_path):
    from scripts.eval import main
    from omegaconf import OmegaConf
    ckpt = tmp_path / "checkpoint_step_100.pt"
    ckpt.touch()
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text("experiment_name: test\n")
    mock_payload = {"task": "lm_harness", "results": {}}
    real_cfg = OmegaConf.create({"experiment_name": "test"})

    with patch("scripts.eval.load_experiment_config", return_value=real_cfg):
        with patch("evals.loading.load_model_for_eval", return_value=(MagicMock(), {})):
            with patch("scripts.eval.run_lm_harness_eval", return_value=mock_payload) as mock_lm:
                with patch("scripts.eval.log_results_to_wandb"):
                    main([
                        "--task", "lm_harness",
                        "--checkpoint", str(ckpt),
                        "--config", str(cfg_path),
                        "--batch-size", "8",
                    ])
    call_kwargs = mock_lm.call_args[1]
    assert call_kwargs["batch_size"] == "8"
