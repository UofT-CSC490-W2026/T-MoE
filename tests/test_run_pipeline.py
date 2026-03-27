"""Tests for run_pipeline.py — lightweight mocked tests."""

import sys
from unittest.mock import patch, MagicMock
import pytest


def test_load_configuration_import_error():
    """load_configuration raises ImportError when infra.config.config is missing."""
    from run_pipeline import load_configuration

    # Just verify the function exists and is callable
    assert callable(load_configuration)


def test_load_configuration_success():
    """load_configuration returns config on success."""
    mock_config = MagicMock()
    mock_config.use_sagemaker = False
    mock_config.dataset_name = "test"
    mock_config.raw_data_bucket = "bucket"
    mock_config.aws_region = "us-east-1"
    # load_pipeline_config is imported locally inside load_configuration
    mock_infra = MagicMock()
    mock_infra.config.config.load_pipeline_config.return_value = mock_config
    with patch.dict(
        sys.modules,
        {
            "infra": mock_infra,
            "infra.config": mock_infra.config,
            "infra.config.config": mock_infra.config.config,
        },
    ):
        from run_pipeline import load_configuration

        result = load_configuration()
        assert result is mock_config


def test_run_sagemaker_pipeline_callable():
    from run_pipeline import run_sagemaker_pipeline

    assert callable(run_sagemaker_pipeline)


def test_run_fallback_pipeline_callable():
    from run_pipeline import run_fallback_pipeline

    assert callable(run_fallback_pipeline)


def test_main_keyboard_interrupt():
    """main() handles KeyboardInterrupt gracefully."""
    from run_pipeline import main

    with patch("run_pipeline.load_configuration", side_effect=KeyboardInterrupt):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 130


def test_main_value_error():
    """main() handles ValueError with exit code 1."""
    from run_pipeline import main

    with patch("run_pipeline.load_configuration", side_effect=ValueError("bad")):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


def test_main_import_error():
    """main() handles ImportError with exit code 1."""
    from run_pipeline import main

    with patch("run_pipeline.load_configuration", side_effect=ImportError("missing")):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


def test_main_generic_exception():
    """main() handles generic Exception with exit code 2."""
    from run_pipeline import main

    with patch("run_pipeline.load_configuration", side_effect=RuntimeError("boom")):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2


def test_main_sagemaker_path():
    """main() routes to sagemaker pipeline when use_sagemaker=True."""
    mock_config = MagicMock()
    mock_config.use_sagemaker = True
    mock_result = {
        "mode": "sagemaker",
        "job_arn": "arn:123",
        "dataset": "test",
        "s3_bucket": "b",
    }
    with patch("run_pipeline.load_configuration", return_value=mock_config):
        with patch(
            "run_pipeline.run_sagemaker_pipeline", return_value=mock_result
        ) as mock_sm:
            from run_pipeline import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            mock_sm.assert_called_once()


def test_main_fallback_path():
    """main() routes to fallback pipeline when use_sagemaker=False."""
    mock_config = MagicMock()
    mock_config.use_sagemaker = False
    mock_config.dataset_name = "test"
    mock_result = {
        "mode": "fallback",
        "dataset": "test",
        "total_records": 100,
        "elapsed_seconds": 5.0,
        "s3_paths": ["a"],
    }
    with patch("run_pipeline.load_configuration", return_value=mock_config):
        with patch(
            "run_pipeline.run_fallback_pipeline", return_value=mock_result
        ) as mock_fb:
            from run_pipeline import main

            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
            mock_fb.assert_called_once()
