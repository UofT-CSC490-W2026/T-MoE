import sys

import pytest

from unittest.mock import patch, MagicMock


def test_load_configuration_import_error_raises():

    mods_to_remove = [k for k in sys.modules if "infra.config" in k]

    saved = {k: sys.modules.pop(k) for k in mods_to_remove}

    try:
        with patch.dict(sys.modules, {"infra.config.config": None}):
            from run_pipeline import load_configuration

            with pytest.raises((ImportError, TypeError)):
                load_configuration()

    finally:
        sys.modules.update(saved)


def test_load_configuration_value_error_raises():

    mock_infra = MagicMock()

    mock_infra.config.config.load_pipeline_config.side_effect = ValueError(
        "missing field"
    )

    with patch.dict(
        sys.modules,
        {
            "infra": mock_infra,
            "infra.config": mock_infra.config,
            "infra.config.config": mock_infra.config.config,
        },
    ):
        from run_pipeline import load_configuration

        with pytest.raises(ValueError, match="missing field"):
            load_configuration()


def test_run_sagemaker_pipeline_import_error():

    from run_pipeline import run_sagemaker_pipeline

    mock_config = MagicMock()

    with patch.dict(sys.modules, {"infra.data_ingestion.run_processing": None}):
        with pytest.raises((ImportError, TypeError)):
            run_sagemaker_pipeline(mock_config)


def test_run_sagemaker_pipeline_success():

    from run_pipeline import run_sagemaker_pipeline

    mock_config = MagicMock()

    mock_config.dataset_name = "test"

    mock_config.raw_data_bucket = "bucket"

    mock_run_proc = MagicMock()

    mock_run_proc.load_terraform_outputs.return_value = None

    mock_run_proc.create_processor.return_value = MagicMock()

    mock_run_proc.run_processing_job.return_value = (
        "arn:aws:sagemaker:us-east-1:123:job/test"
    )

    with patch.dict(
        sys.modules, {"infra.data_ingestion.run_processing": mock_run_proc}
    ):
        result = run_sagemaker_pipeline(mock_config)

    assert result["mode"] == "sagemaker"

    assert "job_arn" in result


def test_run_sagemaker_pipeline_with_tf_outputs():

    from run_pipeline import run_sagemaker_pipeline

    mock_config = MagicMock()

    mock_config.dataset_name = "test"

    mock_config.raw_data_bucket = "bucket"

    mock_run_proc = MagicMock()

    mock_run_proc.load_terraform_outputs.return_value = {"key": "val"}

    mock_run_proc.create_processor.return_value = MagicMock()

    mock_run_proc.run_processing_job.return_value = "arn:123"

    with patch.dict(
        sys.modules, {"infra.data_ingestion.run_processing": mock_run_proc}
    ):
        run_sagemaker_pipeline(mock_config)

    mock_run_proc.apply_terraform_outputs.assert_called_once()


def test_run_fallback_pipeline_import_error():

    from run_pipeline import run_fallback_pipeline

    mock_config = MagicMock()

    with patch.dict(sys.modules, {"infra.data_ingestion.fallback_ingestion": None}):
        with pytest.raises((ImportError, TypeError)):
            run_fallback_pipeline(mock_config)


def test_run_fallback_pipeline_success():

    from run_pipeline import run_fallback_pipeline

    mock_config = MagicMock()

    mock_config.dataset_name = "test"

    mock_config.raw_data_bucket = "bucket"

    mock_config.raw_data_prefix = "datasets/raw"

    mock_config.aws_region = "us-east-1"

    mock_config.dataset_config = None

    mock_config.dataset_splits = ["train"]

    mock_config.output_format = "jsonl"

    mock_config.max_retries = 1

    mock_config.log_level = "INFO"

    mock_fallback_mod = MagicMock()

    mock_instance = MagicMock()

    mock_instance.run.return_value = {"total_records": 50, "s3_paths": ["s3://b/f"]}

    mock_fallback_mod.FallbackIngestion.return_value = mock_instance

    with patch.dict(
        sys.modules, {"infra.data_ingestion.fallback_ingestion": mock_fallback_mod}
    ):
        result = run_fallback_pipeline(mock_config)

    assert result["mode"] == "fallback"

    assert result["total_records"] == 50
