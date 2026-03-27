"""Tests for run_aws_training.py — lightweight mocked tests."""
import sys
import argparse
from unittest.mock import patch, MagicMock
import pytest


def test_imports():
    """Verify run_aws_training can be imported."""
    import run_aws_training  # noqa: F401


def test_log_dataset_status_found(capsys):
    from run_aws_training import _log_dataset_status
    _log_dataset_status(True)


def test_log_dataset_status_not_found(capsys):
    from run_aws_training import _log_dataset_status
    _log_dataset_status(False)


def test_log_dry_run(capsys):
    from run_aws_training import _log_dry_run
    args = argparse.Namespace(mode="local")
    _log_dry_run(True, args)
    _log_dry_run(False, args)


def test_log_completion(capsys):
    from run_aws_training import _log_completion
    _log_completion(True, {"loss": 0.5, "best_loss": 0.4}, "/tmp/out")
    _log_completion(False, {"loss": 0.5, "best_loss": 0.4}, "/tmp/out")


def test_dataset_s3_prefix():
    from run_aws_training import _dataset_s3_prefix
    mock_config = MagicMock()
    mock_config.dataset_name = "wikitext/test"
    mock_config.raw_data_prefix = "datasets/raw"
    result = _dataset_s3_prefix(mock_config)
    assert "wikitext_test" in result


def test_find_latest_timestamp_prefix_no_objects():
    from run_aws_training import _find_latest_timestamp_prefix
    mock_s3 = MagicMock()
    mock_s3.list_objects.return_value = []
    with pytest.raises(RuntimeError, match="No objects found"):
        _find_latest_timestamp_prefix(mock_s3, "bucket", "prefix/")


def test_find_latest_timestamp_prefix_no_timestamps():
    from run_aws_training import _find_latest_timestamp_prefix
    mock_s3 = MagicMock()
    mock_s3.list_objects.return_value = [{"Key": "prefix/notatimestamp/file.txt"}]
    with pytest.raises(RuntimeError, match="No timestamp directories"):
        _find_latest_timestamp_prefix(mock_s3, "bucket", "prefix/")


def test_find_latest_timestamp_prefix_success():
    from run_aws_training import _find_latest_timestamp_prefix
    mock_s3 = MagicMock()
    mock_s3.list_objects.return_value = [
        {"Key": "prefix/20240101-120000/train.jsonl"},
        {"Key": "prefix/20240215-150000/train.jsonl"},
    ]
    result = _find_latest_timestamp_prefix(mock_s3, "bucket", "prefix/")
    assert "20240215-150000" in result


def test_check_dataset_in_s3_not_found():
    from run_aws_training import check_dataset_in_s3
    mock_config = MagicMock()
    mock_config.dataset_name = "test"
    mock_config.raw_data_prefix = "datasets/raw"
    mock_config.aws_region = "us-east-1"
    mock_config.max_retries = 1
    mock_config.raw_data_bucket = "bucket"
    mock_s3client_mod = MagicMock()
    mock_client = MagicMock()
    mock_client.list_objects.return_value = []
    mock_s3client_mod.S3Client.return_value = mock_client
    with patch.dict("sys.modules", {"infra.s3client.client": mock_s3client_mod}):
        result = check_dataset_in_s3(mock_config)
        assert result is False


def test_check_dataset_in_s3_found():
    from run_aws_training import check_dataset_in_s3
    mock_config = MagicMock()
    mock_config.dataset_name = "test"
    mock_config.raw_data_prefix = "datasets/raw"
    mock_config.aws_region = "us-east-1"
    mock_config.max_retries = 1
    mock_config.raw_data_bucket = "bucket"
    mock_s3client_mod = MagicMock()
    mock_client = MagicMock()
    mock_client.list_objects.return_value = [{"Key": "datasets/raw/test/20240101-120000/f.jsonl"}]
    mock_client.dataset_exists.return_value = True
    mock_s3client_mod.S3Client.return_value = mock_client
    with patch.dict("sys.modules", {"infra.s3client.client": mock_s3client_mod}):
        result = check_dataset_in_s3(mock_config)
        assert result is True


def test_run_data_ingestion():
    from run_aws_training import run_data_ingestion
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
    mock_ingestion_mod = MagicMock()
    mock_instance = MagicMock()
    mock_instance.run.return_value = {"total_records": 100}
    mock_ingestion_mod.FallbackIngestion.return_value = mock_instance
    with patch.dict("sys.modules", {"infra.data_ingestion.fallback_ingestion": mock_ingestion_mod}):
        result = run_data_ingestion(mock_config)
        assert result["total_records"] == 100


def test_download_dataset_from_s3(tmp_path):
    from run_aws_training import download_dataset_from_s3
    mock_config = MagicMock()
    mock_config.dataset_name = "test"
    mock_config.raw_data_prefix = "datasets/raw"
    mock_config.aws_region = "us-east-1"
    mock_config.max_retries = 1
    mock_config.raw_data_bucket = "bucket"
    mock_s3client_mod = MagicMock()
    mock_client = MagicMock()
    mock_client.list_objects.return_value = [{"Key": "datasets/raw/test/20240101-120000/f.jsonl"}]
    mock_s3client_mod.S3Client.return_value = mock_client
    mock_s3sync_mod = MagicMock()
    mock_s3sync_mod.download_s3_prefix.return_value = ["file1.jsonl"]
    with patch.dict("sys.modules", {
        "infra.s3client.client": mock_s3client_mod,
        "infra.s3client.s3_sync": mock_s3sync_mod,
    }):
        download_dataset_from_s3(mock_config, str(tmp_path))


def test_download_dataset_from_s3_no_files(tmp_path):
    from run_aws_training import download_dataset_from_s3
    mock_config = MagicMock()
    mock_config.dataset_name = "test"
    mock_config.raw_data_prefix = "datasets/raw"
    mock_config.aws_region = "us-east-1"
    mock_config.max_retries = 1
    mock_config.raw_data_bucket = "bucket"
    mock_s3client_mod = MagicMock()
    mock_client = MagicMock()
    mock_client.list_objects.return_value = [{"Key": "datasets/raw/test/20240101-120000/f.jsonl"}]
    mock_s3client_mod.S3Client.return_value = mock_client
    mock_s3sync_mod = MagicMock()
    mock_s3sync_mod.download_s3_prefix.return_value = []
    with patch.dict("sys.modules", {
        "infra.s3client.client": mock_s3client_mod,
        "infra.s3client.s3_sync": mock_s3sync_mod,
    }):
        with pytest.raises(RuntimeError, match="No files downloaded"):
            download_dataset_from_s3(mock_config, str(tmp_path))


def test_upload_outputs_to_s3():
    from run_aws_training import upload_outputs_to_s3
    mock_config = MagicMock()
    mock_config.raw_data_bucket = "bucket"
    mock_config.aws_region = "us-east-1"
    mock_config.max_retries = 1
    with patch("infra.s3client.s3_sync.upload_experiment_dir", create=True) as mock_upload:
        mock_upload.return_value = {"uploaded": ["f1", "f2"], "failed": []}
        with patch.dict("sys.modules", {"infra.s3client.s3_sync": MagicMock(upload_experiment_dir=mock_upload)}):
            upload_outputs_to_s3(mock_config, "/tmp/outputs/exp1")


def test_upload_outputs_to_s3_with_failures():
    from run_aws_training import upload_outputs_to_s3
    mock_config = MagicMock()
    mock_config.raw_data_bucket = "bucket"
    mock_config.aws_region = "us-east-1"
    mock_config.max_retries = 1
    mock_s3sync = MagicMock()
    mock_s3sync.upload_experiment_dir.return_value = {"uploaded": ["f1"], "failed": ["checkpoint_fail.pt"]}
    with patch.dict("sys.modules", {"infra.s3client.s3_sync": mock_s3sync}):
        with pytest.raises(RuntimeError, match="Critical checkpoint"):
            upload_outputs_to_s3(mock_config, "/tmp/outputs/exp1")


def test_upload_outputs_non_critical_failures():
    from run_aws_training import upload_outputs_to_s3
    mock_config = MagicMock()
    mock_config.raw_data_bucket = "bucket"
    mock_config.aws_region = "us-east-1"
    mock_config.max_retries = 1
    mock_s3sync = MagicMock()
    mock_s3sync.upload_experiment_dir.return_value = {"uploaded": ["f1"], "failed": ["log.txt"]}
    with patch.dict("sys.modules", {"infra.s3client.s3_sync": mock_s3sync}):
        upload_outputs_to_s3(mock_config, "/tmp/outputs/exp1")


def test_run_training():
    from run_aws_training import run_training
    mock_config = MagicMock()
    mock_wf = MagicMock()
    mock_wf.execute_training_workflow.return_value = ("/tmp/out", {"loss": 0.5})
    with patch.dict("sys.modules", {"src.utils.training_workflow": mock_wf}):
        output_dir, metrics = run_training(mock_config, "/tmp/cache")
        assert output_dir == "/tmp/out"


def test_submit_batch_job():
    from run_aws_training import submit_batch_job
    mock_config = MagicMock()
    mock_config.aws_region = "us-east-1"
    mock_boto3 = MagicMock()
    mock_client = MagicMock()
    mock_client.submit_job.return_value = {"jobId": "job-123"}
    mock_boto3.client.return_value = mock_client
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        job_id = submit_batch_job("test_config", mock_config, [])
        assert job_id == "job-123"


def test_wait_for_batch_job_succeeded():
    from run_aws_training import wait_for_batch_job
    mock_boto3 = MagicMock()
    mock_batch = MagicMock()
    mock_batch.describe_jobs.return_value = {
        "jobs": [{"status": "SUCCEEDED", "statusReason": ""}]
    }
    mock_boto3.client.return_value = mock_batch
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = wait_for_batch_job("job-123", "us-east-1", poll_interval=0, stream_logs=False)
        assert result == "SUCCEEDED"


def test_wait_for_batch_job_failed():
    from run_aws_training import wait_for_batch_job
    mock_boto3 = MagicMock()
    mock_batch = MagicMock()
    mock_batch.describe_jobs.return_value = {
        "jobs": [{"status": "FAILED", "statusReason": "OOM"}]
    }
    mock_boto3.client.return_value = mock_batch
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = wait_for_batch_job("job-123", "us-east-1", poll_interval=0, stream_logs=False)
        assert result == "FAILED"


def test_wait_for_batch_job_not_found():
    from run_aws_training import wait_for_batch_job
    mock_boto3 = MagicMock()
    mock_batch = MagicMock()
    mock_batch.describe_jobs.return_value = {"jobs": []}
    mock_boto3.client.return_value = mock_batch
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = wait_for_batch_job("job-123", "us-east-1", poll_interval=0, stream_logs=False)
        assert result == "FAILED"


def test_wait_for_batch_job_with_log_streaming():
    from run_aws_training import wait_for_batch_job
    mock_boto3 = MagicMock()
    mock_batch = MagicMock()
    mock_batch.describe_jobs.return_value = {
        "jobs": [{"status": "SUCCEEDED", "statusReason": "", "container": {"logStreamName": "stream1"}}]
    }
    mock_logs = MagicMock()
    mock_logs.get_log_events.return_value = {"events": [{"message": "hello"}], "nextForwardToken": "tok"}
    def client_factory(svc, **kw):
        if svc == "batch":
            return mock_batch
        return mock_logs
    mock_boto3.client.side_effect = client_factory
    with patch.dict("sys.modules", {"boto3": mock_boto3}):
        result = wait_for_batch_job("job-123", "us-east-1", poll_interval=0, stream_logs=True)
        assert result == "SUCCEEDED"


def test_stream_job_logs_no_stream_name():
    from run_aws_training import _stream_job_logs
    token, name = _stream_job_logs(MagicMock(), {"container": {}}, None, None, "us-east-1")
    assert name is None


def test_stream_job_logs_exception():
    from run_aws_training import _stream_job_logs
    mock_logs = MagicMock()
    mock_logs.get_log_events.side_effect = Exception("fail")
    token, name = _stream_job_logs(mock_logs, {}, "stream1", None, "us-east-1")
    assert name == "stream1"


def test_run_local_mode_dry_run():
    from run_aws_training import run_local_mode
    args = argparse.Namespace(dry_run=True, mode="local", skip_upload=False)
    mock_pc = MagicMock()
    mock_ec = MagicMock()
    with patch("run_aws_training.check_dataset_in_s3", return_value=True):
        run_local_mode(args, mock_pc, mock_ec)


def test_run_batch_mode_dry_run():
    from run_aws_training import run_batch_mode
    args = argparse.Namespace(dry_run=True, mode="batch", config="test")
    mock_pc = MagicMock()
    mock_ec = MagicMock()
    with patch("run_aws_training.check_dataset_in_s3", return_value=False):
        run_batch_mode(args, mock_pc, mock_ec)


def test_run_local_mode_full():
    from run_aws_training import run_local_mode
    args = argparse.Namespace(dry_run=False, mode="local", skip_upload=False)
    mock_pc = MagicMock()
    mock_ec = MagicMock()
    mock_oc = MagicMock()
    mock_oc.select.return_value = "/tmp/cache"
    with patch("run_aws_training.check_dataset_in_s3", return_value=True):
        with patch("run_aws_training.download_dataset_from_s3"):
            with patch("run_aws_training.run_training", return_value=("/tmp/out", {"loss": 0.5, "best_loss": 0.4})):
                with patch("run_aws_training.upload_outputs_to_s3"):
                    with patch.dict("sys.modules", {"omegaconf": MagicMock(OmegaConf=mock_oc)}):
                        run_local_mode(args, mock_pc, mock_ec)


def test_run_local_mode_skip_upload():
    from run_aws_training import run_local_mode
    args = argparse.Namespace(dry_run=False, mode="local", skip_upload=True)
    mock_pc = MagicMock()
    mock_ec = MagicMock()
    mock_oc = MagicMock()
    mock_oc.select.return_value = "/tmp/cache"
    with patch("run_aws_training.check_dataset_in_s3", return_value=True):
        with patch("run_aws_training.download_dataset_from_s3"):
            with patch("run_aws_training.run_training", return_value=("/tmp/out", {"loss": 0.5, "best_loss": 0.4})):
                with patch.dict("sys.modules", {"omegaconf": MagicMock(OmegaConf=mock_oc)}):
                    run_local_mode(args, mock_pc, mock_ec)


def test_run_local_mode_ingest_needed():
    from run_aws_training import run_local_mode
    args = argparse.Namespace(dry_run=False, mode="local", skip_upload=True)
    mock_pc = MagicMock()
    mock_ec = MagicMock()
    mock_oc = MagicMock()
    mock_oc.select.return_value = "/tmp/cache"
    with patch("run_aws_training.check_dataset_in_s3", return_value=False):
        with patch("run_aws_training.run_data_ingestion"):
            with patch("run_aws_training.download_dataset_from_s3"):
                with patch("run_aws_training.run_training", return_value=("/tmp/out", {"loss": 0.5, "best_loss": 0.4})):
                    with patch.dict("sys.modules", {"omegaconf": MagicMock(OmegaConf=mock_oc)}):
                        run_local_mode(args, mock_pc, mock_ec)


def test_run_batch_mode_full():
    from run_aws_training import run_batch_mode
    args = argparse.Namespace(dry_run=False, mode="batch", config="test", overrides=[])
    mock_pc = MagicMock()
    mock_pc.aws_region = "us-east-1"
    mock_ec = MagicMock()
    with patch("run_aws_training.check_dataset_in_s3", return_value=True):
        with patch("run_aws_training.submit_batch_job", return_value="job-123"):
            with patch("run_aws_training.wait_for_batch_job", return_value="SUCCEEDED"):
                with pytest.raises(SystemExit) as exc_info:
                    run_batch_mode(args, mock_pc, mock_ec)
                assert exc_info.value.code == 0


def test_run_batch_mode_failed():
    from run_aws_training import run_batch_mode
    args = argparse.Namespace(dry_run=False, mode="batch", config="test", overrides=[])
    mock_pc = MagicMock()
    mock_pc.aws_region = "us-east-1"
    mock_ec = MagicMock()
    with patch("run_aws_training.check_dataset_in_s3", return_value=True):
        with patch("run_aws_training.submit_batch_job", return_value="job-123"):
            with patch("run_aws_training.wait_for_batch_job", return_value="FAILED"):
                with pytest.raises(SystemExit) as exc_info:
                    run_batch_mode(args, mock_pc, mock_ec)
                assert exc_info.value.code == 1


def test_run_container_mode():
    from run_aws_training import run_container_mode
    args = argparse.Namespace(mode="container")
    mock_pc = MagicMock()
    mock_ec = MagicMock()
    mock_oc = MagicMock()
    mock_oc.select.return_value = "/tmp/cache"
    with patch.dict("sys.modules", {"omegaconf": MagicMock(OmegaConf=mock_oc)}):
        with patch("run_aws_training.download_dataset_from_s3"):
            with patch("run_aws_training.run_training", return_value=("/tmp/out", {"loss": 0.5, "best_loss": 0.4})):
                with patch("run_aws_training.upload_outputs_to_s3"):
                    run_container_mode(args, mock_pc, mock_ec)


def test_run_container_mode_training_fails():
    from run_aws_training import run_container_mode
    args = argparse.Namespace(mode="container")
    mock_pc = MagicMock()
    mock_ec = MagicMock()
    mock_oc = MagicMock()
    mock_oc.select.return_value = "/tmp/cache"
    with patch.dict("sys.modules", {"omegaconf": MagicMock(OmegaConf=mock_oc)}):
        with patch("run_aws_training.download_dataset_from_s3"):
            with patch("run_aws_training.run_training", side_effect=RuntimeError("OOM")):
                with patch("run_aws_training.upload_outputs_to_s3"):
                    with pytest.raises(RuntimeError):
                        run_container_mode(args, mock_pc, mock_ec)


def test_main_keyboard_interrupt():
    from run_aws_training import main
    with patch("run_aws_training.load_configs", side_effect=KeyboardInterrupt):
        with patch("sys.argv", ["run_aws_training.py", "--config", "test", "--mode", "local"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 130


def test_main_value_error():
    from run_aws_training import main
    with patch("run_aws_training.load_configs", side_effect=ValueError("bad")):
        with patch("sys.argv", ["run_aws_training.py", "--config", "test", "--mode", "local"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


def test_main_import_error():
    from run_aws_training import main
    with patch("run_aws_training.load_configs", side_effect=ImportError("missing")):
        with patch("sys.argv", ["run_aws_training.py", "--config", "test", "--mode", "local"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1


def test_main_generic_exception():
    from run_aws_training import main
    with patch("run_aws_training.load_configs", side_effect=RuntimeError("boom")):
        with patch("sys.argv", ["run_aws_training.py", "--config", "test", "--mode", "local"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2


def test_main_success():
    from run_aws_training import main
    mock_pc = MagicMock()
    mock_ec = MagicMock()
    with patch("run_aws_training.load_configs", return_value=(mock_pc, mock_ec)):
        with patch("run_aws_training.run_local_mode"):
            with patch("sys.argv", ["run_aws_training.py", "--config", "test", "--mode", "local"]):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 0


def test_load_configs():
    from run_aws_training import load_configs
    args = argparse.Namespace(config="test", overrides=[])
    mock_pc = MagicMock()
    mock_pc.raw_data_bucket = "bucket"
    mock_pc.dataset_name = "test"
    mock_ec = MagicMock()
    mock_ec.experiment_name = "test_exp"
    mock_infra = MagicMock()
    mock_infra.config.config.load_pipeline_config.return_value = mock_pc
    mock_src_utils = MagicMock()
    mock_src_utils.load_experiment_config.return_value = mock_ec
    mock_omegaconf_mod = MagicMock()
    with patch.dict("sys.modules", {
        "infra": mock_infra,
        "infra.config": mock_infra.config,
        "infra.config.config": mock_infra.config.config,
    }):
        with patch("src.utils.config_loader.load_experiment_config", return_value=mock_ec):
            with patch("omegaconf.OmegaConf.update"):
                pc, ec = load_configs(args)
                assert pc is mock_pc


def test_run_batch_mode_ingest_needed():
    """Covers line 562: run_data_ingestion called when dataset not in S3 in batch mode."""
    from run_aws_training import run_batch_mode
    import argparse
    args = argparse.Namespace(dry_run=False, mode="batch", config="test", overrides=[])
    mock_pc = MagicMock()
    mock_pc.aws_region = "us-east-1"
    mock_ec = MagicMock()
    with patch("run_aws_training.check_dataset_in_s3", return_value=False):
        with patch("run_aws_training.run_data_ingestion") as mock_ingest:
            with patch("run_aws_training.submit_batch_job", return_value="job-456"):
                with patch("run_aws_training.wait_for_batch_job", return_value="SUCCEEDED"):
                    with pytest.raises(SystemExit) as exc_info:
                        run_batch_mode(args, mock_pc, mock_ec)
                    assert exc_info.value.code == 0
                    mock_ingest.assert_called_once()
