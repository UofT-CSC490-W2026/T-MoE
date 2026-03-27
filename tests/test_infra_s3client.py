import pytest

from pathlib import Path

from unittest.mock import patch, MagicMock

                                                                             

          

                                                                             

def _fresh_client():

    
    mock_boto3 = MagicMock()

    mock_session = MagicMock()

    mock_s3 = MagicMock()

    mock_sts = MagicMock()

    mock_sts.get_caller_identity.return_value = {"Account": "123456789"}

    mock_session.client.side_effect = lambda svc, **kw: (

        mock_sts if svc == "sts" else mock_s3

    )

    mock_boto3.Session.return_value = mock_session

    import infra.s3client.client as mod

    with patch.object(mod, "boto3", mock_boto3):

        with patch.object(mod, "BotoConfig", MagicMock()):

            with patch.object(mod, "TransferConfig", MagicMock()):

                client = mod.S3Client(region="us-east-1")

                                                                  

    client._client = mock_s3

    return client, mock_s3

def test_s3client_init():

    client, mock_s3 = _fresh_client()

    assert client.region == "us-east-1"

def test_s3client_init_no_credentials():

    from botocore.exceptions import NoCredentialsError

    mock_boto3 = MagicMock()

    mock_session = MagicMock()

    mock_sts = MagicMock()

    mock_sts.get_caller_identity.side_effect = NoCredentialsError()

    mock_session.client.return_value = mock_sts

    mock_boto3.Session.return_value = mock_session

    import infra.s3client.client as mod

    with patch.object(mod, "boto3", mock_boto3):

        with patch.object(mod, "BotoConfig", MagicMock()):

            with patch.object(mod, "TransferConfig", MagicMock()):

                with pytest.raises(RuntimeError, match="AWS credentials"):

                    mod.S3Client(region="us-east-1")

def test_upload_file_not_found(tmp_path):

    client, mock_s3 = _fresh_client()

    result = client.upload_file(tmp_path / "nonexistent.txt", "bucket", "key")

    assert result is False

def test_upload_file_success(tmp_path):

    client, mock_s3 = _fresh_client()

    f = tmp_path / "data.txt"

    f.write_text("hello")

    result = client.upload_file(str(f), "bucket", "key/data.txt")

    assert result is True

    mock_s3.upload_file.assert_called_once()

def test_upload_file_no_progress(tmp_path):

    
    client, mock_s3 = _fresh_client()

    f = tmp_path / "data.txt"

    f.write_text("hello")

    result = client.upload_file(str(f), "bucket", "key/data.txt", show_progress=False)

    assert result is True

def test_download_file_success(tmp_path):

    client, mock_s3 = _fresh_client()

    dest = tmp_path / "out.txt"

    mock_s3.head_object.return_value = {"ContentLength": 5}

    def fake_download(Bucket, Key, Filename, Callback, Config):

        Path(Filename).write_text("hello")

    mock_s3.download_file.side_effect = fake_download

    result = client.download_file("bucket", "key", str(dest))

    assert result is True

def test_download_file_size_mismatch(tmp_path):

    client, mock_s3 = _fresh_client()

    dest = tmp_path / "out.txt"

    mock_s3.head_object.return_value = {"ContentLength": 999}

    def fake_download(Bucket, Key, Filename, Callback, Config):

        Path(Filename).write_text("hi")

    mock_s3.download_file.side_effect = fake_download

    result = client.download_file("bucket", "key", str(dest))

    assert result is False

def test_download_file_head_error(tmp_path):

    
    from botocore.exceptions import ClientError

    client, mock_s3 = _fresh_client()

    mock_s3.head_object.side_effect = ClientError(

        {"Error": {"Code": "404", "Message": "not found"}}, "HeadObject"

    )

    result = client.download_file("bucket", "key", str(tmp_path / "f"))

    assert result is False

def test_list_objects_success():

    client, mock_s3 = _fresh_client()

    mock_paginator = MagicMock()

    mock_s3.get_paginator.return_value = mock_paginator

    from datetime import datetime

    mock_paginator.paginate.return_value = [

        {

            "Contents": [

                {

                    "Key": "a/b.txt",

                    "Size": 10,

                    "LastModified": datetime(2024, 1, 1),

                    "ETag": "abc",

                }

            ]

        }

    ]

    results = client.list_objects("bucket", "a/")

    assert len(results) == 1

    assert results[0]["Key"] == "a/b.txt"

def test_list_objects_empty():

    client, mock_s3 = _fresh_client()

    mock_paginator = MagicMock()

    mock_s3.get_paginator.return_value = mock_paginator

    mock_paginator.paginate.return_value = [{}]

    results = client.list_objects("bucket", "a/")

    assert results == []

def test_list_objects_client_error():

    from botocore.exceptions import ClientError

    client, mock_s3 = _fresh_client()

    mock_paginator = MagicMock()

    mock_s3.get_paginator.return_value = mock_paginator

    mock_paginator.paginate.side_effect = ClientError(

        {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "ListObjects"

    )

    results = client.list_objects("bucket", "a/")

    assert results == []

def test_delete_objects_empty():

    client, mock_s3 = _fresh_client()

    result = client.delete_objects("bucket", [])

    assert result == {"deleted": [], "errors": []}

def test_delete_objects_success():

    client, mock_s3 = _fresh_client()

    mock_s3.delete_objects.return_value = {"Errors": []}

    result = client.delete_objects("bucket", ["key1", "key2"])

    assert len(result["errors"]) == 0

def test_delete_objects_client_error():

    from botocore.exceptions import ClientError

    client, mock_s3 = _fresh_client()

    mock_s3.delete_objects.side_effect = ClientError(

        {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "DeleteObjects"

    )

    result = client.delete_objects("bucket", ["key1"])

    assert len(result["errors"]) > 0

def test_check_bucket_exists_true():

    client, mock_s3 = _fresh_client()

    mock_s3.head_bucket.return_value = {}

    assert client.check_bucket_exists("bucket") is True

def test_check_bucket_exists_404():

    from botocore.exceptions import ClientError

    client, mock_s3 = _fresh_client()

    mock_s3.head_bucket.side_effect = ClientError(

        {"Error": {"Code": "404", "Message": "not found"}}, "HeadBucket"

    )

    assert client.check_bucket_exists("bucket") is False

def test_check_bucket_exists_403():

    from botocore.exceptions import ClientError

    client, mock_s3 = _fresh_client()

    mock_s3.head_bucket.side_effect = ClientError(

        {"Error": {"Code": "403", "Message": "forbidden"}}, "HeadBucket"

    )

    assert client.check_bucket_exists("bucket") is False

def test_generate_presigned_url_success():

    client, mock_s3 = _fresh_client()

    mock_s3.generate_presigned_url.return_value = "https://example.com/signed"

    url = client.generate_presigned_url("bucket", "key")

    assert url == "https://example.com/signed"

def test_generate_presigned_url_error():

    from botocore.exceptions import ClientError

    client, mock_s3 = _fresh_client()

    mock_s3.generate_presigned_url.side_effect = ClientError(

        {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GeneratePresignedUrl"

    )

    url = client.generate_presigned_url("bucket", "key")

    assert url is None

def test_dataset_exists_found():

    client, mock_s3 = _fresh_client()

    mock_paginator = MagicMock()

    mock_s3.get_paginator.return_value = mock_paginator

    from datetime import datetime

    mock_paginator.paginate.return_value = [

        {

            "Contents": [

                {

                    "Key": "prefix/data.jsonl",

                    "Size": 10,

                    "LastModified": datetime(2024, 1, 1),

                    "ETag": "abc",

                }

            ]

        }

    ]

    assert client.dataset_exists("bucket", "prefix/") is True

def test_dataset_exists_not_found():

    client, mock_s3 = _fresh_client()

    mock_paginator = MagicMock()

    mock_s3.get_paginator.return_value = mock_paginator

    from datetime import datetime

    mock_paginator.paginate.return_value = [

        {

            "Contents": [

                {

                    "Key": "prefix/README.md",

                    "Size": 10,

                    "LastModified": datetime(2024, 1, 1),

                    "ETag": "abc",

                }

            ]

        }

    ]

    assert client.dataset_exists("bucket", "prefix/") is False

def test_progress_callback():

    import infra.s3client.client as mod

    cb = mod.S3Client._progress_callback(100, "test")

    cb(25)

    cb(25)

    cb(25)

    cb(25)

def test_progress_callback_zero_total():

    import infra.s3client.client as mod

    cb = mod.S3Client._progress_callback(0, "test")

    cb(10)                    

                                                                             

         

                                                                             

def test_upload_experiment_dir_not_found():

    from infra.s3client.s3_sync import upload_experiment_dir

    with pytest.raises(FileNotFoundError):

        upload_experiment_dir("/nonexistent/dir", "bucket", "prefix/")

def test_upload_experiment_dir_bucket_inaccessible(tmp_path):

    mock_client = MagicMock()

    mock_client.check_bucket_exists.return_value = False

    mock_s3_mod = MagicMock()

    mock_s3_mod.S3Client.return_value = mock_client

    with patch.dict("sys.modules", {"infra.s3client.client": mock_s3_mod}):

        from infra.s3client import s3_sync

        import importlib

        importlib.reload(s3_sync)

        with pytest.raises(RuntimeError, match="inaccessible"):

            s3_sync.upload_experiment_dir(str(tmp_path), "bucket", "prefix/")

def test_upload_experiment_dir_success(tmp_path):

    (tmp_path / "model.pt").write_text("weights")

    (tmp_path / "config.yaml").write_text("cfg: 1")

    mock_client = MagicMock()

    mock_client.check_bucket_exists.return_value = True

    mock_client.upload_file.return_value = True

    mock_s3_mod = MagicMock()

    mock_s3_mod.S3Client.return_value = mock_client

    with patch.dict("sys.modules", {"infra.s3client.client": mock_s3_mod}):

        from infra.s3client import s3_sync

        import importlib

        importlib.reload(s3_sync)

        result = s3_sync.upload_experiment_dir(str(tmp_path), "bucket", "prefix/")

    assert len(result["uploaded"]) == 2

    assert result["failed"] == []

def test_upload_experiment_dir_partial_failure(tmp_path):

    (tmp_path / "model.pt").write_text("weights")

    mock_client = MagicMock()

    mock_client.check_bucket_exists.return_value = True

    mock_client.upload_file.return_value = False

    mock_s3_mod = MagicMock()

    mock_s3_mod.S3Client.return_value = mock_client

    with patch.dict("sys.modules", {"infra.s3client.client": mock_s3_mod}):

        from infra.s3client import s3_sync

        import importlib

        importlib.reload(s3_sync)

        result = s3_sync.upload_experiment_dir(str(tmp_path), "bucket", "prefix/")

    assert len(result["failed"]) == 1

def test_download_s3_prefix_no_objects(tmp_path):

    mock_client = MagicMock()

    mock_client.list_objects.return_value = []

    mock_s3_mod = MagicMock()

    mock_s3_mod.S3Client.return_value = mock_client

    with patch.dict("sys.modules", {"infra.s3client.client": mock_s3_mod}):

        from infra.s3client import s3_sync

        import importlib

        importlib.reload(s3_sync)

        result = s3_sync.download_s3_prefix("bucket", "prefix/", str(tmp_path))

    assert result == []

def test_download_s3_prefix_success(tmp_path):

    mock_client = MagicMock()

    mock_client.list_objects.return_value = [

        {"Key": "prefix/data.jsonl"},

        {"Key": "prefix/"},                                        

    ]

    mock_client.download_file.return_value = True

    mock_s3_mod = MagicMock()

    mock_s3_mod.S3Client.return_value = mock_client

    with patch.dict("sys.modules", {"infra.s3client.client": mock_s3_mod}):

        from infra.s3client import s3_sync

        import importlib

        importlib.reload(s3_sync)

        result = s3_sync.download_s3_prefix("bucket", "prefix/", str(tmp_path))

    assert len(result) == 1

def test_download_s3_prefix_download_failure(tmp_path):

    mock_client = MagicMock()

    mock_client.list_objects.return_value = [{"Key": "prefix/data.jsonl"}]

    mock_client.download_file.return_value = False

    mock_s3_mod = MagicMock()

    mock_s3_mod.S3Client.return_value = mock_client

    with patch.dict("sys.modules", {"infra.s3client.client": mock_s3_mod}):

        from infra.s3client import s3_sync

        import importlib

        importlib.reload(s3_sync)

        result = s3_sync.download_s3_prefix("bucket", "prefix/", str(tmp_path))

    assert result == []

def test_upload_file_client_error(tmp_path):

    
    from botocore.exceptions import ClientError

    client, mock_s3 = _fresh_client()

    f = tmp_path / "data.txt"

    f.write_text("hello")

    mock_s3.upload_file.side_effect = ClientError(

        {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "PutObject"

    )

    result = client.upload_file(str(f), "bucket", "key")

    assert result is False

def test_upload_file_with_metadata_and_content_type(tmp_path):

    
    client, mock_s3 = _fresh_client()

    f = tmp_path / "data.txt"

    f.write_text("hello")

    result = client.upload_file(

        str(f),

        "bucket",

        "key",

        metadata={"author": "test"},

        content_type="text/plain",

    )

    assert result is True

    call_kwargs = mock_s3.upload_file.call_args.kwargs

    assert call_kwargs["ExtraArgs"]["Metadata"] == {"author": "test"}

    assert call_kwargs["ExtraArgs"]["ContentType"] == "text/plain"

def test_download_file_client_error(tmp_path):

    
    from botocore.exceptions import ClientError

    client, mock_s3 = _fresh_client()

    mock_s3.head_object.return_value = {"ContentLength": 5}

    mock_s3.download_file.side_effect = ClientError(

        {"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetObject"

    )

    result = client.download_file("bucket", "key", str(tmp_path / "out.txt"))

    assert result is False

def test_check_bucket_exists_other_error():

    
    from botocore.exceptions import ClientError

    client, mock_s3 = _fresh_client()

    mock_s3.head_bucket.side_effect = ClientError(

        {"Error": {"Code": "500", "Message": "internal error"}}, "HeadBucket"

    )

    assert client.check_bucket_exists("bucket") is False

def test_s3client_init_with_endpoint_url():

    
    mock_boto3 = MagicMock()

    mock_session = MagicMock()

    mock_s3 = MagicMock()

    mock_sts = MagicMock()

    mock_sts.get_caller_identity.return_value = {"Account": "123456789"}

    mock_session.client.side_effect = lambda svc, **kw: (

        mock_sts if svc == "sts" else mock_s3

    )

    mock_boto3.Session.return_value = mock_session

    import infra.s3client.client as mod

    with patch.object(mod, "boto3", mock_boto3):

        with patch.object(mod, "BotoConfig", MagicMock()):

            with patch.object(mod, "TransferConfig", MagicMock()):

                client = mod.S3Client(

                    region="us-east-1", endpoint_url="http://localhost:9000"

                )

    assert client.endpoint_url == "http://localhost:9000"
