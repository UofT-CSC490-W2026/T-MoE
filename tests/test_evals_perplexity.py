import math

import pytest

import torch

from unittest.mock import MagicMock, patch

VOCAB = 256


def test_cfg_select_dict():
    from evals.perplexity import _cfg_select

    cfg = {"a": {"b": 42}}

    assert _cfg_select(cfg, "a.b") == 42

    assert _cfg_select(cfg, "a.c", default=99) == 99

    assert _cfg_select(cfg, "x.y", default="fallback") == "fallback"


def test_cfg_select_none_mid_path():
    from evals.perplexity import _cfg_select

    cfg = {"a": None}

    assert _cfg_select(cfg, "a.b", default=0) == 0


def test_dtype_name():
    from evals.perplexity import _dtype_name

    assert _dtype_name(torch.float32) == "float32"

    assert _dtype_name(torch.bfloat16) == "bfloat16"


def test_autocast_context_cpu():
    from evals.perplexity import _autocast_context

    ctx = _autocast_context("cpu", torch.float32)

    with ctx:
        pass


def test_autocast_context_cuda():
    from evals.perplexity import _autocast_context

    ctx = _autocast_context("cuda:0", torch.bfloat16)

    assert hasattr(ctx, "__enter__")


def test_summarize_language_model_metrics_basic():
    from evals.perplexity import summarize_language_model_metrics

    m = summarize_language_model_metrics(total_nll=10.0, total_tokens=10)

    assert abs(m["ppl"] - math.exp(1.0)) < 1e-5

    assert "bpb" not in m


def test_summarize_language_model_metrics_with_bpb():
    from evals.perplexity import summarize_language_model_metrics

    m = summarize_language_model_metrics(
        total_nll=10.0, total_tokens=10, total_bytes=100
    )

    assert "bpb" in m


def test_summarize_language_model_metrics_zero_tokens():
    from evals.perplexity import summarize_language_model_metrics

    with pytest.raises(ValueError, match="total_tokens must be positive"):
        summarize_language_model_metrics(total_nll=1.0, total_tokens=0)


def test_summarize_language_model_metrics_zero_bytes():
    from evals.perplexity import summarize_language_model_metrics

    with pytest.raises(ValueError, match="total_bytes must be positive"):
        summarize_language_model_metrics(total_nll=1.0, total_tokens=5, total_bytes=0)


def test_infer_eval_context_length_from_backbone():
    from evals.perplexity import infer_eval_context_length

    model = MagicMock()

    model.backbone.config.max_position_embeddings = 2048

    model.backbone.config.n_positions = None

    result = infer_eval_context_length(model, {})

    assert result == 2048


def test_infer_eval_context_length_fallback():
    from evals.perplexity import infer_eval_context_length

    model = MagicMock()

    model.backbone = None

    result = infer_eval_context_length(model, {"dataset": {"max_seq_len": 512}})

    assert result == 512


def test_infer_eval_context_length_tokenizer_limit():
    from evals.perplexity import infer_eval_context_length

    model = MagicMock()

    model.backbone.config.max_position_embeddings = 4096

    model.backbone.config.n_positions = None

    tokenizer = MagicMock()

    tokenizer.model_max_length = 1024

    result = infer_eval_context_length(model, {}, tokenizer)

    assert result == 1024


def test_infer_eval_context_length_min_2():
    from evals.perplexity import infer_eval_context_length

    model = MagicMock()

    model.backbone = None

    result = infer_eval_context_length(model, {})

    assert result >= 2


def test_document_windows_short_seq():
    from evals.perplexity import _document_windows

    ids = torch.zeros(1, 1, dtype=torch.long)

    windows = list(_document_windows(0, ids, stride=512, max_length=1024))

    assert windows == []


def test_document_windows_basic():
    from evals.perplexity import _document_windows

    ids = torch.arange(100).unsqueeze(0)

    windows = list(_document_windows(0, ids, stride=50, max_length=50))

    assert len(windows) > 0

    for w in windows:
        assert w.window_input.shape[0] == 1


def test_document_windows_single_window():
    from evals.perplexity import _document_windows

    ids = torch.arange(10).unsqueeze(0)

    windows = list(_document_windows(0, ids, stride=512, max_length=512))

    assert len(windows) == 1


def test_tokenize_worker_basic():
    import queue

    from evals.perplexity import _tokenize_worker

    tokenizer = MagicMock()

    tokenizer.return_value = {"input_ids": torch.zeros(1, 5, dtype=torch.long)}

    q = queue.Queue()

    _tokenize_worker(["hello world", "foo bar"], tokenizer, q, max_documents=None)

    items = []

    while True:
        item = q.get()

        if item is None:
            break

        items.append(item)

    assert len(items) == 2


def test_tokenize_worker_max_documents():
    import queue

    from evals.perplexity import _tokenize_worker

    tokenizer = MagicMock()

    tokenizer.return_value = {"input_ids": torch.zeros(1, 5, dtype=torch.long)}

    q = queue.Queue()

    _tokenize_worker(["a", "b", "c", "d"], tokenizer, q, max_documents=2)

    items = []

    while True:
        item = q.get()

        if item is None:
            break

        items.append(item)

    assert len(items) == 2


def test_tokenize_worker_skips_empty():
    import queue

    from evals.perplexity import _tokenize_worker

    tokenizer = MagicMock()

    tokenizer.return_value = {"input_ids": torch.zeros(1, 5, dtype=torch.long)}

    q = queue.Queue()

    _tokenize_worker(["", None, "valid"], tokenizer, q, max_documents=None)

    items = []

    while True:
        item = q.get()

        if item is None:
            break

        items.append(item)

    assert len(items) == 1


def test_tokenize_worker_type_error_fallback():
    import queue

    from evals.perplexity import _tokenize_worker

    call_count = [0]

    def tokenizer(text, **kwargs):
        call_count[0] += 1

        if "verbose" in kwargs:
            raise TypeError("no verbose")

        return {"input_ids": torch.zeros(1, 3, dtype=torch.long)}

    q = queue.Queue()

    _tokenize_worker(["hello"], tokenizer, q, max_documents=None)

    items = []

    while True:
        item = q.get()

        if item is None:
            break

        items.append(item)

    assert len(items) == 1


def _make_window(seq_len=10):
    from evals.perplexity import _Window

    ids = torch.zeros(1, seq_len, dtype=torch.long)

    mask = torch.ones(seq_len - 1, dtype=torch.bool)

    return _Window(doc_idx=0, window_input=ids, valid_mask=mask)


def _make_model_output(batch_size, seq_len, vocab=VOCAB):
    logits = torch.randn(batch_size, seq_len, vocab)

    out = MagicMock()

    out.logits = logits

    return out


def test_run_batched_forward_same_length():
    from evals.perplexity import _run_batched_forward

    w1 = _make_window(10)

    w2 = _make_window(10)

    model = MagicMock()

    model.return_value = _make_model_output(2, 10)

    results = _run_batched_forward(model, [w1, w2], "cpu", torch.float32)

    assert len(results) == 2

    losses, mask = results[0]

    assert losses.shape[0] == 9


def test_run_batched_forward_different_lengths():
    from evals.perplexity import _run_batched_forward

    w1 = _make_window(8)

    w2 = _make_window(12)

    model = MagicMock()

    model.return_value = _make_model_output(2, 12)

    results = _run_batched_forward(model, [w1, w2], "cpu", torch.float32)

    assert len(results) == 2


def test_run_batched_forward_tuple_output():
    from evals.perplexity import _run_batched_forward

    w = _make_window(10)

    model = MagicMock()

    logits = torch.randn(1, 10, VOCAB)

    model.return_value = (logits,)

    results = _run_batched_forward(model, [w], "cpu", torch.float32)

    assert len(results) == 1


def test_compute_document_nll_short_seq():
    from evals.perplexity import compute_document_nll

    model = MagicMock()

    ids = torch.zeros(1, 1, dtype=torch.long)

    nll, tokens = compute_document_nll(
        model, ids, stride=512, max_length=1024, device="cpu"
    )

    assert nll == 0.0

    assert tokens == 0


def test_compute_document_nll_basic():
    from evals.perplexity import compute_document_nll

    model = MagicMock()

    logits = torch.randn(1, 10, VOCAB)

    out = MagicMock()

    out.logits = logits

    model.return_value = out

    ids = torch.zeros(1, 10, dtype=torch.long)

    nll, tokens = compute_document_nll(
        model, ids, stride=10, max_length=10, device="cpu"
    )

    assert tokens > 0

    assert nll > 0


def test_compute_document_nll_invalid_shape():
    from evals.perplexity import compute_document_nll

    model = MagicMock()

    with pytest.raises(ValueError, match="shape"):
        compute_document_nll(
            model, torch.zeros(2, 5), stride=5, max_length=5, device="cpu"
        )


def test_compute_document_nll_invalid_stride():
    from evals.perplexity import compute_document_nll

    model = MagicMock()

    with pytest.raises(ValueError, match="stride"):
        compute_document_nll(
            model, torch.zeros(1, 5), stride=0, max_length=5, device="cpu"
        )


def test_compute_document_nll_invalid_max_length():
    from evals.perplexity import compute_document_nll

    model = MagicMock()

    with pytest.raises(ValueError, match="max_length"):
        compute_document_nll(
            model, torch.zeros(1, 5), stride=5, max_length=1, device="cpu"
        )


def test_compute_document_nll_tuple_output():
    from evals.perplexity import compute_document_nll

    model = MagicMock()

    logits = torch.randn(1, 10, VOCAB)

    model.return_value = (logits,)

    ids = torch.zeros(1, 10, dtype=torch.long)

    nll, tokens = compute_document_nll(
        model, ids, stride=10, max_length=10, device="cpu"
    )

    assert tokens > 0


def test_evaluate_text_documents_basic():
    from evals.perplexity import evaluate_text_documents

    tokenizer = MagicMock()

    tokenizer.return_value = {"input_ids": torch.arange(20).unsqueeze(0)}

    def model_fn(input_ids, **kwargs):
        B, S = input_ids.shape

        logits = torch.randn(B, S, 50)

        out = MagicMock()

        out.logits = logits

        return out

    result = evaluate_text_documents(
        model_fn,
        tokenizer,
        ["hello world " * 10],
        stride=10,
        max_length=20,
        device="cpu",
        include_bpb=True,
        batch_size=1,
    )

    assert "ppl" in result

    assert "bpb" in result


def test_evaluate_text_documents_no_tokens():
    from evals.perplexity import evaluate_text_documents

    tokenizer = MagicMock()

    tokenizer.return_value = {"input_ids": torch.zeros(1, 1, dtype=torch.long)}

    model = MagicMock()

    with pytest.raises(ValueError, match="No tokens scored"):
        evaluate_text_documents(
            model,
            tokenizer,
            ["x"],
            stride=512,
            max_length=512,
            device="cpu",
            include_bpb=False,
            batch_size=1,
        )


def test_load_dataset_texts_streaming():
    from evals.perplexity import _load_dataset_texts

    mock_ds = [{"text": "hello"}, {"text": "world"}]

    mock_datasets = MagicMock()

    mock_datasets.load_dataset.return_value = mock_ds

    spec = {
        "hf_path": "fake/path",
        "split": "test",
        "streaming": True,
        "text_column": "text",
    }

    with patch.dict("sys.modules", {"datasets": mock_datasets}):
        gen, hint = _load_dataset_texts(spec, max_documents=1)

        texts = list(gen)

    assert texts == ["hello"]


def test_load_dataset_texts_non_streaming():
    from evals.perplexity import _load_dataset_texts

    mock_ds = MagicMock()

    mock_ds.__iter__ = MagicMock(return_value=iter([{"text": "a"}, {"text": "b"}]))

    mock_ds.__len__ = MagicMock(return_value=2)

    mock_datasets = MagicMock()

    mock_datasets.load_dataset.return_value = mock_ds

    spec = {
        "hf_path": "fake/path",
        "split": "test",
        "streaming": False,
        "text_column": "text",
    }

    with patch.dict("sys.modules", {"datasets": mock_datasets}):
        gen, hint = _load_dataset_texts(spec)

        texts = list(gen)

    assert len(texts) == 2

    assert hint == 2


def test_load_dataset_texts_missing_column():
    from evals.perplexity import _load_dataset_texts

    mock_ds = [{"other": "hello"}]

    mock_datasets = MagicMock()

    mock_datasets.load_dataset.return_value = mock_ds

    spec = {
        "hf_path": "fake/path",
        "split": "test",
        "streaming": True,
        "text_column": "text",
    }

    with patch.dict("sys.modules", {"datasets": mock_datasets}):
        gen, _ = _load_dataset_texts(spec)

        texts = list(gen)

    assert texts == []


def test_load_tokenizer_for_model():
    from evals.perplexity import _load_tokenizer_for_model

    mock_transformers = MagicMock()

    mock_tok = MagicMock()

    mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tok

    mock_model_lookup = MagicMock(return_value={"hf_name": "EleutherAI/gpt-neo-125m"})

    with patch.dict("sys.modules", {"transformers": mock_transformers}):
        with patch("evals.perplexity.model_lookup", mock_model_lookup):
            tok = _load_tokenizer_for_model({"model": {"model_key": "gpt-neo-125m"}})

    assert tok is mock_tok


def _write_val_shard(path, tokens):
    import struct

    import numpy as np

    arr = np.array(tokens, dtype=np.uint16)

    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(tokens)))

        f.write(struct.pack("<H", 0))

        f.write(arr.tobytes())


def test_evaluate_token_shards_no_shards(tmp_path):
    from evals.perplexity import evaluate_token_shards

    model = MagicMock()

    with pytest.raises(FileNotFoundError, match="No val_shard"):
        evaluate_token_shards(
            model,
            tmp_path,
            stride=4,
            max_length=8,
            device="cpu",
            autocast_dtype=torch.float32,
            batch_size=2,
        )


def test_evaluate_token_shards_basic(tmp_path):
    from evals.perplexity import evaluate_token_shards

    tokens = [i % VOCAB for i in range(100)]

    _write_val_shard(tmp_path / "val_shard_0000.bin", tokens)

    def model_fn(input_ids, **kwargs):
        B, S = input_ids.shape

        out = MagicMock()

        out.logits = torch.randn(B, S, VOCAB)

        return out

    result = evaluate_token_shards(
        model_fn,
        tmp_path,
        stride=8,
        max_length=8,
        device="cpu",
        autocast_dtype=torch.float32,
        batch_size=4,
    )

    assert "ppl" in result

    assert result["tokens_scored"] > 0


def test_evaluate_token_shards_max_tokens(tmp_path):
    from evals.perplexity import evaluate_token_shards

    tokens = [i % VOCAB for i in range(200)]

    _write_val_shard(tmp_path / "val_shard_0000.bin", tokens)

    def model_fn(input_ids, **kwargs):
        B, S = input_ids.shape

        out = MagicMock()

        out.logits = torch.randn(B, S, VOCAB)

        return out

    result = evaluate_token_shards(
        model_fn,
        tmp_path,
        stride=8,
        max_length=8,
        device="cpu",
        autocast_dtype=torch.float32,
        batch_size=4,
        max_tokens=50,
    )

    assert "ppl" in result


def test_evaluate_token_shards_uint32(tmp_path):
    import struct

    import numpy as np

    from evals.perplexity import evaluate_token_shards

    tokens = [i % VOCAB for i in range(100)]

    arr = np.array(tokens, dtype=np.uint32)

    with open(tmp_path / "val_shard_0000.bin", "wb") as f:
        f.write(struct.pack("<Q", len(tokens)))

        f.write(struct.pack("<H", 1))

        f.write(arr.tobytes())

    def model_fn(input_ids, **kwargs):
        B, S = input_ids.shape

        out = MagicMock()

        out.logits = torch.randn(B, S, VOCAB)

        return out

    result = evaluate_token_shards(
        model_fn,
        tmp_path,
        stride=8,
        max_length=8,
        device="cpu",
        autocast_dtype=torch.float32,
        batch_size=4,
    )

    assert "ppl" in result


def test_evaluate_token_shards_legacy_shard(tmp_path):
    import struct

    import numpy as np

    from evals.perplexity import evaluate_token_shards

    tokens = [i % VOCAB for i in range(100)]

    arr = np.array(tokens, dtype=np.uint16)

    with open(tmp_path / "val_shard_0000.bin", "wb") as f:
        f.write(struct.pack("<Q", len(tokens)))

        f.write(arr.tobytes())

    def model_fn(input_ids, **kwargs):
        B, S = input_ids.shape

        out = MagicMock()

        out.logits = torch.randn(B, S, VOCAB)

        return out

    result = evaluate_token_shards(
        model_fn,
        tmp_path,
        stride=8,
        max_length=8,
        device="cpu",
        autocast_dtype=torch.float32,
        batch_size=4,
    )

    assert "ppl" in result


def test_evaluate_token_shards_no_tokens_scored(tmp_path):
    from evals.perplexity import evaluate_token_shards

    import struct

    import numpy as np

    arr = np.array([42], dtype=np.uint16)

    with open(tmp_path / "val_shard_0000.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))

        f.write(struct.pack("<H", 0))

        f.write(arr.tobytes())

    model = MagicMock()

    with pytest.raises(ValueError, match="No tokens scored"):
        evaluate_token_shards(
            model,
            tmp_path,
            stride=8,
            max_length=8,
            device="cpu",
            autocast_dtype=torch.float32,
            batch_size=4,
        )


def test_load_dataset_texts_len_type_error():
    from evals.perplexity import _load_dataset_texts

    mock_ds = MagicMock()

    mock_ds.__iter__ = MagicMock(return_value=iter([{"text": "a"}]))

    mock_ds.__len__ = MagicMock(side_effect=TypeError("no len"))

    mock_datasets = MagicMock()

    mock_datasets.load_dataset.return_value = mock_ds

    spec = {
        "hf_path": "fake/path",
        "split": "test",
        "streaming": False,
        "text_column": "text",
    }

    with patch.dict("sys.modules", {"datasets": mock_datasets}):
        gen, hint = _load_dataset_texts(spec)

    assert hint is None


def test_evaluate_text_documents_multiple_docs():
    from evals.perplexity import evaluate_text_documents

    tokenizer = MagicMock()

    tokenizer.return_value = {"input_ids": torch.zeros(1, 20, dtype=torch.long)}

    def model_fn(input_ids, **kwargs):
        B, S = input_ids.shape

        out = MagicMock()

        out.logits = torch.randn(B, S, VOCAB)

        return out

    result = evaluate_text_documents(
        model_fn,
        tokenizer,
        ["doc one " * 5, "doc two " * 5],
        stride=10,
        max_length=20,
        device="cpu",
        include_bpb=False,
        batch_size=2,
    )

    assert "ppl" in result

    assert "bpb" not in result
