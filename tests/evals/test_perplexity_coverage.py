from __future__ import annotations

import queue

import struct

import tempfile

from pathlib import Path

from unittest.mock import MagicMock, patch

import numpy as np

import pytest

import torch

from evals.perplexity import (
    _autocast_context,
    _cfg_select,
    _run_batched_forward,
    _tokenize_worker,
    _Window,
    compute_document_nll,
    evaluate_text_documents,
    evaluate_token_shards,
    summarize_language_model_metrics,
)


class _SimpleModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 16):

        super().__init__()

        self.vocab_size = vocab_size

    def forward(self, input_ids):

        b, s = input_ids.shape

        return torch.zeros(b, s, self.vocab_size), None


class _SimpleTokenizer:
    model_max_length = 128

    def __call__(self, text, add_special_tokens=False, return_tensors="pt", **kwargs):

        ids = [ord(c) % 16 for c in text.split()]

        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


def _make_shard(path: Path, tokens: list[int], uint32: bool = False) -> None:

    n = len(tokens)

    if uint32:
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", n))

            f.write(struct.pack("<H", 1))

            f.write(np.array(tokens, dtype=np.uint32).tobytes())

    else:
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", n))

            f.write(np.array(tokens, dtype=np.uint16).tobytes())


def test_cfg_select_non_dict_returns_default():

    result = _cfg_select(object(), "model.key", default="fallback")

    assert result == "fallback"


def test_autocast_context_cpu_returns_nullcontext():

    from contextlib import nullcontext

    ctx = _autocast_context("cpu", torch.float32)

    assert isinstance(ctx, type(nullcontext()))


def test_compute_document_nll_short_sequence():

    model = _SimpleModel()

    input_ids = torch.tensor([[5]], dtype=torch.long)

    nll, tokens = compute_document_nll(
        model, input_ids, stride=1, max_length=4, device="cpu"
    )

    assert nll == 0.0

    assert tokens == 0


def test_compute_document_nll_bad_shape_raises():

    model = _SimpleModel()

    with pytest.raises(ValueError, match="shape"):
        compute_document_nll(
            model,
            torch.zeros(2, 5, dtype=torch.long),
            stride=1,
            max_length=4,
            device="cpu",
        )


def test_compute_document_nll_bad_stride_raises():

    model = _SimpleModel()

    with pytest.raises(ValueError, match="stride"):
        compute_document_nll(
            model,
            torch.zeros(1, 5, dtype=torch.long),
            stride=0,
            max_length=4,
            device="cpu",
        )


def test_compute_document_nll_bad_max_length_raises():

    model = _SimpleModel()

    with pytest.raises(ValueError, match="max_length"):
        compute_document_nll(
            model,
            torch.zeros(1, 5, dtype=torch.long),
            stride=1,
            max_length=1,
            device="cpu",
        )


def test_summarize_metrics_zero_tokens_raises():

    with pytest.raises(ValueError, match="total_tokens"):
        summarize_language_model_metrics(total_nll=1.0, total_tokens=0)


def test_summarize_metrics_zero_bytes_raises():

    with pytest.raises(ValueError, match="total_bytes"):
        summarize_language_model_metrics(total_nll=1.0, total_tokens=5, total_bytes=0)


def test_tokenize_worker_typeerror_fallback():

    def _tok(text, add_special_tokens=False, return_tensors="pt", **kwargs):

        if "verbose" in kwargs:
            raise TypeError("unexpected kwarg")

        ids = [0, 1, 2]

        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    out_q: queue.Queue = queue.Queue()

    _tokenize_worker(["hello world"], _tok, out_q, max_documents=None)

    item = out_q.get()

    assert item is not None

    out_q.get()


def test_run_batched_forward_mixed_lengths():

    model = _SimpleModel(vocab_size=16)

    def _forward(input_ids):

        b, s = input_ids.shape

        return (torch.zeros(b, s, 16),)

    model.forward = _forward

    w1 = _Window(
        0, torch.zeros(1, 4, dtype=torch.long), torch.ones(3, dtype=torch.bool)
    )

    w2 = _Window(
        1, torch.zeros(1, 6, dtype=torch.long), torch.ones(5, dtype=torch.bool)
    )

    results = _run_batched_forward(model, [w1, w2], "cpu", torch.float32)

    assert len(results) == 2


def test_evaluate_text_documents_empty_raises():

    tokenizer = MagicMock()

    tokenizer.side_effect = lambda text, **kw: {
        "input_ids": torch.zeros(1, 1, dtype=torch.long)
    }

    model = _SimpleModel()

    with pytest.raises(ValueError, match="No tokens scored"):
        evaluate_text_documents(
            model,
            tokenizer,
            ["x"],
            stride=1,
            max_length=4,
            device="cpu",
        )


def test_evaluate_token_shards_no_files_raises():

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _SimpleModel()

        with pytest.raises(FileNotFoundError):
            evaluate_token_shards(
                model,
                tmpdir,
                stride=2,
                max_length=4,
                device="cpu",
                autocast_dtype=torch.float32,
                batch_size=4,
            )


def test_evaluate_token_shards_world_size_gt1():

    with tempfile.TemporaryDirectory() as tmpdir:
        shard = Path(tmpdir) / "val_shard_0000.bin"

        _make_shard(shard, [i % 16 for i in range(20)])

        model = _SimpleModel(vocab_size=16)

        import torch.distributed as dist_mod

        with (
            patch.object(dist_mod, "all_reduce", return_value=None),
            patch.object(dist_mod, "ReduceOp") as mock_op,
        ):
            mock_op.SUM = 0

            result = evaluate_token_shards(
                model,
                tmpdir,
                stride=2,
                max_length=4,
                device="cpu",
                autocast_dtype=torch.float32,
                batch_size=4,
                world_size=2,
                rank=0,
            )

        assert "ppl" in result


def test_load_dataset_texts_len_typeerror():

    from evals.perplexity import _load_dataset_texts

    class _NoLen:
        def __len__(self):

            raise TypeError("no len")

        def __iter__(self):

            yield {"text": "hello world"}

    mock_ds = _NoLen()

    spec = {"hf_path": "fake", "split": "test", "streaming": False}

    with patch("datasets.load_dataset", return_value=mock_ds):
        gen, hint = _load_dataset_texts(spec)

        assert hint is None

        texts = list(gen)

        assert texts == ["hello world"]


def _make_mock_model(vocab_size=16):

    class _M(torch.nn.Module):
        def __init__(self):

            super().__init__()

            self.backbone = MagicMock()

            self.backbone.config = MagicMock()

            self.backbone.config.max_position_embeddings = 64

            self.backbone.config.n_positions = None

        def forward(self, input_ids):

            b, s = input_ids.shape

            return (torch.zeros(b, s, vocab_size),)

    return _M()


def test_run_perplexity_eval_with_shard_source(tmp_path):

    from evals.perplexity import run_perplexity_eval

    shard = tmp_path / "val_shard_0000.bin"

    _make_shard(shard, [i % 16 for i in range(30)])

    model = _make_mock_model()

    tokenizer = MagicMock()

    tokenizer.model_max_length = 64

    config = MagicMock()

    config.get = MagicMock(return_value=None)

    dataset_specs = [
        {
            "result_prefix": "shards_test",
            "source": "shards",
            "dataset_key": "wikitext-103",
            "include_bpb": False,
        }
    ]

    with (
        patch("evals.perplexity._load_tokenizer_for_model", return_value=tokenizer),
        patch("evals.perplexity.get_shard_dir", return_value=tmp_path),
        patch(
            "evals.perplexity.build_results_payload",
            return_value={"task": "perplexity", "results": {}},
        ),
        patch("evals.perplexity.write_results_json"),
    ):
        result = run_perplexity_eval(
            config=config,
            checkpoint_path="fake.pt",
            model=model,
            device="cpu",
            dataset_specs=dataset_specs,
            autocast_dtype=torch.float32,
            batch_size=4,
        )

        assert result is not None
