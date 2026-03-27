"""Coverage tests for evals/perplexity.py — targeting uncovered lines."""
from __future__ import annotations

import math
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
    _document_windows,
    _run_batched_forward,
    _tokenize_worker,
    _Window,
    compute_document_nll,
    evaluate_text_documents,
    evaluate_token_shards,
    infer_eval_context_length,
    summarize_language_model_metrics,
)


# ── helpers ────────────────────────────────────────────────────────────────────

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
    """Write a minimal val_shard_*.bin file."""
    n = len(tokens)
    if uint32:
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", n))
            f.write(struct.pack("<H", 1))  # dtype_flag=1 → uint32
            f.write(np.array(tokens, dtype=np.uint32).tobytes())
    else:
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", n))
            f.write(np.array(tokens, dtype=np.uint16).tobytes())


# ── line 46: _cfg_select else branch (no .get on plain object) ────────────────

def test_cfg_select_non_dict_returns_default():
    # config is a plain object without .get → hits the else: return default branch
    result = _cfg_select(object(), "model.key", default="fallback")
    assert result == "fallback"


# ── line 118: _autocast_context non-cuda path ─────────────────────────────────

def test_autocast_context_cpu_returns_nullcontext():
    from contextlib import nullcontext
    ctx = _autocast_context("cpu", torch.float32)
    assert isinstance(ctx, type(nullcontext()))


# ── line 190: compute_document_nll seq_len < 2 ────────────────────────────────

def test_compute_document_nll_short_sequence():
    model = _SimpleModel()
    input_ids = torch.tensor([[5]], dtype=torch.long)  # seq_len=1
    nll, tokens = compute_document_nll(model, input_ids, stride=1, max_length=4, device="cpu")
    assert nll == 0.0
    assert tokens == 0


# ── line 245: compute_document_nll invalid input shape ────────────────────────

def test_compute_document_nll_bad_shape_raises():
    model = _SimpleModel()
    with pytest.raises(ValueError, match="shape"):
        compute_document_nll(model, torch.zeros(2, 5, dtype=torch.long), stride=1, max_length=4, device="cpu")


def test_compute_document_nll_bad_stride_raises():
    model = _SimpleModel()
    with pytest.raises(ValueError, match="stride"):
        compute_document_nll(model, torch.zeros(1, 5, dtype=torch.long), stride=0, max_length=4, device="cpu")


def test_compute_document_nll_bad_max_length_raises():
    model = _SimpleModel()
    with pytest.raises(ValueError, match="max_length"):
        compute_document_nll(model, torch.zeros(1, 5, dtype=torch.long), stride=1, max_length=1, device="cpu")


# ── line 265: summarize_language_model_metrics total_bytes <= 0 ───────────────

def test_summarize_metrics_zero_tokens_raises():
    with pytest.raises(ValueError, match="total_tokens"):
        summarize_language_model_metrics(total_nll=1.0, total_tokens=0)


def test_summarize_metrics_zero_bytes_raises():
    with pytest.raises(ValueError, match="total_bytes"):
        summarize_language_model_metrics(total_nll=1.0, total_tokens=5, total_bytes=0)


# ── line 344: _tokenize_worker TypeError fallback ─────────────────────────────

def test_tokenize_worker_typeerror_fallback():
    """Tokenizer that raises TypeError on verbose= kwarg → fallback branch."""
    def _tok(text, add_special_tokens=False, return_tensors="pt", **kwargs):
        if "verbose" in kwargs:
            raise TypeError("unexpected kwarg")
        ids = [0, 1, 2]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    out_q: queue.Queue = queue.Queue()
    _tokenize_worker(["hello world"], _tok, out_q, max_documents=None)
    item = out_q.get()
    assert item is not None  # got a result
    out_q.get()  # sentinel None


# ── lines 384-390: _run_batched_forward mixed-length padding path ─────────────

def test_run_batched_forward_mixed_lengths():
    """Windows of different lengths trigger the padding branch."""
    model = _SimpleModel(vocab_size=16)

    def _forward(input_ids):
        b, s = input_ids.shape
        return (torch.zeros(b, s, 16),)

    model.forward = _forward

    w1 = _Window(0, torch.zeros(1, 4, dtype=torch.long), torch.ones(3, dtype=torch.bool))
    w2 = _Window(1, torch.zeros(1, 6, dtype=torch.long), torch.ones(5, dtype=torch.bool))
    results = _run_batched_forward(model, [w1, w2], "cpu", torch.float32)
    assert len(results) == 2


# ── line 484: evaluate_text_documents no tokens scored ────────────────────────

def test_evaluate_text_documents_empty_raises():
    """All docs too short → no tokens scored → ValueError."""
    tokenizer = MagicMock()
    tokenizer.side_effect = lambda text, **kw: {"input_ids": torch.zeros(1, 1, dtype=torch.long)}
    model = _SimpleModel()
    with pytest.raises(ValueError, match="No tokens scored"):
        evaluate_text_documents(
            model, tokenizer, ["x"],
            stride=1, max_length=4, device="cpu",
        )


# ── line 508: evaluate_token_shards no shard files ────────────────────────────

def test_evaluate_token_shards_no_files_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _SimpleModel()
        with pytest.raises(FileNotFoundError):
            evaluate_token_shards(
                model, tmpdir,
                stride=2, max_length=4, device="cpu",
                autocast_dtype=torch.float32, batch_size=4,
            )


# ── lines 521-529: evaluate_token_shards distributed all-reduce ───────────────

def test_evaluate_token_shards_world_size_gt1():
    """world_size > 1 triggers the dist.all_reduce branch (mocked)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        shard = Path(tmpdir) / "val_shard_0000.bin"
        _make_shard(shard, [i % 16 for i in range(20)])

        model = _SimpleModel(vocab_size=16)

        import torch.distributed as dist_mod

        with patch.object(dist_mod, "all_reduce", return_value=None), \
             patch.object(dist_mod, "ReduceOp") as mock_op:
            mock_op.SUM = 0
            result = evaluate_token_shards(
                model, tmpdir,
                stride=2, max_length=4, device="cpu",
                autocast_dtype=torch.float32, batch_size=4,
                world_size=2, rank=0,
            )
        assert "ppl" in result


# ── line 571: _load_dataset_texts TypeError from len() ────────────────────────

def test_load_dataset_texts_len_typeerror():
    """Dataset that raises TypeError on len() → total_hint = None."""
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


# ── lines 615-711: run_perplexity_eval ────────────────────────────────────────

def _make_mock_model(vocab_size=16):
    """A real nn.Module that returns (logits,) — works with _run_batched_forward."""
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

    with patch("evals.perplexity._load_tokenizer_for_model", return_value=tokenizer), \
         patch("evals.perplexity.get_shard_dir", return_value=tmp_path), \
         patch("evals.perplexity.build_results_payload", return_value={"task": "perplexity", "results": {}}), \
         patch("evals.perplexity.write_results_json"):

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
