from __future__ import annotations

import struct

from unittest.mock import patch, MagicMock

import numpy as np


def _make_streaming_dataset(examples):

    from datasets import IterableDataset

    class _FakeIterableDataset(IterableDataset):
        def __init__(self):

            pass

        def __iter__(self):

            return iter(examples)

    return _FakeIterableDataset()


def test_iter_token_arrays_streaming_basic():

    from scripts.prepare_data import _iter_token_arrays

    examples = [{"text": "hello world"}, {"text": "  "}, {"text": "foo bar"}]

    dataset = _make_streaming_dataset(examples)

    mock_tok = MagicMock()

    mock_tok.return_value = {"input_ids": [[1, 2, 3], [4, 5]]}

    with patch("transformers.AutoTokenizer") as MockTok:
        MockTok.from_pretrained.return_value = mock_tok

        results = list(
            _iter_token_arrays(
                dataset,
                text_column="text",
                tokenizer_name="gpt-neo-125m",
                eos_id=50256,
                num_proc=1,
                vocab_size=50257,
                batch_size=2,
            )
        )

    assert len(results) > 0

    assert all(isinstance(r, np.ndarray) for r in results)


def test_iter_token_arrays_streaming_flush_remainder():

    from scripts.prepare_data import _iter_token_arrays

    examples = [{"text": "hello world"}]

    dataset = _make_streaming_dataset(examples)

    mock_tok = MagicMock()

    mock_tok.return_value = {"input_ids": [[1, 2, 3]]}

    with patch("transformers.AutoTokenizer") as MockTok:
        MockTok.from_pretrained.return_value = mock_tok

        results = list(
            _iter_token_arrays(
                dataset,
                text_column="text",
                tokenizer_name="gpt-neo-125m",
                eos_id=50256,
                num_proc=1,
                vocab_size=50257,
                batch_size=2048,
            )
        )

    assert len(results) == 1


def test_iter_token_arrays_streaming_uint32_vocab():

    from scripts.prepare_data import _iter_token_arrays

    examples = [{"text": "hello"}]

    dataset = _make_streaming_dataset(examples)

    mock_tok = MagicMock()

    mock_tok.return_value = {"input_ids": [[1, 2]]}

    with patch("transformers.AutoTokenizer") as MockTok:
        MockTok.from_pretrained.return_value = mock_tok

        results = list(
            _iter_token_arrays(
                dataset,
                text_column="text",
                tokenizer_name="qwen2",
                eos_id=151643,
                num_proc=1,
                vocab_size=151936,
                batch_size=2048,
            )
        )

    assert results[0].dtype == np.uint32


def test_iter_token_arrays_non_streaming():

    from scripts.prepare_data import _iter_token_arrays

    class _FakeDataset:
        def __iter__(self):

            yield {"text": "hello world"}

            yield {"text": "foo bar baz"}

    mock_pool = MagicMock()

    mock_pool.__enter__ = MagicMock(return_value=mock_pool)

    mock_pool.__exit__ = MagicMock(return_value=False)

    mock_pool.imap.return_value = iter([[[1, 2, 3, 50256], [4, 5, 50256]]])

    with patch("multiprocessing.Pool", return_value=mock_pool):
        results = list(
            _iter_token_arrays(
                _FakeDataset(),
                text_column="text",
                tokenizer_name="gpt-neo-125m",
                eos_id=50256,
                num_proc=1,
                vocab_size=50257,
                batch_size=1,
            )
        )

    assert len(results) == 2


def _make_cfg(dataset_key="wikitext-2", model_key="gpt-neo-125m"):

    cfg = MagicMock()

    cfg.dataset.dataset_key = dataset_key

    cfg.model.model_key = model_key

    return cfg


def test_tokenize_and_pack_basic(tmp_path):

    from scripts.prepare_data import tokenize_and_pack

    cfg = _make_cfg()

    dataset_info = {
        "hf_path": "wikitext",
        "hf_name": "wikitext-2-raw-v1",
        "text_column": "text",
        "streaming": False,
        "splits": {"train": "train", "val": "validation"},
    }

    mock_tok = MagicMock()

    mock_tok.name_or_path = "gpt-neo-125m"

    mock_tok.vocab_size = 50257

    mock_tok.eos_token_id = 50256

    token_arrays = [np.array([1, 2, 3, 50256], dtype=np.uint16)]

    with (
        patch("scripts.prepare_data.get_dataset_info", return_value=dataset_info),
        patch("scripts.prepare_data.get_tokenizer", return_value=(mock_tok, 50256)),
        patch(
            "scripts.prepare_data._iter_token_arrays", return_value=iter(token_arrays)
        ),
        patch("datasets.load_dataset", return_value=MagicMock()),
    ):
        tokenize_and_pack(cfg, tmp_path, num_proc=1)

    shards = list(tmp_path.glob("*.bin"))

    assert len(shards) >= 1


def test_tokenize_and_pack_skips_none_split(tmp_path):

    from scripts.prepare_data import tokenize_and_pack

    cfg = _make_cfg()

    dataset_info = {
        "hf_path": "wikitext",
        "hf_name": None,
        "text_column": "text",
        "streaming": False,
        "splits": {"train": "train", "val": None},
    }

    mock_tok = MagicMock()

    mock_tok.name_or_path = "gpt-neo-125m"

    mock_tok.vocab_size = 50257

    mock_tok.eos_token_id = 50256

    token_arrays = [np.array([1, 2, 3], dtype=np.uint16)]

    with (
        patch("scripts.prepare_data.get_dataset_info", return_value=dataset_info),
        patch("scripts.prepare_data.get_tokenizer", return_value=(mock_tok, 50256)),
        patch(
            "scripts.prepare_data._iter_token_arrays", return_value=iter(token_arrays)
        ),
        patch("datasets.load_dataset", return_value=MagicMock()),
    ):
        tokenize_and_pack(cfg, tmp_path, num_proc=1)

    shards = list(tmp_path.glob("train_*.bin"))

    assert len(shards) >= 1

    val_shards = list(tmp_path.glob("val_*.bin"))

    assert len(val_shards) == 0


def test_tokenize_and_pack_multiple_shards(tmp_path):

    from scripts.prepare_data import tokenize_and_pack, SHARD_SIZE

    cfg = _make_cfg()

    dataset_info = {
        "hf_path": "wikitext",
        "hf_name": None,
        "text_column": "text",
        "streaming": False,
        "splits": {"train": "train"},
    }

    mock_tok = MagicMock()

    mock_tok.name_or_path = "gpt-neo-125m"

    mock_tok.vocab_size = 50257

    mock_tok.eos_token_id = 50256

    big_array = np.ones(SHARD_SIZE + 10, dtype=np.uint16)

    with (
        patch("scripts.prepare_data.get_dataset_info", return_value=dataset_info),
        patch("scripts.prepare_data.get_tokenizer", return_value=(mock_tok, 50256)),
        patch(
            "scripts.prepare_data._iter_token_arrays", return_value=iter([big_array])
        ),
        patch("datasets.load_dataset", return_value=MagicMock()),
    ):
        tokenize_and_pack(cfg, tmp_path, num_proc=1)

    shards = sorted(tmp_path.glob("train_*.bin"))

    assert len(shards) == 2


def test_tokenize_and_pack_uint32_vocab(tmp_path):

    from scripts.prepare_data import tokenize_and_pack

    cfg = _make_cfg(model_key="qwen2-1.5b")

    dataset_info = {
        "hf_path": "wikitext",
        "hf_name": None,
        "text_column": "text",
        "streaming": False,
        "splits": {"train": "train"},
    }

    mock_tok = MagicMock()

    mock_tok.name_or_path = "qwen2-1.5b"

    mock_tok.vocab_size = 151936

    mock_tok.eos_token_id = 151643

    token_arrays = [np.array([1, 2, 3], dtype=np.uint32)]

    with (
        patch("scripts.prepare_data.get_dataset_info", return_value=dataset_info),
        patch("scripts.prepare_data.get_tokenizer", return_value=(mock_tok, 151643)),
        patch(
            "scripts.prepare_data._iter_token_arrays", return_value=iter(token_arrays)
        ),
        patch("datasets.load_dataset", return_value=MagicMock()),
    ):
        tokenize_and_pack(cfg, tmp_path, num_proc=1)

    shards = list(tmp_path.glob("train_*.bin"))

    assert len(shards) == 1

    with open(shards[0], "rb") as f:
        f.read(8)

        dtype_flag = struct.unpack("<H", f.read(2))[0]

    assert dtype_flag == 1


def test_main_uses_cpu_count_when_num_proc_none(tmp_path):

    from scripts.prepare_data import main

    cfg_path = tmp_path / "test.yaml"

    cfg_path.write_text(
        "dataset:\n  dataset_key: wikitext-2\nmodel:\n  model_key: gpt-neo-125m\n"
    )

    with patch(
        "sys.argv",
        ["prepare_data.py", "--config", str(cfg_path), "--out-dir", str(tmp_path)],
    ):
        with patch("scripts.prepare_data.tokenize_and_pack") as mock_pack:
            with patch("multiprocessing.cpu_count", return_value=4):
                main()

                _, _, kwargs = mock_pack.mock_calls[0]

                assert (
                    mock_pack.call_args[1]["num_proc"] == 4
                    or mock_pack.call_args[0][2] == 4
                )
