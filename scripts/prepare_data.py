"""
scripts/prepare_data.py — Stage 1: Offline Data Preparation

Downloads a dataset from HuggingFace, tokenizes it, and packs the tokens
into dense binary shards (.bin files) for zero-padding, zero-waste training.

Shard format: [uint64 token_count (8 bytes)] [uint16 tokens ...]
Compatible with nanoGPT / llm.c binary shard format.

Usage:
    # Default dataset from config:
    python -m scripts.prepare_data --config experiments/gptneo_125m_stress_v6-wikitext.yaml

    # Override dataset from CLI:
    python -m scripts.prepare_data --config experiments/gptneo_125m_stress_v6-wikitext.yaml \
        --dataset fineweb-edu

    # Parallel tokenization (recommended for large datasets):
    python -m scripts.prepare_data --config experiments/gptneo_125m_stress_v6-wikitext.yaml \
        --dataset fineweb-edu --num-proc 8

    # Run via Modal (Stage 1, saves to Modal Volume):
    # Set CONFIG = "experiments/gptneo_125m_stress_v6-wikitext.yaml" in run_modal_training.py, then:
    modal run run_modal_training.py::stage_data

Dataset sizing guide:
    wikitext-2 ~2M tokens → 1 shard — unit tests, smoke tests
    wikitext-103 ~103M tokens → 1 shard — router experiments (current default)
    openwebtext ~9B tokens → ~90 shards — mid-scale pre-training
    fineweb-edu ~10B tokens → ~100 shards — RECOMMENDED: research standard
    c4 ~350B tokens → ~3500 shards — production scale
"""

from __future__ import annotations

import argparse
import multiprocessing
import struct
from pathlib import Path
from typing import Iterator

import numpy as np
from omegaconf import OmegaConf

from src.configs.dataset import get_dataset_info, get_shard_dir

# Shard size: 100M tokens per shard (~200MB on disk as uint16).
SHARD_SIZE = int(1e8)


def parse_args():
    parser = argparse.ArgumentParser(description="T-MoE Data Preparation")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config (e.g. experiments/gptneo_125m_stress_v6-wikitext.yaml)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional override for dataset name. If omitted, reads from config.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for shards. Defaults to data/shards/<dataset_name>/",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help=(
            "Parallel tokenization workers. Defaults to min(8, cpu_count). "
            "Set to 1 to disable parallelism. "
            "Ignored for streaming datasets (streaming tokenizes sequentially)."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=(
            "HuggingFace datasets cache directory. "
            "Pass the S3-downloaded data path to avoid re-downloading from the hub "
            "(required in VPC-isolated environments)."
        ),
    )
    return parser.parse_args()


def load_config(config_path: str, dataset_override: str | None = None):
    """Load config from YAML, optionally overriding the dataset key."""
    cfg = OmegaConf.load(config_path)
    if dataset_override:
        OmegaConf.update(cfg, "dataset.dataset_key", dataset_override)
    return cfg


def get_tokenizer(model_key: str):
    """Load tokenizer from HuggingFace based on model key."""
    from transformers import AutoTokenizer
    from src.configs.model import model_lookup

    model_info = model_lookup(model_key)
    hf_name = model_info["hf_name"]

    print(f"Loading tokenizer: {hf_name}")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = int(
        1e30
    )  # to supress tokenizer warnings about max_length
    return tokenizer, tokenizer.eos_token_id


# Worker-process-level tokenizer cache. Loaded once per worker process,
# reused across all batch calls — avoids ~0.5s from_pretrained overhead per batch.
_worker_tok = None
_worker_tok_name = None


def _tokenize_batch(args: tuple) -> list[list[int]]:
    global _worker_tok, _worker_tok_name
    texts, tokenizer_name, eos_id = args
    if _worker_tok is None or _worker_tok_name != tokenizer_name:
        from transformers import AutoTokenizer

        _worker_tok = AutoTokenizer.from_pretrained(tokenizer_name)
        _worker_tok.model_max_length = int(1e30)
        _worker_tok_name = tokenizer_name
    results = []
    for text in texts:
        if text and text.strip():
            ids = _worker_tok.encode(text, add_special_tokens=False)
            ids.append(eos_id)
            results.append(ids)
    return results


def _iter_token_arrays(
    dataset,
    text_column: str,
    tokenizer_name: str,
    eos_id: int,
    num_proc: int,
    batch_size: int = 512,
) -> Iterator[np.ndarray]:
    """
    Yield numpy uint16 token arrays for each document in the dataset.

    For non-streaming datasets: uses HuggingFace .map() with num_proc workers
    (parallel tokenization, significantly faster than sequential for large datasets).
    For streaming datasets: tokenizes sequentially (HF streaming doesn't support
    multiprocessing map — the bottleneck is network I/O anyway).
    """
    from datasets import IterableDataset

    is_streaming = isinstance(dataset, IterableDataset)

    if is_streaming:
        # Streaming: tokenize in batches via HF batched map
        # (sequential but overlaps network I/O with tokenization)
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name)

        batch_texts: list[str] = []
        for example in dataset:
            text = example[text_column]
            if text and text.strip():
                batch_texts.append(text)
            if len(batch_texts) >= batch_size:
                encoded = tok(batch_texts, add_special_tokens=False)
                for ids in encoded["input_ids"]:
                    yield np.array(ids + [eos_id], dtype=np.uint16)
                batch_texts = []
        if batch_texts:
            encoded = tok(batch_texts, add_special_tokens=False)
            for ids in encoded["input_ids"]:
                yield np.array(ids + [eos_id], dtype=np.uint16)
    else:
        # Non-streaming: parallel tokenization via multiprocessing pool.
        # Batches are generated lazily so we never load the full dataset into RAM.
        def _batch_gen():
            batch = []
            for ex in dataset:
                text = ex[text_column]
                if text and text.strip():
                    batch.append(text)
                    if len(batch) >= batch_size:
                        yield (batch, tokenizer_name, eos_id)
                        batch = []
            if batch:
                yield (batch, tokenizer_name, eos_id)

        print(f"  Tokenizing with {num_proc} workers (batch_size={batch_size})...")
        docs_done = 0
        with multiprocessing.Pool(num_proc) as pool:
            for i, token_lists in enumerate(
                pool.imap(_tokenize_batch, _batch_gen(), chunksize=16)
            ):
                for ids in token_lists:
                    yield np.array(ids, dtype=np.uint16)
                docs_done += len(token_lists)
                if i > 0 and i % 200 == 0:
                    print(f"  ... {docs_done:,} docs tokenized ({i} batches)")


def tokenize_and_pack(cfg, out_dir: Path, num_proc: int = 1, cache_dir: str | None = None) -> None:
    from datasets import load_dataset

    dataset_key = cfg.dataset.dataset_key
    dataset_info = get_dataset_info(dataset_key)
    model_key = cfg.model.model_key
    tokenizer, eos_id = get_tokenizer(model_key)
    tokenizer_name = tokenizer.name_or_path

    use_streaming = dataset_info.get("streaming", False)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Dataset   : {dataset_key}")
    print(f"HF Path   : {dataset_info['hf_path']}")
    if dataset_info.get("hf_name"):
        print(f"HF Name   : {dataset_info['hf_name']}")
    print(f"Streaming : {use_streaming}")
    print(f"Workers   : {num_proc if not use_streaming else 'N/A (streaming)'}")
    print(f"Out Dir   : {out_dir}")
    print(f"{'=' * 60}\n")

    for split_name, hf_split in dataset_info["splits"].items():
        if hf_split is None:
            print(f"Skipping '{split_name}' split (not available for {dataset_key})")
            continue

        print(f"Processing split: {split_name} → {hf_split}")

        dataset = load_dataset(
            path=dataset_info["hf_path"],
            name=dataset_info.get("hf_name"),
            split=hf_split,
            streaming=use_streaming,
            cache_dir=cache_dir,
        )

        shard_index = 0
        token_buffer = np.empty(SHARD_SIZE, dtype=np.uint16)
        token_count = 0
        total_tokens_written = 0

        def flush_shard(buf: np.ndarray, count: int, idx: int) -> Path:
            path = out_dir / f"{split_name}_shard_{idx:04d}.bin"
            with open(path, "wb") as f:
                f.write(struct.pack("<Q", count))
                f.write(buf[:count].tobytes())
            return path

        token_iter = _iter_token_arrays(
            dataset,
            dataset_info["text_column"],
            tokenizer_name,
            eos_id,
            num_proc,
        )

        for tokens in token_iter:
            pos = 0
            while pos < len(tokens):
                space = SHARD_SIZE - token_count
                chunk = tokens[pos : pos + space]
                token_buffer[token_count : token_count + len(chunk)] = chunk
                token_count += len(chunk)
                pos += len(chunk)

                if token_count == SHARD_SIZE:
                    shard_path = flush_shard(token_buffer, token_count, shard_index)
                    print(
                        f"  Wrote shard {shard_index:04d}: {shard_path.name} "
                        f"({token_count:,} tokens)"
                    )
                    total_tokens_written += token_count
                    shard_index += 1
                    token_count = 0

        if token_count > 0:
            shard_path = flush_shard(token_buffer, token_count, shard_index)
            print(
                f"  Wrote shard {shard_index:04d}: {shard_path.name} "
                f"({token_count:,} tokens) [partial]"
            )
            total_tokens_written += token_count
            shard_index += 1

        print(
            f"  '{split_name}' complete: "
            f"{shard_index} shard(s), {total_tokens_written:,} tokens.\n"
        )


def main():
    args = parse_args()
    cfg = load_config(args.config, args.dataset)

    dataset_key = cfg.dataset.dataset_key
    num_proc = (
        args.num_proc
        if args.num_proc is not None
        else min(8, multiprocessing.cpu_count())
    )

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = get_shard_dir(dataset_key, cfg.model.model_key)

    tokenize_and_pack(cfg, out_dir, num_proc=num_proc, cache_dir=args.cache_dir)
    print(f"Data preparation complete. Shards written to: {out_dir}")


if __name__ == "__main__":
    main()
