from __future__ import annotations
import math
import os
import queue
import struct
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, NamedTuple, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from evals.loading import load_model_for_eval
from evals.results_schema import build_results_payload, write_results_json
from src.configs.dataset import get_shard_dir
from src.configs.model import model_lookup

_SHARD_BASE_DIR = os.environ.get("SHARD_BASE_DIR", "data/shards")

DEFAULT_PERPLEXITY_DATASETS: tuple[dict[str, Any], ...] = (
    {
        "result_prefix": "wikitext103",
        "source": "shards",
        "dataset_key": "wikitext-103",
        "include_bpb": False,
    },
    {
        "result_prefix": "pile",
        "source": "shards",
        "dataset_key": "pile-val",
        "include_bpb": False,
    },
)


def _cfg_select(config: Any, key: str, default: Any = None) -> Any:
    current = config
    for part in key.split("."):
        if hasattr(current, "get"):
            current = current.get(part)
        else:
            return default
        if current is None:
            return default
    return current


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _autocast_context(device: str, dtype: torch.dtype):
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def infer_eval_context_length(
    model: Any, config: Any, tokenizer: Any | None = None
) -> int:
    backbone_config = getattr(getattr(model, "backbone", None), "config", None)
    max_length = (
        getattr(backbone_config, "max_position_embeddings", None)
        or getattr(backbone_config, "n_positions", None)
        or _cfg_select(config, "dataset.max_seq_len", 1024)
    )

    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and tokenizer_limit < 1_000_000:
        max_length = min(int(max_length), tokenizer_limit)

    return max(int(max_length), 2)


def summarize_language_model_metrics(
    *,
    total_nll: float,
    total_tokens: int,
    total_bytes: int | None = None,
) -> Dict[str, float]:
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")

    metrics = {"ppl": math.exp(total_nll / total_tokens)}
    if total_bytes is not None:
        if total_bytes <= 0:
            raise ValueError("total_bytes must be positive when provided")
        metrics["bpb"] = total_nll / (math.log(2.0) * total_bytes)
    return metrics


class _Window(NamedTuple):
    doc_idx: int
    window_input: torch.Tensor  # [1, L]
    valid_mask: torch.BoolTensor  # [L-1]


def _document_windows(
    doc_idx: int,
    input_ids: torch.Tensor,
    stride: int,
    max_length: int,
) -> Iterator[_Window]:
    seq_len = int(input_ids.size(1))
    if seq_len < 2:
        return

    for begin in range(0, seq_len, stride):
        begin_loc = max(begin + stride - max_length, 0)
        end_loc = min(begin + stride, seq_len)
        target_len = end_loc - begin

        if end_loc - begin_loc < 2 or target_len <= 0:
            continue

        window_input = input_ids[:, begin_loc:end_loc]  # [1, wlen], still on CPU

        # valid_mask over the [L-1] shifted positions
        label_positions = torch.arange(begin_loc + 1, end_loc)
        score_from = end_loc - target_len
        valid_mask = label_positions >= score_from  # [wlen-1]

        yield _Window(doc_idx=doc_idx, window_input=window_input, valid_mask=valid_mask)

        if end_loc == seq_len:
            break


def _tokenize_worker(
    texts: Iterable[str],
    tokenizer: Any,
    out_queue: queue.Queue,
    max_documents: int | None,
) -> None:
    doc_idx = 0
    for text in texts:
        if not isinstance(text, str) or not text:
            continue
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_tensors="pt",
                verbose=False,
            )
        except TypeError:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_tensors="pt",
            )
        out_queue.put((doc_idx, encoded["input_ids"], text))
        doc_idx += 1
        if max_documents is not None and doc_idx >= max_documents:
            break
    out_queue.put(None)


def _run_batched_forward(
    model: Any,
    windows: list[_Window],
    device: str,
    autocast_dtype: torch.dtype,
) -> list[tuple[torch.Tensor, torch.BoolTensor]]:
    lengths = [w.window_input.size(1) for w in windows]

    # Fast path: all windows same length (stride=max_length, no overlap) — no padding needed
    all_same_len = len(set(lengths)) == 1

    if all_same_len:
        batch = torch.cat([w.window_input for w in windows], dim=0)
    else:
        # Right-pad windows to common length; real tokens sit at [0:wlen]
        max_len = max(lengths)
        padded = []
        for w in windows:
            wlen = w.window_input.size(1)
            if wlen < max_len:
                pad = torch.zeros(1, max_len - wlen, dtype=w.window_input.dtype)
                padded.append(torch.cat([w.window_input, pad], dim=1))
            else:
                padded.append(w.window_input)
        batch = torch.cat(padded, dim=0)

    if device.startswith("cuda"):
        batch = batch.pin_memory().to(device, non_blocking=True)
    else:
        batch = batch.to(device)

    with torch.no_grad(), _autocast_context(device, autocast_dtype):
        outputs = model(input_ids=batch)

    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits
    # Compute cross-entropy in the model's native dtype (bf16/fp16) to avoid
    # materializing a full float32 logits tensor — critical for large vocab models
    # like Qwen (vocab=151k) where float32 cast causes ~40GB memory spike at B=32.

    results = []
    for i, w in enumerate(windows):
        wlen = lengths[i]
        shift_logits = logits[i, : wlen - 1, :]  # [wlen-1, V] — still in autocast dtype
        shift_labels = batch[i, 1:wlen]  # [wlen-1]
        # F.cross_entropy upcasts internally to float32 for numerical stability
        token_losses = F.cross_entropy(
            shift_logits.float(), shift_labels, reduction="none"
        )  # [wlen-1]
        results.append((token_losses, w.valid_mask))

    return results


def compute_document_nll(
    model: Any,
    input_ids: torch.Tensor,
    *,
    stride: int,
    max_length: int,
    device: str,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> tuple[float, int]:
    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError("input_ids must have shape [1, seq_len]")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")

    seq_len = int(input_ids.size(1))
    if seq_len < 2:
        return 0.0, 0

    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for begin in range(0, seq_len, stride):
            begin_loc = max(begin + stride - max_length, 0)
            end_loc = min(begin + stride, seq_len)
            target_len = end_loc - begin

            if end_loc - begin_loc < 2 or target_len <= 0:
                continue

            window_input = input_ids[:, begin_loc:end_loc].to(device)
            with _autocast_context(device, autocast_dtype):
                outputs = model(input_ids=window_input)

            logits = (
                outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits
            )
            shift_logits = logits[:, :-1, :].float()
            shift_labels = window_input[:, 1:]

            label_positions = torch.arange(
                begin_loc + 1,
                end_loc,
                device=window_input.device,
            )
            score_from = end_loc - target_len
            valid_mask = label_positions >= score_from
            if not valid_mask.any():
                continue

            token_losses = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                reduction="none",
            ).view_as(shift_labels)

            total_nll += token_losses[:, valid_mask].sum().item()
            total_tokens += int(valid_mask.sum().item())

            if end_loc == seq_len:
                break

    return total_nll, total_tokens


def evaluate_text_documents(
    model: Any,
    tokenizer: Any,
    texts: Iterable[str],
    *,
    stride: int,
    max_length: int,
    device: str,
    autocast_dtype: torch.dtype = torch.bfloat16,
    max_documents: int | None = None,
    include_bpb: bool = True,
    progress_label: str | None = None,
    batch_size: int = 32,
    total_documents: int | None = None,
) -> Dict[str, float]:
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    documents_scored = 0

    tok_queue: queue.Queue = queue.Queue(maxsize=32)
    worker = threading.Thread(
        target=_tokenize_worker,
        args=(texts, tokenizer, tok_queue, max_documents),
        daemon=True,
    )
    worker.start()

    # per-doc accumulators not yet flushed to totals
    doc_nll: dict[int, float] = {}
    doc_tokens: dict[int, int] = {}
    doc_bytes: dict[int, int] = {}
    doc_pending_windows: dict[int, int] = {}  # windows not yet forwarded

    pending_windows: list[_Window] = []

    def _flush_batch(force: bool = False) -> None:
        nonlocal total_nll, total_tokens, total_bytes, documents_scored

        if not pending_windows:
            return
        if not force and len(pending_windows) < batch_size:
            return

        results = _run_batched_forward(model, pending_windows, device, autocast_dtype)
        for (token_losses, valid_mask), win in zip(results, pending_windows):
            vm = valid_mask.to(token_losses.device)
            if vm.any():
                doc_nll[win.doc_idx] = (
                    doc_nll.get(win.doc_idx, 0.0) + token_losses[vm].sum().item()
                )
                doc_tokens[win.doc_idx] = doc_tokens.get(win.doc_idx, 0) + int(
                    vm.sum().item()
                )
            doc_pending_windows[win.doc_idx] -= 1

        pending_windows.clear()

        # Commit fully-processed docs
        completed = [d for d, rem in doc_pending_windows.items() if rem == 0]
        for d in sorted(completed):
            if doc_tokens.get(d, 0) == 0:
                pass  # skip empty docs (still counted below to match progress)
            else:
                total_nll += doc_nll.pop(d, 0.0)
                total_tokens += doc_tokens.pop(d, 0)
                total_bytes += doc_bytes.pop(d, 0)
                documents_scored += 1
                pbar.update(1)
            del doc_pending_windows[d]

    with tqdm(
        total=total_documents,
        unit="doc",
        desc=progress_label or "perplexity",
        file=sys.stderr,
        dynamic_ncols=True,
        leave=True,
    ) as pbar:
        while True:
            item = tok_queue.get()
            if item is None:
                break
            doc_idx, input_ids, text = item

            windows = list(_document_windows(doc_idx, input_ids, stride, max_length))
            if not windows:
                continue

            doc_pending_windows[doc_idx] = len(windows)
            doc_nll[doc_idx] = 0.0
            doc_tokens[doc_idx] = 0
            doc_bytes[doc_idx] = len(text.encode("utf-8")) if include_bpb else 0

            pending_windows.extend(windows)
            _flush_batch(force=False)

        _flush_batch(force=True)

        # Commit any docs whose windows were in the final forced batch
        completed = [d for d, rem in doc_pending_windows.items() if rem == 0]
        for d in sorted(completed):
            if doc_tokens.get(d, 0) > 0:
                total_nll += doc_nll.pop(d, 0.0)
                total_tokens += doc_tokens.pop(d, 0)
                total_bytes += doc_bytes.pop(d, 0)
                documents_scored += 1
                pbar.update(1)
            doc_pending_windows.pop(d, None)

    worker.join()

    if total_tokens == 0:
        raise ValueError("No tokens scored — all documents were empty or too short.")

    summary = summarize_language_model_metrics(
        total_nll=total_nll,
        total_tokens=total_tokens,
        total_bytes=total_bytes if include_bpb else None,
    )
    summary["documents_scored"] = float(documents_scored)
    summary["tokens_scored"] = float(total_tokens)
    return summary


def evaluate_token_shards(
    model: Any,
    shard_dir: str | Path,
    *,
    stride: int,
    max_length: int,
    device: str,
    autocast_dtype: torch.dtype,
    batch_size: int,
    progress_label: str = "shard-eval",
    max_tokens: int | None = None,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, float]:
    """Evaluate perplexity over token shards.

    When world_size > 1, each rank processes a disjoint slice of windows and
    the results are all-reduced so every rank returns identical metrics.
    """
    shard_dir = Path(shard_dir)
    shard_files = sorted(shard_dir.glob("val_shard_*.bin"))
    if not shard_files:
        raise FileNotFoundError(
            f"No val_shard_*.bin files found in {shard_dir}.\n"
            f"Run scripts/prepare_data.py first."
        )

    token_arrays = []
    for path in shard_files:
        with open(path, "rb") as f:
            n_tokens = struct.unpack("<Q", f.read(8))[0]
        mm = np.memmap(path, dtype=np.uint16, mode="r", offset=8, shape=(n_tokens,))
        token_arrays.append(mm.astype(np.int64))

    all_tokens = np.concatenate(token_arrays)
    n = len(all_tokens)
    if max_tokens is not None:
        n = min(n, max_tokens)

    all_window_starts = list(range(0, n - 1, stride))

    # Distribute windows across ranks — each rank owns a contiguous slice
    # so window indices stay globally consistent (needed for valid_mask logic).
    rank_window_starts = all_window_starts[rank::world_size]

    total_nll = 0.0
    scored_tokens = 0
    windows_scored = 0

    with tqdm(
        total=len(rank_window_starts),
        unit="win",
        desc=f"{progress_label}" + (f"[{rank}/{world_size}]" if world_size > 1 else ""),
        file=sys.stderr,
        dynamic_ncols=True,
        leave=True,
        disable=rank != 0,
    ) as pbar:
        for batch_start in range(0, len(rank_window_starts), batch_size):
            batch_ws = rank_window_starts[batch_start : batch_start + batch_size]
            windows: list[_Window] = []

            for ws in batch_ws:
                begin_loc = ws
                end_loc = min(ws + max_length, n)
                if end_loc - begin_loc < 2:
                    continue

                tokens = torch.tensor(
                    all_tokens[begin_loc:end_loc], dtype=torch.long
                ).unsqueeze(0)

                wlen = tokens.size(1)
                # First global window scores all positions; rest score only last stride
                if ws == 0:
                    valid_mask = torch.ones(wlen - 1, dtype=torch.bool)
                else:
                    valid_mask = torch.zeros(wlen - 1, dtype=torch.bool)
                    n_to_score = min(stride, wlen - 1)
                    valid_mask[-n_to_score:] = True

                windows.append(
                    _Window(
                        doc_idx=ws,  # use token offset as stable doc_idx
                        window_input=tokens,
                        valid_mask=valid_mask,
                    )
                )

            if not windows:
                continue

            results = _run_batched_forward(model, windows, device, autocast_dtype)
            for (token_losses, valid_mask), win in zip(results, windows):
                vm = valid_mask.to(token_losses.device)
                if vm.any():
                    total_nll += token_losses[vm].sum().item()
                    scored_tokens += int(vm.sum().item())
            windows_scored += len(windows)
            pbar.update(len(windows))

    # All-reduce across ranks so every rank returns the same aggregate metrics
    if world_size > 1:
        import torch.distributed as dist

        stats = torch.tensor(
            [total_nll, float(scored_tokens), float(windows_scored)], device=device
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_nll = stats[0].item()
        scored_tokens = int(stats[1].item())
        windows_scored = int(stats[2].item())

    if scored_tokens == 0:
        raise ValueError(f"No tokens scored from shards in {shard_dir}")

    summary = summarize_language_model_metrics(
        total_nll=total_nll,
        total_tokens=scored_tokens,
    )
    summary["tokens_scored"] = float(scored_tokens)
    summary["windows_scored"] = float(windows_scored)
    return summary


def _load_dataset_texts(
    dataset_spec: Dict[str, Any],
    *,
    max_documents: int | None = None,
) -> tuple[Iterable[str], int | None]:
    from datasets import load_dataset

    spec_cap = dataset_spec.get("max_documents")
    effective_cap = (
        min(c for c in [max_documents, spec_cap] if c is not None)
        if any(c is not None for c in [max_documents, spec_cap])
        else None
    )

    streaming = dataset_spec.get("streaming", False)
    dataset = load_dataset(
        dataset_spec["hf_path"],
        dataset_spec.get("hf_name"),
        split=dataset_spec["split"],
        streaming=streaming,
    )

    text_column = dataset_spec.get("text_column", "text")
    total_hint: int | None = None
    if not streaming:
        try:
            total_hint = len(dataset)
            if effective_cap is not None:
                total_hint = min(total_hint, effective_cap)
        except TypeError:
            total_hint = None

    def _gen():
        count = 0
        for row in dataset:
            if text_column not in row:
                continue
            yield row[text_column]
            count += 1
            if effective_cap is not None and count >= effective_cap:
                break

    return _gen(), total_hint


def _load_tokenizer_for_model(config: Any):
    from transformers import AutoTokenizer

    model_info = model_lookup(_cfg_select(config, "model.model_key", "gpt-neo-125m"))
    tokenizer = AutoTokenizer.from_pretrained(model_info["hf_name"])
    return tokenizer


def run_perplexity_eval(
    config: Any,
    checkpoint_path: str | Path,
    *,
    model: Any | None = None,
    checkpoint_info: Dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    device: str = "cuda",
    stride: int = 512,
    max_documents: int | None = None,
    dataset_specs: Sequence[Dict[str, Any]] = DEFAULT_PERPLEXITY_DATASETS,
    autocast_dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 32,
    # Hard cap on context length — prevents OOM on large-context models (e.g. Qwen 32k).
    # PPL is evaluated at 2048 tokens regardless of model's max_position_embeddings.
    max_eval_length: int = 2048,
    rank: int = 0,
    world_size: int = 1,
) -> Dict[str, Any]:
    if model is None:
        model, checkpoint_info = load_model_for_eval(
            config=config,
            checkpoint_path=checkpoint_path,
            device=device,
            dtype=autocast_dtype if device.startswith("cuda") else None,
        )
    tokenizer = _load_tokenizer_for_model(config)
    max_length = min(
        infer_eval_context_length(model, config, tokenizer), max_eval_length
    )
    model_key = _cfg_select(config, "model.model_key", "gpt-neo-125m")

    results: Dict[str, float] = {}
    dataset_metadata: Dict[str, Any] = {}

    for dataset_spec in dataset_specs:
        prefix = dataset_spec["result_prefix"]
        include_bpb = dataset_spec.get("include_bpb", False)

        if dataset_spec.get("source") == "shards":
            dataset_key = dataset_spec["dataset_key"]
            shard_dir = get_shard_dir(dataset_key, model_key, base=_SHARD_BASE_DIR)
            # Convert max_documents to a token cap: pile-val averages ~75k tokens/doc
            max_tokens = max_documents * 75_000 if max_documents is not None else None
            summary = evaluate_token_shards(
                model,
                shard_dir,
                stride=stride,
                max_length=max_length,
                device=device,
                autocast_dtype=autocast_dtype,
                batch_size=batch_size,
                progress_label=prefix,
                max_tokens=max_tokens,
                rank=rank,
                world_size=world_size,
            )
            dataset_metadata[prefix] = {
                "tokens_scored": int(summary["tokens_scored"]),
                "windows_scored": int(summary["windows_scored"]),
                "shard_dir": str(shard_dir),
            }
        else:
            texts, total_hint = _load_dataset_texts(
                dataset_spec, max_documents=max_documents
            )
            summary = evaluate_text_documents(
                model,
                tokenizer,
                texts,
                stride=stride,
                max_length=max_length,
                device=device,
                autocast_dtype=autocast_dtype,
                max_documents=max_documents,
                include_bpb=include_bpb,
                progress_label=prefix,
                batch_size=batch_size,
                total_documents=total_hint,
            )
            dataset_metadata[prefix] = {
                "documents_scored": int(summary["documents_scored"]),
                "tokens_scored": int(summary["tokens_scored"]),
                "split": dataset_spec.get("split", ""),
            }

        results[f"{prefix}_ppl"] = summary["ppl"]
        if "bpb" in summary:
            results[f"{prefix}_bpb"] = summary["bpb"]

    payload = build_results_payload(
        task="perplexity",
        checkpoint_path=checkpoint_path,
        checkpoint_info=checkpoint_info,
        config=config,
        results=results,
        metadata={
            "dtype": _dtype_name(autocast_dtype),
            "stride": int(stride),
            "max_length": int(max_length),
            "device": device,
            "datasets": dataset_metadata,
            "torch_version": torch.__version__,
        },
    )

    if output_path is not None and rank == 0:
        write_results_json(payload, output_path)

    if rank == 0:
        print("\n── Perplexity Results ──────────────────────────")
        for k, v in results.items():
            print(f"  {k:<30} {v:.2f}")
        print("────────────────────────────────────────────────\n")

    return payload
