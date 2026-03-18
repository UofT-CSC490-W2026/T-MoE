from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import torch
import torch.nn.functional as F

from evals.loading import load_model_for_eval
from evals.results_schema import build_results_payload, write_results_json
from src.configs.model import model_lookup

DEFAULT_PERPLEXITY_DATASETS: tuple[dict[str, Any], ...] = (
    {
        "result_prefix": "wikitext103",
        "hf_path": "wikitext",
        "hf_name": "wikitext-103-raw-v1",
        "split": "test",
        "streaming": False,
        "text_column": "text",
        "include_bpb": True,
    },
    {
        "result_prefix": "c4",
        "hf_path": "allenai/c4",
        "hf_name": "en",
        "split": "validation",
        "streaming": True,
        "text_column": "text",
        "include_bpb": False,
    },
)


def _cfg_select(config: Any, key: str, default: Any = None) -> Any:
    current = config
    for part in key.split("."):
        if hasattr(current, "get"):
            current = current.get(part)
        elif isinstance(current, dict):
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


def infer_eval_context_length(model: Any, config: Any, tokenizer: Any | None = None) -> int:
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


def compute_document_nll(
    model: Any,
    input_ids: torch.Tensor,
    *,
    stride: int,
    max_length: int,
    device: str,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> tuple[float, int]:
    """
    Compute next-token NLL over one tokenized document using overlapping windows.

    Each target token is scored exactly once, even though windows overlap.
    """
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

            logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.logits
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


def summarize_language_model_metrics(
    *,
    total_nll: float,
    total_tokens: int,
    total_bytes: int | None = None,
) -> Dict[str, float]:
    if total_tokens <= 0:
        raise ValueError("total_tokens must be positive")

    metrics = {"ppl": float(torch.exp(torch.tensor(total_nll / total_tokens)).item())}
    if total_bytes is not None:
        if total_bytes <= 0:
            raise ValueError("total_bytes must be positive when provided")
        metrics["bpb"] = total_nll / (torch.log(torch.tensor(2.0)).item() * total_bytes)
    return metrics


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
) -> Dict[str, float]:
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    documents_scored = 0

    for text in texts:
        if not isinstance(text, str) or not text:
            continue

        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        doc_nll, doc_tokens = compute_document_nll(
            model,
            input_ids,
            stride=stride,
            max_length=max_length,
            device=device,
            autocast_dtype=autocast_dtype,
        )
        if doc_tokens == 0:
            continue

        total_nll += doc_nll
        total_tokens += doc_tokens
        if include_bpb:
            total_bytes += len(text.encode("utf-8"))

        documents_scored += 1
        if max_documents is not None and documents_scored >= max_documents:
            break

    summary = summarize_language_model_metrics(
        total_nll=total_nll,
        total_tokens=total_tokens,
        total_bytes=total_bytes if include_bpb else None,
    )
    summary["documents_scored"] = float(documents_scored)
    summary["tokens_scored"] = float(total_tokens)
    return summary


def _load_dataset_texts(
    dataset_spec: Dict[str, Any],
    *,
    max_documents: int | None = None,
) -> Iterable[str]:
    from datasets import load_dataset

    dataset = load_dataset(
        dataset_spec["hf_path"],
        dataset_spec.get("hf_name"),
        split=dataset_spec["split"],
        streaming=dataset_spec.get("streaming", False),
    )

    text_column = dataset_spec.get("text_column", "text")
    count = 0
    for row in dataset:
        if text_column not in row:
            continue
        yield row[text_column]
        count += 1
        if max_documents is not None and count >= max_documents:
            break


def _load_tokenizer_for_model(config: Any):
    from transformers import AutoTokenizer

    model_info = model_lookup(_cfg_select(config, "model.model_key", "gpt-neo-125m"))
    tokenizer = AutoTokenizer.from_pretrained(model_info["hf_name"])
    return tokenizer


def run_perplexity_eval(
    config: Any,
    checkpoint_path: str | Path,
    *,
    output_path: str | Path | None = None,
    device: str = "cuda",
    stride: int = 512,
    max_documents: int | None = None,
    dataset_specs: Sequence[Dict[str, Any]] = DEFAULT_PERPLEXITY_DATASETS,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    model, checkpoint_info = load_model_for_eval(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        dtype=autocast_dtype if device.startswith("cuda") else None,
    )
    tokenizer = _load_tokenizer_for_model(config)
    max_length = infer_eval_context_length(model, config, tokenizer)

    results: Dict[str, float] = {}
    dataset_metadata: Dict[str, Any] = {}
    for dataset_spec in dataset_specs:
        summary = evaluate_text_documents(
            model,
            tokenizer,
            _load_dataset_texts(dataset_spec, max_documents=max_documents),
            stride=stride,
            max_length=max_length,
            device=device,
            autocast_dtype=autocast_dtype,
            max_documents=max_documents,
            include_bpb=dataset_spec.get("include_bpb", False),
        )
        prefix = dataset_spec["result_prefix"]
        results[f"{prefix}_ppl"] = summary["ppl"]
        if "bpb" in summary:
            results[f"{prefix}_bpb"] = summary["bpb"]
        dataset_metadata[prefix] = {
            "documents_scored": int(summary["documents_scored"]),
            "tokens_scored": int(summary["tokens_scored"]),
            "split": dataset_spec["split"],
        }

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

    if output_path is not None:
        write_results_json(payload, output_path)
    return payload
