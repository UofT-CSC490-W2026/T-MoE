from __future__ import annotations

import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import torch

from evals.loading import load_model_for_eval
from evals.results_schema import build_results_payload, write_results_json


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def _autocast_context(device: str, dtype: torch.dtype):
    if device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _device_synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device=device)


def summarize_timing_measurements(
    durations_s: Sequence[float],
    *,
    batch_size: int,
    seq_len: int,
) -> Dict[str, float]:
    if not durations_s:
        raise ValueError("durations_s must not be empty")

    tokens_per_pass = batch_size * seq_len
    throughputs = [tokens_per_pass / duration for duration in durations_s]
    ms_per_token = [(duration * 1000.0) / tokens_per_pass for duration in durations_s]

    p50 = statistics.median(ms_per_token)
    p95_index = max(0, min(len(ms_per_token) - 1, int(round(0.95 * (len(ms_per_token) - 1)))))
    p95 = sorted(ms_per_token)[p95_index]

    return {
        "throughput_tokens_per_sec_mean": statistics.mean(throughputs),
        "throughput_tokens_per_sec_std": statistics.pstdev(throughputs) if len(throughputs) > 1 else 0.0,
        "latency_ms_per_token_p50": p50,
        "latency_ms_per_token_p95": p95,
    }


def _profile_loaded_model(
    model: Any,
    *,
    device: str,
    seq_len: int,
    batch_sizes: Iterable[int],
    warmup_iters: int,
    benchmark_iters: int,
    autocast_dtype: torch.dtype,
) -> Dict[str, Any]:
    vocab_size = getattr(model, "vocab_size", None) or getattr(model.backbone.config, "vocab_size", 50257)
    metrics: Dict[str, Any] = {}

    for batch_size in batch_sizes:
        input_ids = torch.randint(
            low=0,
            high=int(vocab_size),
            size=(batch_size, seq_len),
            device=device,
            dtype=torch.long,
        )

        with torch.inference_mode():
            for _ in range(warmup_iters):
                with _autocast_context(device, autocast_dtype):
                    model(input_ids=input_ids)
            _device_synchronize(device)

            durations = []
            for _ in range(benchmark_iters):
                start = time.perf_counter()
                with _autocast_context(device, autocast_dtype):
                    model(input_ids=input_ids)
                _device_synchronize(device)
                durations.append(time.perf_counter() - start)

        metrics[f"batch_{batch_size}"] = summarize_timing_measurements(
            durations,
            batch_size=batch_size,
            seq_len=seq_len,
        )

    peak_memory_bytes = None
    if device.startswith("cuda"):
        batch_size = max(batch_sizes)
        input_ids = torch.randint(
            low=0,
            high=int(vocab_size),
            size=(batch_size, seq_len),
            device=device,
            dtype=torch.long,
        )
        torch.cuda.reset_peak_memory_stats(device=device)
        with torch.inference_mode():
            with _autocast_context(device, autocast_dtype):
                model(input_ids=input_ids)
        _device_synchronize(device)
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device=device))

    metrics["peak_memory_bytes"] = peak_memory_bytes
    return metrics


def _flatten_efficiency_results(profile: Dict[str, Any], batch_sizes: Iterable[int]) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for batch_size in batch_sizes:
        batch_metrics = profile[f"batch_{batch_size}"]
        results[f"batch_{batch_size}_throughput_tokens_per_sec_mean"] = batch_metrics[
            "throughput_tokens_per_sec_mean"
        ]
        results[f"batch_{batch_size}_throughput_tokens_per_sec_std"] = batch_metrics[
            "throughput_tokens_per_sec_std"
        ]
        results[f"batch_{batch_size}_latency_ms_per_token_p50"] = batch_metrics[
            "latency_ms_per_token_p50"
        ]
        results[f"batch_{batch_size}_latency_ms_per_token_p95"] = batch_metrics[
            "latency_ms_per_token_p95"
        ]
    if profile.get("peak_memory_bytes") is not None:
        results["peak_memory_bytes"] = float(profile["peak_memory_bytes"])
    return results


def _compute_overhead_ratios(
    current_profile: Dict[str, Any],
    reference_profile: Dict[str, Any],
    batch_sizes: Iterable[int],
) -> Dict[str, float]:
    ratios: Dict[str, float] = {}
    for batch_size in batch_sizes:
        current = current_profile[f"batch_{batch_size}"]["latency_ms_per_token_p50"]
        reference = reference_profile[f"batch_{batch_size}"]["latency_ms_per_token_p50"]
        if reference > 0:
            ratios[f"router_overhead_ratio_batch_{batch_size}"] = current / reference
    return ratios


def run_efficiency_eval(
    config: Any,
    checkpoint_path: str | Path,
    *,
    output_path: str | Path | None = None,
    device: str = "cuda",
    batch_sizes: Sequence[int] = (1, 32),
    seq_len: int = 1024,
    warmup_iters: int = 10,
    benchmark_iters: int = 100,
    reference_checkpoint_path: str | Path | None = None,
    reference_config: Any | None = None,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    model, checkpoint_info = load_model_for_eval(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
        dtype=autocast_dtype if device.startswith("cuda") else None,
    )
    profile = _profile_loaded_model(
        model,
        device=device,
        seq_len=seq_len,
        batch_sizes=batch_sizes,
        warmup_iters=warmup_iters,
        benchmark_iters=benchmark_iters,
        autocast_dtype=autocast_dtype,
    )

    results = _flatten_efficiency_results(profile, batch_sizes)
    metadata: Dict[str, Any] = {
        "device": device,
        "dtype": _dtype_name(autocast_dtype),
        "batch_sizes": list(batch_sizes),
        "seq_len": int(seq_len),
        "warmup_iters": int(warmup_iters),
        "benchmark_iters": int(benchmark_iters),
        "torch_version": torch.__version__,
    }

    if reference_checkpoint_path is not None:
        reference_model, _ = load_model_for_eval(
            config=reference_config or config,
            checkpoint_path=reference_checkpoint_path,
            device=device,
            dtype=autocast_dtype if device.startswith("cuda") else None,
        )
        reference_profile = _profile_loaded_model(
            reference_model,
            device=device,
            seq_len=seq_len,
            batch_sizes=batch_sizes,
            warmup_iters=warmup_iters,
            benchmark_iters=benchmark_iters,
            autocast_dtype=autocast_dtype,
        )
        results.update(_compute_overhead_ratios(profile, reference_profile, batch_sizes))
        metadata["reference_checkpoint_path"] = str(reference_checkpoint_path)
        metadata["reference_profile"] = reference_profile

    payload = build_results_payload(
        task="efficiency",
        checkpoint_path=checkpoint_path,
        checkpoint_info=checkpoint_info,
        config=config,
        results=results,
        metadata=metadata,
    )

    if output_path is not None:
        write_results_json(payload, output_path)
    return payload
