from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from evals.loading import load_model_for_eval
from evals.perplexity import _load_tokenizer_for_model
from evals.results_schema import build_results_payload, write_results_json

ZERO_SHOT_TASKS: tuple[str, ...] = (
    "hellaswag",
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "boolq",
    "openbookqa",
)
FIVE_SHOT_TASKS: tuple[str, ...] = ("mmlu",)

PRIMARY_METRICS: dict[str, tuple[str, ...]] = {
    "hellaswag": ("acc_norm,none", "acc_norm"),
    "piqa": ("acc,none", "acc"),
    "winogrande": ("acc,none", "acc"),
    "arc_easy": ("acc_norm,none", "acc_norm"),
    "arc_challenge": ("acc_norm,none", "acc_norm"),
    "boolq": ("acc,none", "acc"),
    "openbookqa": ("acc_norm,none", "acc_norm"),
    "mmlu": ("acc,none", "acc"),
}


def _build_harness_model(
    model: Any, tokenizer: Any, *, device: str, batch_size: int | str
):
    from lm_eval.models.huggingface import HFLM

    return HFLM(
        pretrained=model.backbone,
        tokenizer=tokenizer,
        backend="causal",
        device=device,
        batch_size=batch_size,
    )


def _simple_evaluate(**kwargs):
    import logging
    import warnings
    import datasets as _ds

    _ds.disable_progress_bars()
    logging.getLogger("datasets").setLevel(logging.ERROR)
    logging.getLogger("lm_eval").setLevel(logging.ERROR)
    logging.getLogger("lm_eval.evaluator").setLevel(logging.ERROR)
    logging.getLogger("lm_eval.tasks").setLevel(logging.ERROR)

    warnings.filterwarnings("ignore", message=".*pretrained.*not of type str.*")
    warnings.filterwarnings("ignore", message=".*Overwriting default num_fewshot.*")
    warnings.filterwarnings(
        "ignore", message=".*Combined length of context.*exceeds.*maximum length.*"
    )
    warnings.filterwarnings("ignore", message=".*Truncating.*tokens from the left.*")
    warnings.filterwarnings("ignore", message=".*Token indices sequence length.*")

    # lm_eval.models.huggingface emits these two via its module logger at WARNING level.
    # We install a filter on that logger to drop them while keeping everything else
    # (so tqdm "Running loglikelihood requests" bars still render — tqdm uses its own
    # output path and is unaffected by logger filters).
    _hflm_logger = logging.getLogger("lm_eval.models.huggingface")
    _hflm_logger.setLevel(logging.WARNING)

    _BLOCKED_MSGS = (
        "pretrained` model kwarg is not of type",
        "Passed an already-initialized model",
        "assuming single-process call",
        "Token indices sequence length is longer",
        "Combined length of context",
    )

    class _BlockFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return not any(b in record.getMessage() for b in _BLOCKED_MSGS)

    _f = _BlockFilter()
    _hflm_logger.addFilter(_f)
    # Also add to root logger to catch any re-routed messages
    logging.getLogger().addFilter(_f)

    try:
        from lm_eval.evaluator import simple_evaluate

        result = simple_evaluate(**kwargs)
    finally:
        _hflm_logger.removeFilter(_f)
        logging.getLogger().removeFilter(_f)
        _ds.enable_progress_bars()
    return result


def _extract_primary_metric(raw_results: Dict[str, Any], task_name: str) -> float:
    task_metrics = raw_results.get("results", {}).get(task_name, {})
    for metric_name in PRIMARY_METRICS[task_name]:
        if metric_name in task_metrics:
            return float(task_metrics[metric_name])
    available = sorted(task_metrics.keys())
    raise KeyError(
        f"Could not find expected metric for task '{task_name}'. Available metrics: {available}"
    )


def _collect_mmlu_breakdown(raw_results: Dict[str, Any]) -> Dict[str, float]:
    breakdown: Dict[str, float] = {}
    for task_name, metrics in raw_results.get("results", {}).items():
        if not task_name.startswith("mmlu_") or task_name == "mmlu":
            continue
        for metric_name in PRIMARY_METRICS["mmlu"]:
            if metric_name in metrics:
                breakdown[task_name] = float(metrics[metric_name])
                break
    return breakdown


def _resolve_batch_sizes(
    batch_size: int | str | Mapping[str, int | str],
) -> tuple[int | str, int | str]:
    if isinstance(batch_size, Mapping):
        zero_shot_batch_size = batch_size.get(
            "zero_shot",
            batch_size.get("default", 1),
        )
        five_shot_batch_size = batch_size.get(
            "five_shot",
            batch_size.get("default", zero_shot_batch_size),
        )
        return zero_shot_batch_size, five_shot_batch_size
    return batch_size, batch_size


def run_lm_harness_eval(
    config: Any,
    checkpoint_path: str | Path,
    *,
    model: Any | None = None,
    checkpoint_info: Dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    device: str = "cuda",
    batch_size: int | str | Mapping[str, int | str] = 1,
    limit: int | float | None = None,
    zero_shot_tasks: Sequence[str] = ZERO_SHOT_TASKS,
    five_shot_tasks: Sequence[str] = FIVE_SHOT_TASKS,
) -> Dict[str, Any]:
    if model is None:
        model, checkpoint_info = load_model_for_eval(
            config=config,
            checkpoint_path=checkpoint_path,
            device=device,
            dtype=torch.bfloat16 if device.startswith("cuda") else None,
        )
    tokenizer = _load_tokenizer_for_model(config)
    zero_shot_batch_size, five_shot_batch_size = _resolve_batch_sizes(batch_size)

    zero_shot_harness_model = (
        _build_harness_model(
            model,
            tokenizer,
            device=device,
            batch_size=zero_shot_batch_size,
        )
        if zero_shot_tasks
        else None
    )
    if five_shot_tasks:
        if (
            zero_shot_harness_model is not None
            and five_shot_batch_size == zero_shot_batch_size
        ):
            five_shot_harness_model = zero_shot_harness_model
        else:
            five_shot_harness_model = _build_harness_model(
                model,
                tokenizer,
                device=device,
                batch_size=five_shot_batch_size,
            )
    else:
        five_shot_harness_model = None

    zero_shot_eval = (
        _simple_evaluate(
            model=zero_shot_harness_model,
            tasks=list(zero_shot_tasks),
            num_fewshot=0,
            batch_size=zero_shot_batch_size,
            device=device,
            limit=limit,
            log_samples=False,
        )
        if zero_shot_tasks
        else {"results": {}}
    )
    five_shot_eval = (
        _simple_evaluate(
            model=five_shot_harness_model,
            tasks=list(five_shot_tasks),
            num_fewshot=5,
            batch_size=five_shot_batch_size,
            device=device,
            limit=limit,
            log_samples=False,
        )
        if five_shot_tasks
        else {"results": {}}
    )

    results: Dict[str, float] = {}
    for task_name in zero_shot_tasks:
        results[task_name] = _extract_primary_metric(zero_shot_eval, task_name)
    for task_name in five_shot_tasks:
        results[task_name] = _extract_primary_metric(five_shot_eval, task_name)

    payload = build_results_payload(
        task="lm_harness",
        checkpoint_path=checkpoint_path,
        checkpoint_info=checkpoint_info,
        config=config,
        results=results,
        metadata={
            "device": device,
            "batch_size": batch_size,
            "limit": limit,
            "dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
            "torch_version": torch.__version__,
            "shots": {
                "zero_shot": list(zero_shot_tasks),
                "five_shot": list(five_shot_tasks),
            },
            "mmlu_subjects": _collect_mmlu_breakdown(five_shot_eval),
            "raw_results": {
                "zero_shot": zero_shot_eval,
                "five_shot": five_shot_eval,
            },
        },
    )

    if output_path is not None:
        write_results_json(payload, output_path)

    print("\n── LM Harness Results ──────────────────────────")
    for task, score in results.items():
        print(f"  {task:<30} {score:.4f}")
    print("────────────────────────────────────────────────\n")

    return payload
