from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

from evals.efficiency import run_efficiency_eval
from evals.lm_harness_runner import run_lm_harness_eval
from evals.perplexity import run_perplexity_eval
from evals.results_schema import infer_checkpoint_step
from evals.results_schema import log_results_to_wandb


SUPPORTED_TASKS = ("perplexity", "lm_harness", "efficiency")

# Defaults used when neither CLI nor YAML provides a value
_EVAL_DEFAULTS = {
    "stride": 512,
    "max_documents": None,
    "max_eval_length": 2048,
    "batch_size": 32,
    "lm_harness_batch_size": "1",
    "limit": None,
    "seq_len": 1024,
    "warmup_iters": 10,
    "benchmark_iters": 100,
}


def load_experiment_config(config_path_or_name: str, overrides=None):
    from src.utils.config_loader import load_experiment_config as _load

    return _load(config_path_or_name, overrides)


def _get_eval_param(config: Any, key: str, cli_value: Any, sentinel: Any = None) -> Any:
    """
    Resolve an eval parameter with priority: CLI > YAML eval section > hardcoded default.
    `sentinel` is the value that means "not provided by CLI" (usually None or argparse default).
    """
    from omegaconf import OmegaConf

    if cli_value is not sentinel:
        return cli_value
    from omegaconf import DictConfig

    if not isinstance(config, DictConfig):
        config = OmegaConf.create(config)
    yaml_val = OmegaConf.select(config, f"eval.{key}", default=None)
    if yaml_val is not None:
        return yaml_val
    return _EVAL_DEFAULTS.get(key, sentinel)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run post-training evaluation on a saved T-MoE checkpoint."
    )
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--all-checkpoints", action="store_true")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda")
    # All eval params are optional — fall back to YAML eval section, then hardcoded defaults
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--max-eval-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lm-harness-batch-size", type=int, default=None)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--benchmark-iters", type=int, default=None)
    parser.add_argument("--reference-checkpoint", default=None)
    parser.add_argument("--reference-config", default=None)
    parser.add_argument("overrides", nargs="*")
    return parser


def _default_output_dir(config) -> Path:
    experiment_name = config.get("experiment_name", "experiment")
    return Path("outputs") / experiment_name / "eval"


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    step = infer_checkpoint_step(path)
    if step is None:
        return (10**18, path.name)
    return (step, path.name)


def _resolve_checkpoint_paths(checkpoint: str, all_checkpoints: bool) -> list[Path]:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_dir() or all_checkpoints:
        checkpoint_dir = (
            checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
        )
        checkpoint_paths = sorted(
            checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=_checkpoint_sort_key,
        )
        if not checkpoint_paths:
            raise FileNotFoundError(
                f"No checkpoint_step_*.pt files found in '{checkpoint_dir}'"
            )
        return checkpoint_paths
    return [checkpoint_path]


def _resolve_output_path(
    task: str,
    config,
    output_dir: str | None,
    *,
    checkpoint_path: Path | None = None,
    multiple_checkpoints: bool = False,
) -> Path:
    eval_dir = Path(output_dir) if output_dir else _default_output_dir(config)
    if multiple_checkpoints:
        if checkpoint_path is None:
            raise ValueError(
                "checkpoint_path is required when multiple_checkpoints=True"
            )
        return eval_dir / "history" / checkpoint_path.stem / f"{task}.json"
    return eval_dir / f"{task}.json"


def _get_dist_info() -> tuple[int, int]:
    """Read rank/world_size from torchrun env vars. Returns (0, 1) if not distributed."""
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


def _init_distributed(rank: int, world_size: int) -> None:
    if world_size <= 1:
        return
    import atexit
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        atexit.register(dist.destroy_process_group)


def run_task(args: argparse.Namespace):
    rank, world_size = _get_dist_info()
    _init_distributed(rank, world_size)

    device = args.device
    if world_size > 1 and device == "cuda":
        device = f"cuda:{rank}"

    config = load_experiment_config(args.config, args.overrides or [])
    checkpoint_paths = _resolve_checkpoint_paths(args.checkpoint, args.all_checkpoints)
    multiple_checkpoints = len(checkpoint_paths) > 1

    # Resolve all eval params: CLI > YAML eval section > hardcoded defaults
    stride = _get_eval_param(config, "stride", args.stride)
    max_documents = _get_eval_param(config, "max_documents", args.max_documents)
    max_eval_length = _get_eval_param(config, "max_eval_length", args.max_eval_length)
    batch_size = _get_eval_param(config, "batch_size", args.batch_size)
    lm_batch_size = _get_eval_param(
        config, "lm_harness_batch_size", args.lm_harness_batch_size
    )
    # lm_harness uses string batch sizes; fall back to --batch-size if not set separately
    if (
        lm_batch_size == _EVAL_DEFAULTS["lm_harness_batch_size"]
        and args.batch_size is not None
    ):
        lm_batch_size = str(args.batch_size)
    limit = _get_eval_param(config, "limit", args.limit)
    seq_len = _get_eval_param(config, "seq_len", args.seq_len)
    warmup_iters = _get_eval_param(config, "warmup_iters", args.warmup_iters)
    benchmark_iters = _get_eval_param(config, "benchmark_iters", args.benchmark_iters)

    reference_config = None
    if args.task == "efficiency" and args.reference_config:
        reference_config = load_experiment_config(args.reference_config, [])

    payloads = []
    for checkpoint_path in checkpoint_paths:
        output_path = _resolve_output_path(
            args.task,
            config,
            args.output_dir,
            checkpoint_path=checkpoint_path,
            multiple_checkpoints=multiple_checkpoints,
        )

        from evals.loading import load_model_for_eval
        import torch

        autocast_dtype = torch.bfloat16 if device.startswith("cuda") else None
        model, checkpoint_info = load_model_for_eval(
            config=config,
            checkpoint_path=checkpoint_path,
            device=device,
            dtype=autocast_dtype,
        )

        if args.task == "perplexity":
            payload = run_perplexity_eval(
                config=config,
                checkpoint_path=checkpoint_path,
                model=model,
                checkpoint_info=checkpoint_info,
                output_path=output_path,
                device=device,
                stride=stride,
                max_documents=max_documents,
                max_eval_length=max_eval_length,
                batch_size=batch_size,
                rank=rank,
                world_size=world_size,
            )
        elif args.task == "lm_harness":
            payload = run_lm_harness_eval(
                config=config,
                checkpoint_path=checkpoint_path,
                model=model,
                checkpoint_info=checkpoint_info,
                output_path=output_path,
                device=device,
                batch_size=lm_batch_size,
                limit=limit,
            )
        elif args.task == "efficiency":
            payload = run_efficiency_eval(
                config=config,
                checkpoint_path=checkpoint_path,
                model=model,
                checkpoint_info=checkpoint_info,
                output_path=output_path,
                device=device,
                seq_len=seq_len,
                warmup_iters=warmup_iters,
                benchmark_iters=benchmark_iters,
                reference_checkpoint_path=args.reference_checkpoint,
                reference_config=reference_config,
            )
        else:
            raise NotImplementedError(f"Task '{args.task}' is not implemented yet.")

        if rank == 0:
            log_results_to_wandb(payload, config=config)
        payloads.append(payload)

    if multiple_checkpoints:
        return payloads
    return payloads[0]


def main(argv: Sequence[str] | None = None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    return run_task(args)


if __name__ == "__main__":
    main()
