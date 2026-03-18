from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from evals.efficiency import run_efficiency_eval
from evals.lm_harness_runner import run_lm_harness_eval
from evals.perplexity import run_perplexity_eval
from evals.results_schema import log_results_to_wandb


SUPPORTED_TASKS = ("perplexity", "lm_harness", "efficiency")


def load_experiment_config(config_path_or_name: str, overrides=None):
    from src.utils.config_loader import load_experiment_config as _load

    return _load(config_path_or_name, overrides)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run post-training evaluation on a saved T-MoE checkpoint."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=SUPPORTED_TASKS,
        help="Evaluation task to run.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the saved checkpoint to evaluate.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Experiment config path or bare experiment name.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where task JSON artifacts should be written. Defaults to outputs/<experiment>/eval.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run evaluation on, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Sliding-window stride for perplexity evaluation.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Optional cap on the number of evaluation documents, useful for smoke tests.",
    )
    parser.add_argument(
        "--batch-size",
        default=1,
        help="Batch size for lm-evaluation-harness tasks.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="Optional example limit for lm-evaluation-harness, useful for smoke tests.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1024,
        help="Sequence length for inference efficiency profiling.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=10,
        help="Number of warmup iterations for efficiency profiling.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=100,
        help="Number of timed iterations for efficiency profiling.",
    )
    parser.add_argument(
        "--reference-checkpoint",
        default=None,
        help="Optional reference checkpoint path for router overhead ratio benchmarking.",
    )
    parser.add_argument(
        "--reference-config",
        default=None,
        help="Optional config for the reference checkpoint. Defaults to --config.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional OmegaConf dotlist overrides, e.g. training.lr=1e-4",
    )
    return parser


def _default_output_dir(config) -> Path:
    experiment_name = config.get("experiment_name", "experiment")
    return Path("outputs") / experiment_name / "eval"


def _resolve_output_path(task: str, config, output_dir: str | None) -> Path:
    eval_dir = Path(output_dir) if output_dir else _default_output_dir(config)
    return eval_dir / f"{task}.json"


def run_task(args: argparse.Namespace):
    config = load_experiment_config(args.config, args.overrides or [])
    output_path = _resolve_output_path(args.task, config, args.output_dir)

    if args.task == "perplexity":
        payload = run_perplexity_eval(
            config=config,
            checkpoint_path=args.checkpoint,
            output_path=output_path,
            device=args.device,
            stride=args.stride,
            max_documents=args.max_documents,
        )
    elif args.task == "lm_harness":
        payload = run_lm_harness_eval(
            config=config,
            checkpoint_path=args.checkpoint,
            output_path=output_path,
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
        )
    elif args.task == "efficiency":
        reference_config = None
        if args.reference_config:
            reference_config = load_experiment_config(args.reference_config, [])
        payload = run_efficiency_eval(
            config=config,
            checkpoint_path=args.checkpoint,
            output_path=output_path,
            device=args.device,
            seq_len=args.seq_len,
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
            reference_checkpoint_path=args.reference_checkpoint,
            reference_config=reference_config,
        )
    else:
        raise NotImplementedError(f"Task '{args.task}' is not implemented yet.")

    log_results_to_wandb(payload, config=config)
    return payload


def main(argv: Sequence[str] | None = None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    return run_task(args)


if __name__ == "__main__":
    main()
