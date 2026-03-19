from __future__ import annotations

import hashlib
import json
import numbers
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency in lightweight envs
    WANDB_AVAILABLE = False
    wandb = None

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - exercised implicitly in lightweight envs
    class _OmegaConfShim:
        @staticmethod
        def is_config(value: Any) -> bool:
            return False

        @staticmethod
        def to_container(config: Any, resolve: bool = True) -> Any:
            return config

        @staticmethod
        def select(config: Any, key: str, default: Any = None) -> Any:
            current = config
            for part in key.split("."):
                if not isinstance(current, dict) or part not in current:
                    return default
                current = current[part]
            return current

    OmegaConf = _OmegaConfShim()

EVAL_WANDB_RUN_VERSION = 5


def _cfg_select(config: Any, key: str, default: Any = None) -> Any:
    if OmegaConf.is_config(config):
        return OmegaConf.select(config, key, default=default)

    current = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _to_plain_python(value: Any) -> Any:
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)

    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, str | bool) or isinstance(value, numbers.Number):
        return value
    if isinstance(value, dict):
        return {str(k): _to_plain_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_python(v) for v in value]
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _to_plain_python(value.tolist())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def get_git_commit(cwd: str | Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def infer_checkpoint_step(
    checkpoint_path: str | Path,
    checkpoint_info: Dict[str, Any] | None = None,
) -> int | None:
    if checkpoint_info and checkpoint_info.get("step") is not None:
        return int(checkpoint_info["step"])

    checkpoint_name = Path(checkpoint_path).stem
    parts = checkpoint_name.split("_")
    if parts[-2:] and len(parts) >= 2 and parts[-2] == "step":
        try:
            return int(parts[-1])
        except ValueError:
            return None
    return None


def build_results_payload(
    *,
    task: str,
    checkpoint_path: str | Path,
    config: Any,
    results: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
    checkpoint_info: Dict[str, Any] | None = None,
    experiment_name: str | None = None,
    eval_timestamp: str | None = None,
    git_commit: str | None = None,
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    experiment_name = experiment_name or _cfg_select(
        config, "experiment_name", checkpoint_path.parent.name
    )

    payload = {
        "experiment_name": experiment_name,
        "checkpoint_step": infer_checkpoint_step(
            checkpoint_path=checkpoint_path,
            checkpoint_info=checkpoint_info,
        ),
        "checkpoint_path": str(checkpoint_path),
        "eval_timestamp": eval_timestamp or _utc_now_iso(),
        "git_commit": git_commit or get_git_commit(checkpoint_path.parent),
        "task": task,
        "config": _to_plain_python(config),
        "results": _to_plain_python(results),
        "metadata": _to_plain_python(metadata or {}),
    }
    return payload


def write_results_json(payload: Dict[str, Any], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_plain_python(payload), handle, indent=2)
        handle.write("\n")
    return output_path


def flatten_scalars(
    data: Dict[str, Any],
    *,
    prefix: str = "",
    separator: str = "/",
) -> Dict[str, float | int | bool | str]:
    flattened: Dict[str, float | int | bool | str] = {}
    for key, value in data.items():
        full_key = f"{prefix}{separator}{key}" if prefix else str(key)
        value = _to_plain_python(value)

        if isinstance(value, dict):
            flattened.update(
                flatten_scalars(value, prefix=full_key, separator=separator)
            )
            continue

        if isinstance(value, str | bool) or isinstance(value, numbers.Number):
            flattened[full_key] = value

    return flattened


def _eval_wandb_project(config: Any) -> str:
    project = _cfg_select(config, "logging.project")
    if isinstance(project, str) and project.strip():
        return project.strip()
    env_project = os.environ.get("WANDB_PROJECT")
    if isinstance(env_project, str) and env_project.strip():
        return env_project.strip()
    return "tmoe"


def _eval_wandb_entity(config: Any) -> str | None:
    entity = _cfg_select(config, "logging.entity")
    if isinstance(entity, str) and entity.strip():
        return entity.strip()
    env_entity = os.environ.get("WANDB_ENTITY")
    if isinstance(env_entity, str) and env_entity.strip():
        return env_entity.strip()
    return None


def _eval_run_name(payload: Dict[str, Any], config: Any) -> str:
    experiment_name = payload.get("experiment_name") or _cfg_select(
        config, "experiment_name", "experiment"
    )
    task = payload.get("task") or "eval"
    return f"eval/{experiment_name}/{task}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "experiment"


def _eval_run_id(payload: Dict[str, Any], config: Any) -> str:
    experiment_name = str(
        payload.get("experiment_name")
        or _cfg_select(config, "experiment_name", "experiment")
    )
    task = str(payload.get("task") or "eval")
    identity = f"{experiment_name}:{task}"
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]
    return (
        f"eval-v{EVAL_WANDB_RUN_VERSION}-"
        f"{_slugify(experiment_name)[:24]}-{_slugify(task)[:16]}-{digest}"
    )


def _wandb_history_payload(payload: Dict[str, Any]) -> Dict[str, float | int | bool | str]:
    task = str(payload.get("task", "eval"))
    return flatten_scalars(payload.get("results", {}), prefix=f"eval/{task}")


def _wandb_summary_payload(payload: Dict[str, Any]) -> Dict[str, float | int | bool | str]:
    task = str(payload.get("task", "eval"))
    metadata = dict(payload.get("metadata", {}))
    metadata.pop("raw_results", None)
    metadata.pop("mmlu_subjects", None)
    scalars = flatten_scalars(metadata, prefix=f"eval/{task}/meta")

    checkpoint_step = payload.get("checkpoint_step")
    if checkpoint_step is not None:
        scalars["eval/latest_checkpoint_step"] = int(checkpoint_step)

    git_commit = payload.get("git_commit")
    if isinstance(git_commit, str) and git_commit:
        scalars["eval/git_commit"] = git_commit

    return scalars


def _build_mmlu_table(payload: Dict[str, Any]):
    mmlu_subjects = payload.get("metadata", {}).get("mmlu_subjects", {})
    if not isinstance(mmlu_subjects, dict) or not mmlu_subjects:
        return None

    table = wandb.Table(columns=["subject", "accuracy"])
    for subject, accuracy in sorted(mmlu_subjects.items()):
        table.add_data(subject, accuracy)
    return table


def log_results_to_wandb(
    payload: Dict[str, Any],
    *,
    config: Any | None = None,
) -> bool:
    if not WANDB_AVAILABLE:
        return False

    config = config or payload.get("config") or {}
    if _cfg_select(config, "logging.enabled", True) is False:
        return False
    if _cfg_select(config, "logging.mode") == "disabled":
        return False

    init_kwargs = {
        "project": _eval_wandb_project(config),
        "name": _eval_run_name(payload, config),
        "id": _eval_run_id(payload, config),
        "resume": "allow",
        "group": f"eval/{payload.get('experiment_name') or _cfg_select(config, 'experiment_name', 'experiment')}",
        "job_type": "eval",
        "config": {
            "experiment_name": payload.get("experiment_name")
            or _cfg_select(config, "experiment_name", "experiment"),
            "eval_schema_version": EVAL_WANDB_RUN_VERSION,
        },
    }
    entity = _eval_wandb_entity(config)
    if entity is not None:
        init_kwargs["entity"] = entity

    mode = _cfg_select(config, "logging.mode")
    if isinstance(mode, str) and mode in {"online", "offline"}:
        init_kwargs["mode"] = mode

    try:
        run = wandb.init(**init_kwargs)
    except Exception:
        return False

    if run is None:
        return False

    checkpoint_step = payload.get("checkpoint_step")
    log_payload: Dict[str, Any] = dict(_wandb_history_payload(payload))
    mmlu_table = _build_mmlu_table(payload)
    if checkpoint_step is None:
        run.log(log_payload)
        if mmlu_table is not None:
            run.log({f"eval/{payload.get('task', 'lm_harness')}/mmlu_subjects": mmlu_table})
    else:
        run.log(log_payload, step=int(checkpoint_step))
        if mmlu_table is not None:
            run.log(
                {f"eval/{payload.get('task', 'lm_harness')}/mmlu_subjects": mmlu_table},
                step=int(checkpoint_step),
            )

    summary_payload = _wandb_summary_payload(payload)
    if summary_payload:
        run.summary.update(summary_payload)
    finish = getattr(run, "finish", None)
    if callable(finish):
        finish()
    else:
        wandb.finish()
    return True
