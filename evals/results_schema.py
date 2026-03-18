from __future__ import annotations

import json
import numbers
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

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
