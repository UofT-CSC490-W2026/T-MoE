from __future__ import annotations

import os
import torch

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def _detect_dtype() -> torch.dtype:
    env = os.environ.get("TMOE_DTYPE", "").lower().strip()
    if env:
        if env in _DTYPE_MAP:
            return _DTYPE_MAP[env]
        raise ValueError(
            f"Unknown TMOE_DTYPE={env!r}. Valid: {', '.join(sorted(_DTYPE_MAP))}"
        )

    if torch.cuda.is_available():
        if torch.cuda.get_device_capability()[0] >= 8:
            return torch.bfloat16
        return torch.float16

    return torch.float32


COMPUTE_DTYPE: torch.dtype = _detect_dtype()


def set_compute_dtype(dtype: torch.dtype) -> None:
    global COMPUTE_DTYPE
    COMPUTE_DTYPE = dtype


def get_compute_dtype() -> torch.dtype:
    return COMPUTE_DTYPE


def is_mixed_precision() -> bool:
    return COMPUTE_DTYPE in (torch.bfloat16, torch.float16)


def needs_grad_scaler() -> bool:
    """Only float16 needs GradScaler; bfloat16 has the same exponent range as float32."""
    return COMPUTE_DTYPE == torch.float16
