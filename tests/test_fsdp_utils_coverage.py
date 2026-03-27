"""Minimal coverage tests for src/training/fsdp_utils.py."""
from __future__ import annotations

import os
from unittest.mock import patch, MagicMock

import pytest
import torch
import torch.nn as nn


# ── init_distributed ──────────────────────────────────────────────────────────

def test_init_distributed_no_rank_env():
    from src.training.fsdp_utils import init_distributed
    with patch.dict(os.environ, {}, clear=True):
        result = init_distributed()
    assert result == (False, 0, 0, 1)


def test_init_distributed_no_cuda(monkeypatch):
    """Lines 29-39: RANK set but no CUDA → RuntimeError."""
    from src.training.fsdp_utils import init_distributed
    env = {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}
    with patch.dict(os.environ, env):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="CUDA"):
                init_distributed()


def test_init_distributed_local_rank_exceeds_devices(monkeypatch):
    """Line 44: LOCAL_RANK >= device_count → RuntimeError."""
    from src.training.fsdp_utils import init_distributed
    env = {"RANK": "0", "LOCAL_RANK": "5", "WORLD_SIZE": "1"}
    with patch.dict(os.environ, env):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=1):
                with pytest.raises(RuntimeError, match="LOCAL_RANK"):
                    init_distributed()


def test_init_distributed_success(monkeypatch):
    """Lines 49+: successful distributed init (mocked)."""
    from src.training.fsdp_utils import init_distributed
    env = {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"}
    with patch.dict(os.environ, env):
        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.device_count", return_value=2):
                with patch("torch.distributed.init_process_group"):
                    with patch("torch.cuda.set_device"):
                        with patch("torch.distributed.barrier"):
                            result = init_distributed()
    assert result == (True, 0, 0, 2)


# ── cleanup_distributed ───────────────────────────────────────────────────────

def test_cleanup_distributed_not_initialized():
    from src.training.fsdp_utils import cleanup_distributed
    with patch("torch.distributed.is_available", return_value=True):
        with patch("torch.distributed.is_initialized", return_value=False):
            cleanup_distributed()  # no-op


def test_cleanup_distributed_initialized():
    from src.training.fsdp_utils import cleanup_distributed
    with patch("torch.distributed.is_available", return_value=True):
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.destroy_process_group") as mock_destroy:
                cleanup_distributed()
    mock_destroy.assert_called_once()


# ── is_main_process ───────────────────────────────────────────────────────────

def test_is_main_process_not_distributed():
    from src.training.fsdp_utils import is_main_process
    with patch("torch.distributed.is_available", return_value=False):
        assert is_main_process() is True


def test_is_main_process_rank0():
    from src.training.fsdp_utils import is_main_process
    with patch("torch.distributed.is_available", return_value=True):
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_rank", return_value=0):
                assert is_main_process() is True


def test_is_main_process_rank1():
    from src.training.fsdp_utils import is_main_process
    with patch("torch.distributed.is_available", return_value=True):
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.get_rank", return_value=1):
                assert is_main_process() is False


# ── get_model_for_attr_access ─────────────────────────────────────────────────

def test_get_model_for_attr_access_plain():
    from src.training.fsdp_utils import get_model_for_attr_access
    model = nn.Linear(4, 4)
    assert get_model_for_attr_access(model) is model


def test_get_model_for_attr_access_ddp():
    from src.training.fsdp_utils import get_model_for_attr_access
    from torch.nn.parallel import DistributedDataParallel as DDP
    inner = nn.Linear(4, 4)
    mock_ddp = MagicMock(spec=DDP)
    mock_ddp.module = inner
    assert get_model_for_attr_access(mock_ddp) is inner


# ── wrap_model_for_distributed ────────────────────────────────────────────────

def test_wrap_model_for_distributed_ddp():
    """Lines 107-130: dispatches to DDP by default."""
    from src.training.fsdp_utils import wrap_model_for_distributed

    model = nn.Linear(4, 4)
    cfg = MagicMock()
    cfg.distributed = MagicMock()
    cfg.distributed.strategy = "ddp"

    mock_wrapped = MagicMock()
    with patch("src.training.fsdp_utils.wrap_model_with_ddp", return_value=mock_wrapped) as mock_ddp:
        result = wrap_model_for_distributed(model, cfg, local_rank=0, device=torch.device("cpu"))
    mock_ddp.assert_called_once()
    assert result is mock_wrapped


def test_wrap_model_for_distributed_fsdp():
    """Lines 107-130: dispatches to FSDP when strategy=fsdp."""
    from src.training.fsdp_utils import wrap_model_for_distributed

    model = nn.Linear(4, 4)
    cfg = MagicMock()
    cfg.distributed = MagicMock()
    cfg.distributed.strategy = "fsdp"

    mock_wrapped = MagicMock()
    with patch("src.training.fsdp_utils.wrap_model_with_fsdp", return_value=mock_wrapped) as mock_fsdp:
        result = wrap_model_for_distributed(model, cfg, local_rank=0, device=torch.device("cpu"))
    mock_fsdp.assert_called_once()
    assert result is mock_wrapped


def test_wrap_model_for_distributed_no_dist_cfg():
    """Lines 107-130: no distributed config → defaults to ddp."""
    from src.training.fsdp_utils import wrap_model_for_distributed

    model = nn.Linear(4, 4)
    cfg = MagicMock(spec=[])  # no distributed attr

    mock_wrapped = MagicMock()
    with patch("src.training.fsdp_utils.wrap_model_with_ddp", return_value=mock_wrapped):
        result = wrap_model_for_distributed(model, cfg, local_rank=0, device=torch.device("cpu"))
    assert result is mock_wrapped


# ── wrap_model_with_ddp ───────────────────────────────────────────────────────

def test_wrap_model_with_ddp():
    """Lines 135-148: wraps with DDP."""
    from src.training.fsdp_utils import wrap_model_with_ddp

    model = nn.Linear(4, 4)
    cfg = MagicMock()
    cfg.distributed = MagicMock()
    cfg.distributed.find_unused_parameters = False

    mock_ddp = MagicMock()
    with patch("torch.nn.parallel.DistributedDataParallel", return_value=mock_ddp):
        with patch("torch.distributed.is_initialized", return_value=False):
            with patch("src.training.fsdp_utils.is_main_process", return_value=True):
                result = wrap_model_with_ddp(model, local_rank=0, cfg=cfg)
    assert result is mock_ddp


def test_wrap_model_with_ddp_with_barrier():
    """Lines 135-148: barrier called when dist initialized."""
    from src.training.fsdp_utils import wrap_model_with_ddp

    model = nn.Linear(4, 4)
    cfg = MagicMock()
    cfg.distributed = MagicMock()
    cfg.distributed.find_unused_parameters = True

    mock_ddp = MagicMock()
    with patch("torch.nn.parallel.DistributedDataParallel", return_value=mock_ddp):
        with patch("torch.distributed.is_initialized", return_value=True):
            with patch("torch.distributed.barrier") as mock_barrier:
                with patch("src.training.fsdp_utils.is_main_process", return_value=False):
                    wrap_model_with_ddp(model, local_rank=0, cfg=cfg)
    mock_barrier.assert_called_once()


# ── _get_fsdp_wrap_targets ────────────────────────────────────────────────────

def test_get_fsdp_wrap_targets_gpt_neo():
    """Lines 157+: returns GPTNeoBlock for gpt-neo model."""
    from src.training.fsdp_utils import _get_fsdp_wrap_targets

    cfg = MagicMock()
    cfg.model.model_key = "gpt-neo-125m"

    with patch("src.configs.model.model_lookup", return_value={"model_type": "gpt_neo"}):
        targets = _get_fsdp_wrap_targets(cfg)
    assert len(targets) == 1


def test_get_fsdp_wrap_targets_qwen2():
    """Lines 157+: returns Qwen2DecoderLayer for qwen2 model."""
    from src.training.fsdp_utils import _get_fsdp_wrap_targets

    cfg = MagicMock()
    cfg.model.model_key = "qwen2-1.5b"

    with patch("src.configs.model.model_lookup", return_value={"model_type": "qwen2"}):
        targets = _get_fsdp_wrap_targets(cfg)
    assert len(targets) == 1


# ── wrap_model_with_fsdp ──────────────────────────────────────────────────────

def test_wrap_model_with_fsdp():
    """Lines 157-243: wraps with FSDP (all mocked)."""
    from src.training.fsdp_utils import wrap_model_with_fsdp

    model = nn.Linear(4, 4)
    cfg = MagicMock()
    cfg.distributed = MagicMock()
    cfg.distributed.mixed_precision = False
    cfg.distributed.cpu_offload = False
    cfg.distributed.activation_checkpointing = False
    cfg.distributed.sharding_strategy = "SHARD_GRAD_OP"

    # Use a real class so isinstance checks work
    class _FakeFSDP(nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()
        def modules(self):
            return iter([self])

    _FakeFSDP()

    with patch("src.training.fsdp_utils._get_fsdp_wrap_targets", return_value={nn.Linear}), \
         patch("torch.distributed.fsdp.wrap.ModuleWrapPolicy", return_value=MagicMock()), \
         patch("torch.distributed.is_initialized", return_value=False), \
         patch("src.training.fsdp_utils.is_main_process", return_value=True), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        # Patch FSDP inside the module namespace
        import torch.distributed.fsdp as fsdp_mod
        with patch.object(fsdp_mod, "FullyShardedDataParallel", _FakeFSDP):
            result = wrap_model_with_fsdp(model, cfg, device=torch.device("cpu"))
    assert isinstance(result, _FakeFSDP)


def test_wrap_model_with_fsdp_with_mixed_precision_and_checkpointing():
    """Lines 157-243: mixed_precision + activation_checkpointing paths."""
    from src.training.fsdp_utils import wrap_model_with_fsdp

    model = nn.Linear(4, 4)
    cfg = MagicMock()
    cfg.distributed = MagicMock()
    cfg.distributed.mixed_precision = True
    cfg.distributed.cpu_offload = True
    cfg.distributed.activation_checkpointing = True
    cfg.distributed.sharding_strategy = "FULL_SHARD"

    class _FakeFSDP(nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()
        def modules(self):
            return iter([self])

    with patch("src.training.fsdp_utils._get_fsdp_wrap_targets", return_value={nn.Linear}), \
         patch("torch.distributed.fsdp.wrap.ModuleWrapPolicy", return_value=MagicMock()), \
         patch("torch.distributed.is_initialized", return_value=True), \
         patch("torch.distributed.barrier"), \
         patch("src.training.fsdp_utils.is_main_process", return_value=True), \
         patch("src.training.fsdp_utils._apply_activation_checkpointing") as mock_ckpt, \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.is_mixed_precision", return_value=True):
        import torch.distributed.fsdp as fsdp_mod
        with patch.object(fsdp_mod, "FullyShardedDataParallel", _FakeFSDP):
            wrap_model_with_fsdp(model, cfg, device=torch.device("cpu"))
    mock_ckpt.assert_called_once()


def test_wrap_model_with_fsdp_no_shard_strategy():
    """Lines 157-243: unknown sharding strategy + _fsdp_count==0 warning."""
    from src.training.fsdp_utils import wrap_model_with_fsdp

    model = nn.Linear(4, 4)
    cfg = MagicMock()
    cfg.distributed = MagicMock()
    cfg.distributed.mixed_precision = False
    cfg.distributed.cpu_offload = False
    cfg.distributed.activation_checkpointing = False
    cfg.distributed.sharding_strategy = "UNKNOWN_STRATEGY"

    class _FakeFSDP(nn.Module):
        def __init__(self, *a, **kw):
            super().__init__()
        def modules(self):
            return iter([])  # empty → _fsdp_count == 0 → warning printed

    with patch("src.training.fsdp_utils._get_fsdp_wrap_targets", return_value={nn.Linear}), \
         patch("torch.distributed.fsdp.wrap.ModuleWrapPolicy", return_value=MagicMock()), \
         patch("torch.distributed.is_initialized", return_value=False), \
         patch("src.training.fsdp_utils.is_main_process", return_value=True), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        import torch.distributed.fsdp as fsdp_mod
        with patch.object(fsdp_mod, "FullyShardedDataParallel", _FakeFSDP):
            result = wrap_model_with_fsdp(model, cfg, device=torch.device("cpu"))
    assert isinstance(result, _FakeFSDP)


# ── _apply_activation_checkpointing ──────────────────────────────────────────

def test_apply_activation_checkpointing():
    """Lines 247-258: applies activation checkpointing."""
    from src.training.fsdp_utils import _apply_activation_checkpointing

    model = nn.Linear(4, 4)
    with patch("torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing") as mock_apply, \
         patch("torch.distributed.algorithms._checkpoint.checkpoint_wrapper.checkpoint_wrapper", return_value=MagicMock()), \
         patch("src.training.fsdp_utils.is_main_process", return_value=True):
        _apply_activation_checkpointing(model, nn.Linear)
    mock_apply.assert_called_once()
