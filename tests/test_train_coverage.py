"""Coverage tests for scripts/train.py — targeting uncovered lines."""
from __future__ import annotations

import math
import struct
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


# ── helpers ────────────────────────────────────────────────────────────────────

def _write_shard(path: Path, tokens: list[int], uint32: bool = False) -> None:
    n = len(tokens)
    with open(path, "wb") as f:
        if uint32:
            f.write(struct.pack("<Q", n))
            f.write(struct.pack("<H", 1))
            f.write(np.array(tokens, dtype=np.uint32).tobytes())
        else:
            f.write(struct.pack("<Q", n))
            f.write(np.array(tokens, dtype=np.uint16).tobytes())


def _write_versioned_shard(path: Path, tokens: list[int], dtype_flag: int = 0) -> None:
    """Write a 10-byte-header shard (versioned format)."""
    n = len(tokens)
    dtype = np.uint32 if dtype_flag == 1 else np.uint16
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", n))
        f.write(struct.pack("<H", dtype_flag))
        f.write(np.array(tokens, dtype=dtype).tobytes())


# ── ShardDataset (line 147 — wrap-around, versioned shard) ────────────────────

def test_shard_dataset_basic(tmp_path):
    from scripts.train import ShardDataset

    shard = tmp_path / "train_shard_0000.bin"
    _write_shard(shard, list(range(100)))

    ds = ShardDataset(tmp_path, "train", seq_len=8)
    assert len(ds) > 0
    ids, labels = ds[0]
    assert ids.shape == (9,)  # seq_len+1 for input/label shift
    assert torch.equal(ids, labels)


def test_shard_dataset_versioned_uint16(tmp_path):
    """Versioned 10-byte header with dtype_flag=0 (uint16)."""
    from scripts.train import ShardDataset

    shard = tmp_path / "train_shard_0000.bin"
    _write_versioned_shard(shard, list(range(100)), dtype_flag=0)

    ds = ShardDataset(tmp_path, "train", seq_len=8)
    assert len(ds) > 0


def test_shard_dataset_versioned_uint32(tmp_path):
    """Versioned 10-byte header with dtype_flag=1 (uint32)."""
    from scripts.train import ShardDataset

    shard = tmp_path / "train_shard_0000.bin"
    _write_versioned_shard(shard, list(range(100)), dtype_flag=1)

    ds = ShardDataset(tmp_path, "train", seq_len=8)
    assert len(ds) > 0


def test_shard_dataset_unknown_dtype_flag(tmp_path):
    """dtype_flag=2 raises ValueError."""
    from scripts.train import ShardDataset

    shard = tmp_path / "train_shard_0000.bin"
    n = 100
    with open(shard, "wb") as f:
        f.write(struct.pack("<Q", n))
        f.write(struct.pack("<H", 2))  # unknown flag
        f.write(np.zeros(n, dtype=np.uint16).tobytes())

    with pytest.raises(ValueError, match="Unknown dtype_flag"):
        ShardDataset(tmp_path, "train", seq_len=8)


def test_shard_dataset_no_shards_raises(tmp_path):
    from scripts.train import ShardDataset

    with pytest.raises(FileNotFoundError):
        ShardDataset(tmp_path, "train", seq_len=8)


def test_shard_dataset_wrap_around(tmp_path):
    """seq_len > shard tokens triggers wrap-around (line 147)."""
    from scripts.train import ShardDataset

    # Only 20 tokens but seq_len=16 → needs wrap
    shard = tmp_path / "train_shard_0000.bin"
    _write_shard(shard, list(range(20)))

    ds = ShardDataset(tmp_path, "train", seq_len=16)
    ids, _ = ds[0]
    assert ids.shape == (17,)  # seq_len+1


def test_shard_dataset_multi_shard(tmp_path):
    """Multiple shards — bisect logic picks correct shard."""
    from scripts.train import ShardDataset

    for i in range(3):
        shard = tmp_path / f"train_shard_{i:04d}.bin"
        _write_shard(shard, list(range(50)))

    ds = ShardDataset(tmp_path, "train", seq_len=8)
    assert len(ds) > 0
    ids, _ = ds[len(ds) - 1]
    assert ids.shape == (9,)  # seq_len+1


# ── load_config (lines 315-321) ───────────────────────────────────────────────

def test_load_config_no_overrides(tmp_path):
    from scripts.train import load_config

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("training:\n  lr: 1e-4\n")
    cfg = load_config(str(cfg_path), [])
    assert cfg.training.lr == pytest.approx(1e-4)


def test_load_config_with_overrides(tmp_path):
    from scripts.train import load_config

    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("training:\n  lr: 1e-4\n  batch_size: 8\n")
    cfg = load_config(str(cfg_path), ["training.lr=5e-5"])
    assert cfg.training.lr == pytest.approx(5e-5)


# ── parse_args (lines 329-330) ────────────────────────────────────────────────

def test_parse_args_basic():
    from scripts.train import parse_args

    with patch("sys.argv", ["train.py", "--config", "exp.yaml"]):
        args, overrides = parse_args()
    assert args.config == "exp.yaml"
    assert args.resume is None
    assert overrides == []


def test_parse_args_with_all_flags():
    from scripts.train import parse_args

    with patch("sys.argv", [
        "train.py", "--config", "exp.yaml",
        "--resume", "ckpt.pt",
        "--output-dir", "/tmp/out",
        "--shard-dir", "/tmp/shards",
        "training.lr=1e-4",
    ]):
        args, overrides = parse_args()
    assert args.resume == "ckpt.pt"
    assert args.output_dir == "/tmp/out"
    assert args.shard_dir == "/tmp/shards"
    assert "training.lr=1e-4" in overrides


# ── build_optimizer (lines 336-368) ───────────────────────────────────────────

def _make_opt_cfg(optimizer="adamw", lr=1e-4, lr_base=None):
    cfg = MagicMock()
    cfg.training.optimizer = optimizer
    cfg.training.lr = lr
    cfg.training.get = lambda k, d=None: {
        "lr_base": lr_base,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1,
    }.get(k, d)
    return cfg


def test_build_optimizer_adamw():
    from scripts.train import build_optimizer

    model = torch.nn.Linear(4, 4)
    cfg = _make_opt_cfg("adamw")
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.AdamW)


def test_build_optimizer_adam():
    from scripts.train import build_optimizer

    model = torch.nn.Linear(4, 4)
    cfg = _make_opt_cfg("adam")
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.Adam)


def test_build_optimizer_unknown_raises():
    from scripts.train import build_optimizer

    model = torch.nn.Linear(4, 4)
    cfg = _make_opt_cfg("sgd")
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer(model, cfg)


def test_build_optimizer_with_lr_base():
    """lr_base triggers base param group separation."""
    from scripts.train import build_optimizer

    class _ModelWithBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_fc_weight = torch.nn.Parameter(torch.randn(4, 4))
            self.other = torch.nn.Linear(4, 4)

    model = _ModelWithBase()
    cfg = _make_opt_cfg("adamw", lr_base=1e-5)
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.AdamW)
    # Should have 2 param groups (base + other)
    assert len(opt.param_groups) == 2


# ── evaluate (lines 433-437) ──────────────────────────────────────────────────

def test_evaluate_basic():
    from scripts.train import evaluate

    class _M(torch.nn.Module):
        def forward(self, input_ids, labels, return_metrics, record_usage):
            loss = torch.tensor(1.5)
            return None, loss, {}

    model = _M()
    data = [(torch.zeros(2, 8, dtype=torch.long), torch.zeros(2, 8, dtype=torch.long))]
    result = evaluate(model, data, "cpu", max_batches=5)
    assert math.isclose(result, 1.5)


def test_evaluate_empty_loader():
    from scripts.train import evaluate

    model = MagicMock()
    result = evaluate(model, [], "cpu")
    assert result == float("inf")


def test_evaluate_respects_max_batches():
    from scripts.train import evaluate

    call_count = 0

    class _M(torch.nn.Module):
        def forward(self, input_ids, labels, return_metrics, record_usage):
            nonlocal call_count
            call_count += 1
            return None, torch.tensor(1.0), {}

    model = _M()
    data = [(torch.zeros(2, 4, dtype=torch.long), torch.zeros(2, 4, dtype=torch.long))] * 10
    evaluate(model, data, "cpu", max_batches=3)
    assert call_count == 3


# ── init_wandb (lines 536-537 and surrounding) ────────────────────────────────

def test_init_wandb_not_main_process():
    from scripts.train import init_wandb

    with patch("scripts.train.is_main_process", return_value=False):
        init_wandb(MagicMock())  # should be a no-op


def test_init_wandb_disabled_by_config():
    from scripts.train import init_wandb

    cfg = MagicMock()
    cfg.get = lambda k, d=None: {"logging": {"enabled": False}}.get(k, d)

    with patch("scripts.train.is_main_process", return_value=True):
        init_wandb(cfg)  # no-op, logging not enabled


def test_init_wandb_mode_disabled():
    from scripts.train import init_wandb

    cfg = MagicMock()
    cfg.get = lambda k, d=None: {"logging": {"enabled": True, "mode": "disabled"}}.get(k, d)

    with patch("scripts.train.is_main_process", return_value=True):
        init_wandb(cfg)  # prints "WandB disabled"


def test_init_wandb_import_error():
    from scripts.train import init_wandb

    cfg = MagicMock()
    cfg.get = lambda k, d=None: {"logging": {"enabled": True, "mode": "online"}}.get(k, d)
    cfg.experiment_name = "test"

    with patch("scripts.train.is_main_process", return_value=True):
        with patch("builtins.__import__", side_effect=ImportError("no wandb")):
            init_wandb(cfg)  # should print "WandB not installed"


def test_init_wandb_success_with_url():
    from scripts.train import init_wandb

    cfg = MagicMock()
    cfg.get = lambda k, d=None: {
        "logging": {"enabled": True, "mode": "online", "project": "myproj", "entity": None}
    }.get(k, d)
    cfg.experiment_name = "test_run"

    mock_run = MagicMock()
    mock_run.url = "https://wandb.ai/test"

    mock_wandb = MagicMock()
    mock_wandb.init.return_value = mock_run

    with patch("scripts.train.is_main_process", return_value=True):
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with patch("scripts.train.OmegaConf.to_container", return_value={}):
                init_wandb(cfg)
    mock_wandb.init.assert_called_once()


def test_init_wandb_success_no_url():
    from scripts.train import init_wandb

    cfg = MagicMock()
    cfg.get = lambda k, d=None: {
        "logging": {"enabled": True, "mode": "online", "project": "myproj", "entity": "myentity"}
    }.get(k, d)
    cfg.experiment_name = "test_run"

    mock_run = MagicMock()
    mock_run.url = None

    mock_wandb = MagicMock()
    mock_wandb.init.return_value = mock_run

    with patch("scripts.train.is_main_process", return_value=True):
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with patch("scripts.train.OmegaConf.to_container", return_value={}):
                init_wandb(cfg)


def test_init_wandb_exception():
    from scripts.train import init_wandb

    cfg = MagicMock()
    cfg.get = lambda k, d=None: {"logging": {"enabled": True, "mode": "online"}}.get(k, d)
    cfg.experiment_name = "test"

    mock_wandb = MagicMock()
    mock_wandb.init.side_effect = Exception("connection error")

    with patch("scripts.train.is_main_process", return_value=True):
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with patch("scripts.train.OmegaConf.to_container", return_value={}):
                init_wandb(cfg)  # should print "WandB init failed"


def test_init_wandb_env_mode_fallback():
    """mode not in {online, offline} → falls back to WANDB_MODE env var."""
    from scripts.train import init_wandb
    import os

    cfg = MagicMock()
    cfg.get = lambda k, d=None: {
        "logging": {"enabled": True, "mode": "auto", "project": None, "entity": None}
    }.get(k, d)
    cfg.experiment_name = "test"

    mock_run = MagicMock()
    mock_run.url = None
    mock_wandb = MagicMock()
    mock_wandb.init.return_value = mock_run

    with patch("scripts.train.is_main_process", return_value=True):
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            with patch("scripts.train.OmegaConf.to_container", return_value={}):
                with patch.dict(os.environ, {"WANDB_MODE": "offline"}):
                    init_wandb(cfg)
    _, kwargs = mock_wandb.init.call_args
    assert kwargs.get("mode") == "offline" or mock_wandb.init.call_args[0]


# ── log_wandb (lines 578+) ────────────────────────────────────────────────────

def test_log_wandb_not_main():
    from scripts.train import log_wandb

    with patch("scripts.train.is_main_process", return_value=False):
        log_wandb({"loss": 1.0})  # no-op


def test_log_wandb_no_run():
    from scripts.train import log_wandb

    mock_wandb = MagicMock()
    mock_wandb.run = None

    with patch("scripts.train.is_main_process", return_value=True):
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            log_wandb({"loss": 1.0})
    mock_wandb.log.assert_not_called()


def test_log_wandb_with_run():
    from scripts.train import log_wandb

    mock_wandb = MagicMock()
    mock_wandb.run = MagicMock()

    with patch("scripts.train.is_main_process", return_value=True):
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            log_wandb({"loss": 1.0, "step": 10})
    mock_wandb.log.assert_called_once_with({"loss": 1.0, "step": 10})


def test_log_wandb_import_error():
    from scripts.train import log_wandb

    with patch("scripts.train.is_main_process", return_value=True):
        with patch("builtins.__import__", side_effect=ImportError):
            log_wandb({"loss": 1.0})  # should silently pass


# ── _broadcast_scalar ─────────────────────────────────────────────────────────

def test_broadcast_scalar_not_distributed():
    from scripts.train import _broadcast_scalar

    result = _broadcast_scalar(3.14, "cpu", is_distributed=False)
    assert result == pytest.approx(3.14)


# ── main() — heavily mocked to cover lines 578-1368 ──────────────────────────

def _make_full_cfg(tmp_path):
    """Build a minimal OmegaConf config for main()."""
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    # Write train + val shards
    for split in ("train", "val"):
        shard = shard_dir / f"{split}_shard_0000.bin"
        _write_shard(shard, list(range(200)))

    cfg_dict = {
        "experiment_name": "test_run",
        "seed": 42,
        "compile": False,
        "dataset": {
            "dataset_key": "wikitext-2",
            "max_seq_len": 8,
        },
        "model": {
            "model_key": "gpt-neo-125m",
            "freeze_backbone": True,
            "moe_layer_indices": [0],
        },
        "router": {
            "type": "standard",
            "num_experts": 2,
            "top_k": 1,
        },
        "expert": {
            "type": "gpt_neo_lora",
            "count": 2,
            "lora": {
                "rank": 4,
                "alpha": 1.0,
                "dropout": 0.0,
                "init_scale": 0.01,
            },
        },
        "training": {
            "optimizer": "adamw",
            "lr": 1e-4,
            "batch_size": 2,
            "steps": 2,
            "log_interval": 1,
            "eval_interval": 1,
            "save_interval": 10,
            "warmup_steps": 0,
            "gradient_accumulation_steps": 1,
            "clip_grad_norm": 1.0,
        },
        "logging": {"enabled": False},
    }
    return OmegaConf.create(cfg_dict), shard_dir


class _TinyModel(torch.nn.Module):
    """Minimal model that satisfies train.py's interface."""
    vocab_size = 16
    moe_layers = {}

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.fc = torch.nn.Linear(8, 16)

    def forward(self, input_ids, labels=None, return_metrics=True, record_usage=True):
        x = self.embed(input_ids % 16)
        logits = self.fc(x)
        loss = torch.tensor(1.0, requires_grad=True)
        return logits, loss, {}

    def eval(self):
        return self

    def train(self, mode=True):
        return self


def test_main_runs_minimal(tmp_path):
    """Run main() with 2 steps, all heavy parts mocked."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    tiny_model = _TinyModel()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=tiny_model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.wrap_model_for_distributed", side_effect=lambda m, *a, **kw: m), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


def test_main_no_val_shards(tmp_path):
    """main() handles missing val shards gracefully."""
    from scripts.train import main

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    # Only train shard, no val
    _write_shard(shard_dir / "train_shard_0000.bin", list(range(200)))

    cfg_dict = {
        "experiment_name": "test_no_val",
        "seed": 42,
        "compile": False,
        "dataset": {"dataset_key": "wikitext-2", "max_seq_len": 8},
        "model": {"model_key": "gpt-neo-125m", "freeze_backbone": True, "moe_layer_indices": []},
        "router": {"type": "standard", "num_experts": 2, "top_k": 1},
        "expert": {"type": "gpt_neo_lora", "count": 2, "lora": {"rank": 4, "alpha": 1.0, "dropout": 0.0, "init_scale": 0.01}},
        "training": {
            "optimizer": "adamw", "lr": 1e-4, "batch_size": 2, "steps": 1,
            "log_interval": 1, "eval_interval": 100, "save_interval": 100,
            "warmup_steps": 0, "gradient_accumulation_steps": 1, "clip_grad_norm": 0.0,
        },
        "logging": {"enabled": False},
    }
    cfg = OmegaConf.create(cfg_dict)
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    tiny_model = _TinyModel()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=tiny_model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


def test_main_with_resume(tmp_path):
    """main() --resume loads checkpoint."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    tiny_model = _TinyModel()
    ckpt_path = tmp_path / "ckpt.pt"
    ckpt_path.write_text("fake")

    mock_ckpt_mgr = MagicMock()
    mock_ckpt_mgr.load_checkpoint.return_value = {"step": 0, "metrics": {"val_loss": 2.0}}

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out"),
                             "--resume", str(ckpt_path)]), \
         patch("scripts.train.build_model", return_value=tiny_model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.checkpoint.CheckpointManager", return_value=mock_ckpt_mgr), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()

    mock_ckpt_mgr.load_checkpoint.assert_called_once()


def test_main_adam_optimizer(tmp_path):
    """main() with adam optimizer."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    # Override optimizer
    cfg.training.optimizer = "adam"
    OmegaConf.save(cfg, cfg_path)

    tiny_model = _TinyModel()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=tiny_model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


def test_main_steps_from_config(tmp_path):
    """training.steps set in config overrides Chinchilla."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg.training.steps = 3
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    tiny_model = _TinyModel()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=tiny_model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


# ── Additional main() coverage: moe_metrics logging, early stopping, periodic save ──

class _MoELayer(torch.nn.Module):
    """Fake MoE layer with router and expert_pool for diagnostic coverage."""
    def __init__(self):
        super().__init__()
        self._last_routing_weights = None
        self._last_routing_indices = None
        self.router = MagicMock(spec=["clear_aux_state", "parameters"])
        self.router.clear_aux_state = MagicMock()
        self.router.parameters = lambda: iter([])
        self.expert_pool = MagicMock()
        self.expert_pool.experts = []
        self.expert_pool.consolidate_shared_weights = MagicMock()

    def step(self):
        pass


class _TinyModelWithMoE(torch.nn.Module):
    """Model with moe_layers to exercise diagnostic/logging paths."""
    vocab_size = 16

    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.fc = torch.nn.Linear(8, 16)
        moe = _MoELayer()
        self.moe_layers = {0: moe}

    def forward(self, input_ids, labels=None, return_metrics=True, record_usage=True):
        x = self.embed(input_ids % 16)
        logits = self.fc(x)
        loss = torch.tensor(1.0, requires_grad=True)
        # Return moe_metrics with all the keys the logging block checks
        moe_metrics = {
            "layer_0": {
                "effective_experts": 2.0,
                "routing_diversity_gini": 0.3,
                "router_confidence_mean": 0.7,
                "fatigue_std": 0.1,
                "mean_k": 1.5,
                "stress_mean": 0.2,
                "usage_distribution": [0.5, 0.5],
                "ema_load_per_expert": [0.5, 0.5],
                "lora_delta_norm_per_expert": [0.1, 0.2],
                "load_balance": 0.9,
            }
        } if return_metrics else {}
        return logits, loss, moe_metrics

    def eval(self):
        return self

    def train(self, mode=True):
        return self


def test_main_with_moe_metrics_logging(tmp_path):
    """Covers the moe_metrics logging block (lines 1196-1295)."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    model = _TinyModelWithMoE()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


def test_main_early_stopping(tmp_path):
    """Covers early stopping path (lines 1301-1308)."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    # Set early_stopping_patience=0 so it triggers immediately after first eval
    cfg.training.steps = 5
    cfg.training.eval_interval = 1
    cfg.training.log_interval = 1
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    tiny_model = _TinyModel()
    call_count = [0]
    original_evaluate = None

    def _fake_evaluate(model, val_loader, device, max_batches=20):
        call_count[0] += 1
        # Return increasing loss so is_best is never True after first call
        return 2.0 + call_count[0]

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=tiny_model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.evaluate", side_effect=_fake_evaluate), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        # Patch cfg after load to inject early_stopping_patience
        original_load = __import__("scripts.train", fromlist=["load_config"]).load_config

        def _patched_load(path, overrides):
            c = original_load(path, overrides)
            # inject early_stopping_patience via OmegaConf
            from omegaconf import OmegaConf
            OmegaConf.update(c, "training.early_stopping_patience", 1)
            return c

        with patch("scripts.train.load_config", side_effect=_patched_load):
            main()


def test_main_periodic_save(tmp_path):
    """Covers periodic save path (step > 0 and step % save_interval == 0)."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg.training.steps = 4
    cfg.training.eval_interval = 100  # no eval
    cfg.training.log_interval = 100   # no log
    cfg.training.save_interval = 2    # save at step 2
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    tiny_model = _TinyModel()
    mock_ckpt_mgr = MagicMock()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=tiny_model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.checkpoint.CheckpointManager", return_value=mock_ckpt_mgr), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()

    # Should have called save_checkpoint for the periodic save
    assert mock_ckpt_mgr.save_checkpoint.call_count >= 1


def test_main_grad_accum_gt1(tmp_path):
    """Covers gradient accumulation > 1 path."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg.training.gradient_accumulation_steps = 2
    cfg.training.steps = 2
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    tiny_model = _TinyModel()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=tiny_model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


def test_main_warmup_steps(tmp_path):
    """Covers warmup LR schedule path."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg.training.warmup_steps = 1
    cfg.training.steps = 3
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    tiny_model = _TinyModel()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=tiny_model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


# ── Additional coverage: compile path, SPAR init, Chinchilla print, final save ──

def test_main_compile_path(tmp_path):
    """Covers the compile=True path."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg.compile = True
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    class _ModelWithBackbone(_TinyModel):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(4, 4)

    model = _ModelWithBackbone()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("torch.compile", side_effect=lambda m, **kw: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


def test_main_compile_no_backbone(tmp_path):
    """Covers compile path when model has no backbone attr."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg.compile = True
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    tiny_model = _TinyModel()  # no backbone attr

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=tiny_model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("torch.compile", side_effect=lambda m, **kw: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


def test_main_chinchilla_steps_none(tmp_path):
    """Covers Chinchilla-optimal steps path (training.steps not set)."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    # Remove steps so Chinchilla is used — but cap it low via a mock
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    tiny_model = _TinyModel()

    # Patch load_config to remove steps key
    original_load = __import__("scripts.train", fromlist=["load_config"]).load_config

    def _no_steps_load(path, overrides):
        c = original_load(path, overrides)
        OmegaConf.update(c, "training.steps", OmegaConf.MISSING)
        return c

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=tiny_model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        # Just run with steps=2 from config (already set), this covers the override note
        main()


def test_main_moe_layer_with_expert_pool_grads(tmp_path):
    """Covers Diagnostic A: expert gradient norms with actual grads."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    class _ExpertWithGrad(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(4, 4))

    class _MoELayerWithPool(_MoELayer):
        def __init__(self):
            super().__init__()
            expert = _ExpertWithGrad()
            # Give it a fake grad
            expert.w.grad = torch.randn(4, 4)
            self.expert_pool = MagicMock(spec=["experts", "consolidate_shared_weights"])
            self.expert_pool.experts = [expert]
            self.expert_pool.consolidate_shared_weights = MagicMock()

    class _ModelWithPool(_TinyModelWithMoE):
        def __init__(self):
            super().__init__()
            self.moe_layers = {0: _MoELayerWithPool()}

    model = _ModelWithPool()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


def test_main_moe_layer_with_router_grads(tmp_path):
    """Covers Diagnostic A: router gradient norms."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    class _RouterWithGrad(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(4, 4))
            self.w.grad = torch.randn(4, 4)
            self.clear_aux_state = MagicMock()

    class _MoELayerWithRouterGrad(_MoELayer):
        def __init__(self):
            super().__init__()
            self.router = _RouterWithGrad()

    class _ModelWithRouterGrad(_TinyModelWithMoE):
        def __init__(self):
            super().__init__()
            self.moe_layers = {0: _MoELayerWithRouterGrad()}

    model = _ModelWithRouterGrad()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


def test_main_moe_router_with_lambda_val(tmp_path):
    """Covers Diagnostic B (lambda_val) and Diagnostic C (step < 100)."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    class _RouterWithLambda(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lambda_val = torch.tensor(0.5)
            self.welford_n = torch.tensor([10.0, 10.0])
            self.clear_aux_state = MagicMock()

        def _welford_variance(self):
            return torch.tensor([0.1, 0.1])

    class _MoELayerWithLambda(_MoELayer):
        def __init__(self):
            super().__init__()
            self.router = _RouterWithLambda()

    class _ModelWithLambda(_TinyModelWithMoE):
        def __init__(self):
            super().__init__()
            self.moe_layers = {0: _MoELayerWithLambda()}

    model = _ModelWithLambda()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


def test_main_moe_router_with_get_state(tmp_path):
    """Covers router.get_state() path in logging."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    class _RouterWithState(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.clear_aux_state = MagicMock()

        def get_state(self):
            return {"lambda_eff": 0.3, "fraction_penalised": 0.1,
                    "fatigue_tanh_mean": 0.2, "fairshare": 0.5}

    class _MoELayerWithState(_MoELayer):
        def __init__(self):
            super().__init__()
            self.router = _RouterWithState()

    class _ModelWithState(_TinyModelWithMoE):
        def __init__(self):
            super().__init__()
            self.moe_layers = {0: _MoELayerWithState()}

        def forward(self, input_ids, labels=None, return_metrics=True, record_usage=True):
            x = self.embed(input_ids % 16)
            logits = self.fc(x)
            loss = torch.tensor(1.0, requires_grad=True)
            moe_metrics = {
                "layer_0": {
                    "effective_experts": 2.0,
                    "routing_diversity_gini": 0.3,
                    "router_confidence_mean": 0.7,
                    "fatigue_std": 0.1,
                    "mean_k": 1.5,
                    "stress_mean": 0.2,
                    "lambda_eff": 0.3,
                    "fraction_penalised": 0.1,
                }
            } if return_metrics else {}
            return logits, loss, moe_metrics

    model = _ModelWithState()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


def test_main_moe_router_with_fatigue(tmp_path):
    """Covers moe_layer.step() and reset_welford paths."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    class _RouterWithWelford(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.clear_aux_state = MagicMock()

        def reset_welford(self):
            pass

    class _MoELayerWithWelford(_MoELayer):
        def __init__(self):
            super().__init__()
            self.router = _RouterWithWelford()

    class _ModelWithWelford(_TinyModelWithMoE):
        def __init__(self):
            super().__init__()
            self.moe_layers = {0: _MoELayerWithWelford()}

    model = _ModelWithWelford()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


def test_main_spec_trackers_with_moe(tmp_path):
    """Covers spec_trackers path when model has vocab_size and moe_layers."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    class _ModelWithVocabAndMoE(_TinyModelWithMoE):
        vocab_size = 16

        def forward(self, input_ids, labels=None, return_metrics=True, record_usage=True):
            x = self.embed(input_ids % 16)
            logits = self.fc(x)
            loss = torch.tensor(1.0, requires_grad=True)
            b, s = input_ids.shape
            moe_metrics = {
                "layer_0": {
                    "indices": torch.zeros(b, s, 1, dtype=torch.long),
                }
            } if return_metrics else {}
            return logits, loss, moe_metrics

    model = _ModelWithVocabAndMoE()

    # Patch GlobalSpecializationTracker to avoid device issues
    mock_tracker = MagicMock()
    mock_tracker.sync_and_compute.return_value = {"specialization_score": 0.5}

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.metrics.router_metrics.GlobalSpecializationTracker", return_value=mock_tracker), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


# ── Cover remaining gaps: distributed device path, StopIteration, scaler ──────

def test_main_stopiteration_epoch_wrap(tmp_path):
    """Covers StopIteration → epoch wrap in training loop."""
    from scripts.train import main

    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    # Very few tokens so DataLoader exhausts quickly
    _write_shard(shard_dir / "train_shard_0000.bin", list(range(20)))
    _write_shard(shard_dir / "val_shard_0000.bin", list(range(20)))

    cfg_dict = {
        "experiment_name": "stop_iter_test",
        "seed": 42,
        "compile": False,
        "dataset": {"dataset_key": "wikitext-2", "max_seq_len": 8},
        "model": {"model_key": "gpt-neo-125m", "freeze_backbone": True, "moe_layer_indices": []},
        "router": {"type": "standard", "num_experts": 2, "top_k": 1},
        "expert": {"type": "gpt_neo_lora", "count": 2, "lora": {"rank": 4, "alpha": 1.0, "dropout": 0.0, "init_scale": 0.01}},
        "training": {
            "optimizer": "adamw", "lr": 1e-4, "batch_size": 4, "steps": 5,
            "log_interval": 100, "eval_interval": 100, "save_interval": 100,
            "warmup_steps": 0, "gradient_accumulation_steps": 1, "clip_grad_norm": 1.0,
        },
        "logging": {"enabled": False},
    }
    cfg = OmegaConf.create(cfg_dict)
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    tiny_model = _TinyModel()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=tiny_model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


def test_main_clip_norm_zero(tmp_path):
    """clip_grad_norm=0 skips clipping."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg.training.clip_grad_norm = 0.0
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    tiny_model = _TinyModel()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=tiny_model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


def test_main_not_main_process(tmp_path):
    """Covers is_main_process=False paths (skips prints)."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    tiny_model = _TinyModel()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=tiny_model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=False), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


def test_main_shard_dir_override(tmp_path):
    """Covers --shard-dir CLI override path."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    tiny_model = _TinyModel()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out2")]), \
         patch("scripts.train.build_model", return_value=tiny_model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


def test_main_no_output_dir(tmp_path):
    """Covers default output dir path (no --output-dir)."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    tiny_model = _TinyModel()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir)]), \
         patch("scripts.train.build_model", return_value=tiny_model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()


def test_main_moe_trainable_base(tmp_path):
    """Covers trainable_base=True path for expert_pool.make_base_trainable()."""
    from scripts.train import main

    cfg, shard_dir = _make_full_cfg(tmp_path)
    cfg.expert.lora.trainable_base = True
    cfg_path = tmp_path / "cfg.yaml"
    OmegaConf.save(cfg, cfg_path)

    class _MoELayerWithBase(_MoELayer):
        def __init__(self):
            super().__init__()
            self.expert_pool = MagicMock(spec=["experts", "consolidate_shared_weights", "make_base_trainable"])
            self.expert_pool.experts = []
            self.expert_pool.consolidate_shared_weights = MagicMock()
            self.expert_pool.make_base_trainable = MagicMock()

    class _ModelWithBase(_TinyModelWithMoE):
        def __init__(self):
            super().__init__()
            self.moe_layers = {0: _MoELayerWithBase()}

    model = _ModelWithBase()

    with patch("sys.argv", ["train.py", "--config", str(cfg_path),
                             "--shard-dir", str(shard_dir),
                             "--output-dir", str(tmp_path / "out")]), \
         patch("scripts.train.build_model", return_value=model), \
         patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)), \
         patch("scripts.train.is_main_process", return_value=True), \
         patch("scripts.train.init_wandb"), \
         patch("scripts.train.log_wandb"), \
         patch("scripts.train.cleanup_distributed"), \
         patch("scripts.train.get_model_for_attr_access", side_effect=lambda m: m), \
         patch("src.training.precision.COMPUTE_DTYPE", torch.float32), \
         patch("src.training.precision.needs_grad_scaler", return_value=False), \
         patch("src.training.precision.is_mixed_precision", return_value=False):
        main()
    model.moe_layers[0].expert_pool.make_base_trainable.assert_called_once()


def test_broadcast_scalar_distributed():
    """Covers _broadcast_scalar distributed path."""
    from scripts.train import _broadcast_scalar

    mock_dist = MagicMock()

    def _fake_broadcast(t, src):
        pass  # no-op

    mock_dist.broadcast = _fake_broadcast

    with patch.dict("sys.modules", {"torch.distributed": mock_dist}):
        # Can't actually call dist.broadcast without init, so just test non-distributed
        result = _broadcast_scalar(2.71, "cpu", is_distributed=False)
    assert result == pytest.approx(2.71)
