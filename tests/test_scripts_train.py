"""Tests for scripts/train.py — lightweight mocked tests."""

import struct
import numpy as np
import torch
import pytest
from unittest.mock import patch, MagicMock


def test_shard_dataset(tmp_path):
    """Test ShardDataset reads binary shards correctly."""
    from scripts.train import ShardDataset

    # Create a minimal shard: 8-byte header + uint16 tokens
    tokens = np.arange(100, dtype=np.uint16)
    shard_path = tmp_path / "train_shard_0000.bin"
    with open(shard_path, "wb") as f:
        f.write(struct.pack("<Q", len(tokens)))
        f.write(tokens.tobytes())
    ds = ShardDataset(tmp_path, "train", seq_len=10)
    assert len(ds) > 0
    ids, labels = ds[0]
    assert ids.shape == (11,)
    assert labels.shape == (11,)


def test_shard_dataset_versioned(tmp_path):
    """Test ShardDataset with versioned (10-byte header) uint32 shards."""
    from scripts.train import ShardDataset

    tokens = np.arange(100, dtype=np.uint32)
    shard_path = tmp_path / "train_shard_0000.bin"
    with open(shard_path, "wb") as f:
        f.write(struct.pack("<QH", len(tokens), 1))  # dtype_flag=1 → uint32
        f.write(tokens.tobytes())
    ds = ShardDataset(tmp_path, "train", seq_len=10)
    assert len(ds) > 0
    ids, labels = ds[0]
    assert ids.shape == (11,)


def test_shard_dataset_no_shards(tmp_path):
    """ShardDataset raises FileNotFoundError when no shards exist."""
    from scripts.train import ShardDataset

    with pytest.raises(FileNotFoundError):
        ShardDataset(tmp_path, "train", seq_len=10)


def test_shard_dataset_unknown_dtype(tmp_path):
    """ShardDataset raises ValueError for unknown dtype_flag."""
    from scripts.train import ShardDataset

    tokens = np.arange(100, dtype=np.uint32)
    shard_path = tmp_path / "train_shard_0000.bin"
    with open(shard_path, "wb") as f:
        f.write(struct.pack("<QH", len(tokens), 99))  # unknown dtype_flag
        f.write(tokens.tobytes())
    with pytest.raises(ValueError, match="Unknown dtype_flag"):
        ShardDataset(tmp_path, "train", seq_len=10)


def test_shard_dataset_wraparound(tmp_path):
    """Test ShardDataset wraps around when reaching end of tokens."""
    from scripts.train import ShardDataset

    tokens = np.arange(20, dtype=np.uint16)
    shard_path = tmp_path / "train_shard_0000.bin"
    with open(shard_path, "wb") as f:
        f.write(struct.pack("<Q", len(tokens)))
        f.write(tokens.tobytes())
    ds = ShardDataset(tmp_path, "train", seq_len=10)
    # Access last item which may wrap around
    ids, labels = ds[len(ds) - 1]
    assert ids.shape == (11,)


def test_load_config(tmp_path):
    from scripts.train import load_config

    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text("experiment_name: test\ntraining:\n  lr: 0.001\n")
    cfg = load_config(str(cfg_path), [])
    assert cfg.experiment_name == "test"


def test_load_config_with_overrides(tmp_path):
    from scripts.train import load_config

    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text("experiment_name: test\ntraining:\n  lr: 0.001\n")
    cfg = load_config(str(cfg_path), ["training.lr=0.01"])
    assert cfg.training.lr == 0.01


def test_parse_args():
    from scripts.train import parse_args

    with patch("sys.argv", ["train.py", "--config", "test.yaml"]):
        args, overrides = parse_args()
        assert args.config == "test.yaml"
        assert args.resume is None


def test_parse_args_with_resume():
    from scripts.train import parse_args

    with patch(
        "sys.argv", ["train.py", "--config", "test.yaml", "--resume", "ckpt.pt"]
    ):
        args, overrides = parse_args()
        assert args.resume == "ckpt.pt"


def test_evaluate():
    from scripts.train import evaluate

    model = MagicMock()
    model.eval = MagicMock()
    model.train = MagicMock()
    model.return_value = (None, torch.tensor(1.5), None)
    loader = [
        (torch.zeros(2, 10, dtype=torch.long), torch.zeros(2, 10, dtype=torch.long))
    ]
    loss = evaluate(model, loader, "cpu", max_batches=1)
    assert isinstance(loss, float)


def test_evaluate_empty_loader():
    from scripts.train import evaluate

    model = MagicMock()
    model.eval = MagicMock()
    model.train = MagicMock()
    loss = evaluate(model, [], "cpu", max_batches=1)
    assert loss == float("inf")


def test_init_wandb_not_main():
    from scripts.train import init_wandb

    cfg = MagicMock()
    with patch("scripts.train.is_main_process", return_value=False):
        init_wandb(cfg)


def test_init_wandb_disabled():
    from scripts.train import init_wandb

    cfg = MagicMock()
    cfg.get.return_value = {"enabled": False}
    with patch("scripts.train.is_main_process", return_value=True):
        init_wandb(cfg)


def test_init_wandb_mode_disabled():
    from scripts.train import init_wandb

    cfg = MagicMock()
    logging_cfg = {"enabled": True, "mode": "disabled"}
    cfg.get.return_value = logging_cfg
    with patch("scripts.train.is_main_process", return_value=True):
        init_wandb(cfg)


def test_init_wandb_import_error():
    from scripts.train import init_wandb

    cfg = MagicMock()
    logging_cfg = MagicMock()
    logging_cfg.get.side_effect = lambda k, d=None: {
        "enabled": True,
        "mode": "online",
        "project": "test",
    }.get(k, d)
    cfg.get.return_value = logging_cfg
    with patch("scripts.train.is_main_process", return_value=True):
        with patch.dict("sys.modules", {"wandb": None}):
            init_wandb(cfg)


def test_log_wandb_not_main():
    from scripts.train import log_wandb

    with patch("scripts.train.is_main_process", return_value=False):
        log_wandb({"loss": 1.0})


def test_log_wandb_no_wandb():
    from scripts.train import log_wandb

    with patch("scripts.train.is_main_process", return_value=True):
        with patch.dict("sys.modules", {"wandb": None}):
            log_wandb({"loss": 1.0})


def test_broadcast_scalar_not_distributed():
    from scripts.train import _broadcast_scalar

    result = _broadcast_scalar(3.14, "cpu", False)
    assert result == 3.14


def test_build_optimizer_adamw():
    from scripts.train import build_optimizer

    model = torch.nn.Linear(10, 10)
    cfg = MagicMock()
    cfg.training.optimizer = "adamw"
    cfg.training.lr = 1e-3
    cfg.training.get.side_effect = lambda k, d=None: {
        "lr_base": None,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1,
    }.get(k, d)
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.AdamW)


def test_build_optimizer_adam():
    from scripts.train import build_optimizer

    model = torch.nn.Linear(10, 10)
    cfg = MagicMock()
    cfg.training.optimizer = "adam"
    cfg.training.lr = 1e-3
    cfg.training.get.side_effect = lambda k, d=None: {
        "lr_base": None,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1,
    }.get(k, d)
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.Adam)


def test_build_optimizer_unknown():
    from scripts.train import build_optimizer

    model = torch.nn.Linear(10, 10)
    cfg = MagicMock()
    cfg.training.optimizer = "sgd_unknown"
    cfg.training.lr = 1e-3
    cfg.training.get.side_effect = lambda k, d=None: {
        "lr_base": None,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1,
    }.get(k, d)
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer(model, cfg)


def test_build_optimizer_with_base_lr():
    from scripts.train import build_optimizer

    model = torch.nn.Module()
    model.shared_fc_weight = torch.nn.Parameter(torch.randn(10, 10))
    model.other_param = torch.nn.Parameter(torch.randn(5, 5))
    model.register_parameter("shared_fc_weight", model.shared_fc_weight)
    model.register_parameter("other_param", model.other_param)
    cfg = MagicMock()
    cfg.training.optimizer = "adamw"
    cfg.training.lr = 1e-3
    cfg.training.get.side_effect = lambda k, d=None: {
        "lr_base": 1e-5,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1,
    }.get(k, d)
    opt = build_optimizer(model, cfg)
    assert len(opt.param_groups) == 2
