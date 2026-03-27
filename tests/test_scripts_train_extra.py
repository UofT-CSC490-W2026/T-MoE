import struct
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_load_config_no_overrides(tmp_path):
    from scripts.train import load_config

    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("experiment_name: test\ntraining:\n  lr: 1e-4\n")
    cfg = load_config(str(cfg_file), [])
    assert cfg.experiment_name == "test"


def test_load_config_with_overrides(tmp_path):
    from scripts.train import load_config

    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("experiment_name: test\ntraining:\n  lr: 1e-4\n")
    cfg = load_config(str(cfg_file), ["training.lr=5e-5"])
    assert abs(cfg.training.lr - 5e-5) < 1e-10


def test_parse_args_basic():
    from scripts.train import parse_args

    with patch("sys.argv", ["train.py", "--config", "experiments/test.yaml"]):
        args, overrides = parse_args()
    assert args.config == "experiments/test.yaml"
    assert args.resume is None
    assert args.output_dir is None


def test_parse_args_with_resume_and_output():
    from scripts.train import parse_args

    with patch(
        "sys.argv",
        [
            "train.py",
            "--config",
            "exp.yaml",
            "--resume",
            "/tmp/ckpt.pt",
            "--output-dir",
            "/tmp/out",
            "--shard-dir",
            "/tmp/shards",
        ],
    ):
        args, overrides = parse_args()
    assert args.resume == "/tmp/ckpt.pt"
    assert args.output_dir == "/tmp/out"
    assert args.shard_dir == "/tmp/shards"


def _write_shard(path: Path, tokens: list, dtype_flag: int = 0):
    arr = np.array(tokens, dtype=np.uint16 if dtype_flag == 0 else np.uint32)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(tokens)))
        f.write(struct.pack("<H", dtype_flag))
        f.write(arr.tobytes())


def _write_legacy_shard(path: Path, tokens: list):
    arr = np.array(tokens, dtype=np.uint16)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(tokens)))
        f.write(arr.tobytes())


def test_shard_dataset_basic(tmp_path):
    from scripts.train import ShardDataset

    tokens = list(range(200))
    _write_shard(tmp_path / "train_shard_0000.bin", tokens)
    ds = ShardDataset(tmp_path, "train", seq_len=10)
    assert len(ds) > 0
    ids, labels = ds[0]
    assert ids.shape[0] == 11


def test_shard_dataset_legacy_shard(tmp_path):
    from scripts.train import ShardDataset

    tokens = list(range(200))
    _write_legacy_shard(tmp_path / "train_shard_0000.bin", tokens)
    ds = ShardDataset(tmp_path, "train", seq_len=10)
    assert len(ds) > 0


def test_shard_dataset_uint32(tmp_path):
    from scripts.train import ShardDataset

    tokens = list(range(200))
    _write_shard(tmp_path / "train_shard_0000.bin", tokens, dtype_flag=1)
    ds = ShardDataset(tmp_path, "train", seq_len=10)
    assert len(ds) > 0


def test_shard_dataset_no_shards(tmp_path):
    from scripts.train import ShardDataset

    with pytest.raises(FileNotFoundError, match="No shards found"):
        ShardDataset(tmp_path, "train", seq_len=10)


def test_shard_dataset_unknown_dtype_flag(tmp_path):
    from scripts.train import ShardDataset

    tokens = list(range(200))
    arr = np.array(tokens, dtype=np.uint16)
    with open(tmp_path / "train_shard_0000.bin", "wb") as f:
        f.write(struct.pack("<Q", len(tokens)))
        f.write(struct.pack("<H", 99))
        f.write(arr.tobytes())
    with pytest.raises(ValueError, match="Unknown dtype_flag"):
        ShardDataset(tmp_path, "train", seq_len=10)


def test_shard_dataset_getitem_wrap(tmp_path):
    from scripts.train import ShardDataset

    tokens = list(range(20))
    _write_shard(tmp_path / "train_shard_0000.bin", tokens)
    ShardDataset(tmp_path, "train", seq_len=15)
    _write_shard(tmp_path / "train_shard_0001.bin", list(range(20, 25)))
    ShardDataset(tmp_path, "train", seq_len=23)
    ds3 = ShardDataset(tmp_path, "train", seq_len=30)
    if len(ds3) > 0:
        ids, _ = ds3[0]
        assert ids.shape[0] == 31


def test_shard_dataset_multiple_shards(tmp_path):
    from scripts.train import ShardDataset

    for i in range(3):
        _write_shard(tmp_path / f"train_shard_{i:04d}.bin", list(range(100)))
    ds = ShardDataset(tmp_path, "train", seq_len=10)
    assert len(ds) > 0
    ids, _ = ds[0]
    assert ids.shape[0] == 11


def _simple_model():
    model = torch.nn.Linear(4, 4)
    return model


def _make_cfg(optimizer="adamw", lr=1e-4, lr_base=None):
    cfg = MagicMock()
    cfg.training.optimizer = optimizer
    cfg.training.lr = lr
    cfg.training.get.side_effect = lambda k, default=None: {
        "lr_base": lr_base,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1,
    }.get(k, default)
    return cfg


def test_build_optimizer_adamw():
    from scripts.train import build_optimizer

    model = _simple_model()
    cfg = _make_cfg("adamw")
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.AdamW)


def test_build_optimizer_adam():
    from scripts.train import build_optimizer

    model = _simple_model()
    cfg = _make_cfg("adam")
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.Adam)


def test_build_optimizer_unknown():
    from scripts.train import build_optimizer

    model = _simple_model()
    cfg = _make_cfg("sgd")
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer(model, cfg)


def test_build_optimizer_with_lr_base():
    from scripts.train import build_optimizer

    class ModelWithMixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.shared_fc_weight = torch.nn.Parameter(torch.randn(4, 4))
            self.frozen = torch.nn.Parameter(torch.randn(4, 4), requires_grad=False)
            self.other = torch.nn.Linear(4, 4)

    model = ModelWithMixed()
    cfg = _make_cfg("adamw", lr=1e-4, lr_base=1e-5)
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.AdamW)


def test_evaluate_basic():
    from scripts.train import evaluate

    model = MagicMock()
    loss_tensor = torch.tensor(1.5)
    model.return_value = (None, loss_tensor, {})
    x = torch.zeros(2, 10, dtype=torch.long)
    y = torch.zeros(2, 10, dtype=torch.long)
    loader = [(x, y)] * 3
    result = evaluate(model, loader, "cpu", max_batches=2)
    assert abs(result - 1.5) < 1e-5


def test_evaluate_empty_loader():
    from scripts.train import evaluate

    model = MagicMock()
    result = evaluate(model, [], "cpu")
    assert result == float("inf")


def test_init_wandb_disabled():
    from scripts.train import init_wandb

    cfg = MagicMock()
    cfg.get.return_value = {"enabled": False}
    with patch("scripts.train.is_main_process", return_value=True):
        init_wandb(cfg)


def test_init_wandb_mode_disabled():
    from scripts.train import init_wandb

    cfg = MagicMock()
    cfg.get.return_value = {"enabled": True, "mode": "disabled"}
    with patch("scripts.train.is_main_process", return_value=True):
        init_wandb(cfg)


def test_init_wandb_not_main_process():
    from scripts.train import init_wandb

    cfg = MagicMock()
    with patch("scripts.train.is_main_process", return_value=False):
        init_wandb(cfg)


def test_init_wandb_import_error():
    from scripts.train import init_wandb

    cfg = MagicMock()
    cfg.get.return_value = {"enabled": True, "mode": "online"}
    with patch("scripts.train.is_main_process", return_value=True):
        with patch.dict("sys.modules", {"wandb": None}):
            init_wandb(cfg)


def test_init_wandb_success():
    from scripts.train import init_wandb
    from omegaconf import OmegaConf

    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_run.url = None
    mock_wandb.init.return_value = mock_run
    cfg = OmegaConf.create(
        {
            "experiment_name": "test_exp",
            "logging": {"enabled": True, "mode": "online", "project": "test"},
        }
    )
    with patch("scripts.train.is_main_process", return_value=True):
        with _patch_wandb(mock_wandb):
            init_wandb(cfg)


def test_log_wandb_not_main():
    from scripts.train import log_wandb

    with patch("scripts.train.is_main_process", return_value=False):
        log_wandb({"loss": 1.0})


def test_log_wandb_no_run():
    from scripts.train import log_wandb

    mock_wandb = MagicMock()
    mock_wandb.run = None
    with patch("scripts.train.is_main_process", return_value=True):
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            log_wandb({"loss": 1.0})


def test_log_wandb_success():
    from scripts.train import log_wandb

    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_wandb.run = mock_run
    with patch("scripts.train.is_main_process", return_value=True):
        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            log_wandb({"loss": 1.0})


def test_broadcast_scalar_non_distributed():
    from scripts.train import _broadcast_scalar

    result = _broadcast_scalar(3.14, "cpu", is_distributed=False)
    assert abs(result - 3.14) < 1e-6


def test_build_model_mocked(tmp_path):
    from scripts.train import build_model

    cfg = MagicMock()
    cfg.model.model_key = "gpt-neo-125m"
    cfg.model.freeze_backbone = True
    cfg.model.moe_layer_indices = [0]
    cfg.router.type = "standard"
    cfg.router.num_experts = 2
    cfg.router.top_k = 1
    cfg.router.get.return_value = 0.1
    cfg.expert.lora.rank = 4
    cfg.expert.lora.alpha = 1.0
    cfg.expert.lora.dropout = 0.0
    cfg.expert.lora.init_scale = 0.01
    cfg.expert.lora.get.return_value = 0.0
    cfg.expert.count = 2
    cfg.expert.type = "gpt_neo_lora"
    mock_model = MagicMock()
    mock_model.num_layers = 12
    mock_model.hidden_dim = 64
    mock_model.get_mlp_at.return_value = MagicMock()
    mock_registry = MagicMock()
    mock_registry.get.return_value = MagicMock(return_value=mock_model)
    mock_router = MagicMock()
    mock_lora_layer = MagicMock()
    with patch("src.core.ModelRegistry", mock_registry):
        with patch(
            "src.configs.model.model_lookup",
            return_value={"hf_name": "x", "model_type": "gpt_neo", "variant": "125m"},
        ):
            with patch("src.routers.create_router", return_value=mock_router):
                with patch(
                    "src.layers.lora_moe.LoRAMoELayer.from_pretrained_mlp",
                    return_value=mock_lora_layer,
                ):
                    with patch("src.experts.lora.LoRAConfig"):
                        with patch("src.project_types.ExpertType"):

                            model = build_model(cfg)
    assert model is not None


def test_main_missing_config():
    from scripts.train import main

    with patch("sys.argv", ["train.py"]):
        with pytest.raises(SystemExit):
            main()


def test_main_keyboard_interrupt(tmp_path):
    from scripts.train import main

    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(
        "experiment_name: test\n"
        "seed: 42\n"
        "compile: false\n"
        "dataset:\n  dataset_key: wikitext-103\n  max_seq_len: 16\n"
        "model:\n  model_key: gpt-neo-125m\n  freeze_backbone: true\n  moe_layer_indices: [0]\n"
        "router:\n  type: standard\n  num_experts: 2\n  top_k: 1\n"
        "expert:\n  count: 2\n  type: gpt_neo_lora\n  lora:\n    rank: 4\n    alpha: 1.0\n    dropout: 0.0\n    init_scale: 0.01\n"
        "training:\n  lr: 1e-4\n  batch_size: 2\n  steps: 1\n  optimizer: adamw\n"
        "logging:\n  enabled: false\n"
    )
    with patch("sys.argv", ["train.py", "--config", str(cfg_file)]):
        with patch("scripts.train.init_distributed", return_value=(False, 0, 0, 1)):
            with patch("scripts.train.build_model", side_effect=KeyboardInterrupt):
                with patch("scripts.train.cleanup_distributed"):
                    with pytest.raises((KeyboardInterrupt, SystemExit, Exception)):
                        main()


def _patch_wandb(mock_wandb):
    import sys
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        orig = sys.modules.get("wandb")
        sys.modules["wandb"] = mock_wandb
        try:
            yield
        finally:
            if orig is None:
                sys.modules.pop("wandb", None)
            else:
                sys.modules["wandb"] = orig

    return _ctx()


def test_init_wandb_with_entity():
    from scripts.train import init_wandb
    from omegaconf import OmegaConf

    mock_wandb = MagicMock()
    mock_run = MagicMock()
    mock_run.url = "https://wandb.ai/test"
    mock_wandb.init.return_value = mock_run
    cfg = OmegaConf.create(
        {
            "experiment_name": "test_exp",
            "logging": {
                "enabled": True,
                "mode": "online",
                "project": "test",
                "entity": "myteam",
            },
        }
    )
    with patch("scripts.train.is_main_process", return_value=True):
        with _patch_wandb(mock_wandb):
            init_wandb(cfg)
    mock_wandb.init.assert_called_once()
    assert "entity" in mock_wandb.init.call_args.kwargs


def test_init_wandb_exception():
    from scripts.train import init_wandb
    from omegaconf import OmegaConf

    mock_wandb = MagicMock()
    mock_wandb.init.side_effect = Exception("wandb error")
    cfg = OmegaConf.create(
        {
            "experiment_name": "test_exp",
            "logging": {"enabled": True, "mode": "online", "project": "test"},
        }
    )
    with patch("scripts.train.is_main_process", return_value=True):
        with _patch_wandb(mock_wandb):
            init_wandb(cfg)


def test_initialize_router_prototypes_no_spar_routers():
    from scripts.train import _initialize_router_prototypes

    model = torch.nn.Linear(4, 4)
    loader = [
        (torch.zeros(2, 4, dtype=torch.long), torch.zeros(2, 4, dtype=torch.long))
    ]
    with patch("scripts.train.get_model_for_attr_access", return_value=model):
        _initialize_router_prototypes(model, loader, "cpu", is_distributed=False)


def test_shard_dataset_cross_shard_boundary(tmp_path):
    from scripts.train import ShardDataset

    _write_shard(tmp_path / "train_shard_0000.bin", list(range(15)))
    _write_shard(tmp_path / "train_shard_0001.bin", list(range(15, 30)))
    ds = ShardDataset(tmp_path, "train", seq_len=10)
    assert len(ds) > 0
    for i in range(len(ds)):
        ids, _ = ds[i]
        assert ids.shape[0] == 11


def test_log_wandb_import_error():
    from scripts.train import log_wandb
    import sys

    with patch("scripts.train.is_main_process", return_value=True):
        orig = sys.modules.get("wandb")
        sys.modules["wandb"] = None
        try:
            log_wandb({"loss": 1.0})
        finally:
            if orig is None:
                sys.modules.pop("wandb", None)
            else:
                sys.modules["wandb"] = orig
        assert sys.modules.get("wandb") is orig


def test_build_optimizer_lr_base_no_matching_params():
    from scripts.train import build_optimizer

    model = torch.nn.Linear(4, 4)
    cfg = MagicMock()
    cfg.training.optimizer = "adamw"
    cfg.training.lr = 1e-4
    cfg.training.get.side_effect = lambda k, default=None: {
        "lr_base": 1e-5,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1,
    }.get(k, default)
    opt = build_optimizer(model, cfg)
    assert isinstance(opt, torch.optim.AdamW)


def test_shard_dataset_val_split(tmp_path):
    from scripts.train import ShardDataset

    _write_shard(tmp_path / "val_shard_0000.bin", list(range(100)))
    ds = ShardDataset(tmp_path, "val", seq_len=10)
    assert len(ds) > 0
