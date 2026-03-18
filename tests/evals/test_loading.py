import torch

import evals.loading as loading
from evals.loading import build_model_from_config, load_model_for_eval
from src.layers.lora_moe import LoRAMoELayer
from src.training import CheckpointManager


class _FakeBlock(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = torch.nn.Module()
        self.mlp.c_fc = torch.nn.Linear(hidden_dim, 4 * hidden_dim)
        self.mlp.c_proj = torch.nn.Linear(4 * hidden_dim, hidden_dim)


class _FakeBackbone(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.transformer = torch.nn.Module()
        self.transformer.h = torch.nn.ModuleList(
            [_FakeBlock(hidden_dim) for _ in range(num_layers)]
        )


class _FakeModel(torch.nn.Module):
    def __init__(self, variant, freeze_backbone, moe_layer_indices, device):
        super().__init__()
        self.variant = variant
        self.freeze_backbone = freeze_backbone
        self.moe_layer_indices = list(moe_layer_indices)
        self.device = device
        self.backbone = _FakeBackbone(hidden_dim=768, num_layers=12)
        self.moe_layers = {}
        self.to_calls = []

    def inject_moe_layers(self, moe_layers):
        for idx, moe_layer in moe_layers.items():
            self.backbone.transformer.h[idx].mlp = moe_layer
            self.moe_layers[str(idx)] = moe_layer

    def to(self, device=None, dtype=None, **kwargs):
        self.to_calls.append(
            {
                "device": None if device is None else str(device),
                "dtype": None if dtype is None else str(dtype),
            }
        )
        return self


def _test_config():
    return {
        "experiment_name": "smoketest",
        "model": {
            "model_key": "gpt-neo-125m",
            "freeze_backbone": True,
            "moe_layer_indices": [1, 3, 5],
        },
        "router": {
            "type": "metabolic",
            "num_experts": 4,
            "top_k": 2,
            "temperature": 0.7,
            "noise_std": 0.05,
            "metabolic": {
                "lambda_metabolic": 0.5,
                "gamma_recovery": 0.05,
                "beta_cost": 0.4,
                "warmup_steps": 0,
            },
        },
        "expert": {
            "type": "gpt_neo_lora",
            "count": 4,
            "lora": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.0,
                "init_scale": 0.01,
            },
        },
    }


def _patch_model_registry(monkeypatch):
    monkeypatch.setattr(
        loading,
        "model_lookup",
        lambda _: {"model_type": "fake_model", "variant": "tiny", "hidden_dim": 768},
    )
    monkeypatch.setattr(loading.ModelRegistry, "get", lambda _: _FakeModel)


def test_build_model_from_config_injects_requested_moe_layers(monkeypatch):
    _patch_model_registry(monkeypatch)

    model = build_model_from_config(_test_config(), device="cuda:0")

    assert set(model.moe_layers.keys()) == {"1", "3", "5"}
    assert model.to_calls[-1]["device"] == "cuda:0"
    assert model.to_calls[-1]["dtype"] is None
    for idx in (1, 3, 5):
        assert isinstance(model.backbone.transformer.h[idx].mlp, LoRAMoELayer)


def test_load_model_for_eval_returns_checkpoint_info(monkeypatch, tmp_path):
    def fake_load_checkpoint(self, model, optimizer=None, scheduler=None, checkpoint_path=None, load_best=False):
        assert checkpoint_path is not None
        model.loaded_checkpoint_path = checkpoint_path
        return {"step": 42, "metrics": {"loss": 1.23}, "metadata": {"source": "test"}}

    _patch_model_registry(monkeypatch)
    monkeypatch.setattr(CheckpointManager, "load_checkpoint", fake_load_checkpoint)

    checkpoint_path = tmp_path / "checkpoint_step_42.pt"
    checkpoint_path.write_bytes(b"placeholder")

    model, checkpoint_info = load_model_for_eval(
        _test_config(),
        checkpoint_path=checkpoint_path,
        device="cpu",
    )

    assert checkpoint_info["step"] == 42
    assert checkpoint_info["metrics"]["loss"] == 1.23
    assert model.loaded_checkpoint_path == checkpoint_path
    assert model.to_calls[-1]["device"] == "cpu"
    assert model.to_calls[-1]["dtype"] is None
    assert model.training is False


def test_load_model_for_eval_applies_explicit_dtype(monkeypatch, tmp_path):
    def fake_load_checkpoint(
        self, model, optimizer=None, scheduler=None, checkpoint_path=None, load_best=False
    ):
        return {"step": 42, "metrics": {}, "metadata": {}}

    _patch_model_registry(monkeypatch)
    monkeypatch.setattr(CheckpointManager, "load_checkpoint", fake_load_checkpoint)

    checkpoint_path = tmp_path / "checkpoint_step_42.pt"
    checkpoint_path.write_bytes(b"placeholder")

    model, _ = load_model_for_eval(
        _test_config(),
        checkpoint_path=checkpoint_path,
        device="cuda:0",
        dtype=torch.bfloat16,
    )

    assert model.to_calls[-1]["device"] == "cuda:0"
    assert model.to_calls[-1]["dtype"] == "torch.bfloat16"
