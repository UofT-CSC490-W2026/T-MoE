import torch
import torch.nn as nn

from src.training.checkpoint import _align_orig_mod


class _Nested(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)


class _MockCompiledBackbone(nn.Module):
    """Mimics model.module when backbone has been torch.compile()'d.

    torch.compile wraps the module as OptimizedModule, replacing it with
    a submodule named '_orig_mod'.  We replicate the key structure manually
    since we can't call torch.compile in a unit test (no GPU required).
    """

    def __init__(self):
        super().__init__()
        self._orig_mod = _Nested()

    def state_dict(self, **kwargs):
        # Mirrors what model.module.state_dict() returns when backbone is compiled.
        raw = super().state_dict(**kwargs)
        # raw keys are like "_orig_mod.linear.weight" — prefix with "backbone."
        return {"backbone." + k: v for k, v in raw.items()}


class _MockPlainBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _Nested()


def test_align_adds_orig_mod_when_model_has_it():
    # Checkpoint was saved without _orig_mod; model now has it (compiled run).
    target = _MockCompiledBackbone()

    # Keys as if saved from a non-compiled run (no _orig_mod)
    ckpt_sd = {
        "backbone.linear.weight": torch.zeros(4, 4),
        "backbone.linear.bias": torch.zeros(4),
    }

    aligned = _align_orig_mod(ckpt_sd, target)

    assert "backbone._orig_mod.linear.weight" in aligned
    assert "backbone._orig_mod.linear.bias" in aligned
    assert "backbone.linear.weight" not in aligned
    assert "backbone.linear.bias" not in aligned


def test_align_strips_orig_mod_when_model_lacks_it():
    # Checkpoint was saved with _orig_mod; model is not compiled.
    target = _MockPlainBackbone()

    ckpt_sd = {
        "backbone._orig_mod.linear.weight": torch.zeros(4, 4),
        "backbone._orig_mod.linear.bias": torch.zeros(4),
    }

    aligned = _align_orig_mod(ckpt_sd, target)

    assert "backbone.linear.weight" in aligned
    assert "backbone.linear.bias" in aligned
    assert "backbone._orig_mod.linear.weight" not in aligned


def test_align_noop_when_both_have_orig_mod():
    target = _MockCompiledBackbone()

    ckpt_sd = {
        "backbone._orig_mod.linear.weight": torch.zeros(4, 4),
    }

    aligned = _align_orig_mod(ckpt_sd, target)
    assert aligned is ckpt_sd  # same object — no copy


def test_align_noop_when_neither_has_orig_mod():
    target = _MockPlainBackbone()

    ckpt_sd = {
        "backbone.linear.weight": torch.zeros(4, 4),
    }

    aligned = _align_orig_mod(ckpt_sd, target)
    assert aligned is ckpt_sd


def test_align_router_buffers_get_remapped():
    # Specifically exercise router buffer key patterns that were lost in the bug.
    target = _MockCompiledBackbone()

    ckpt_sd = {
        "backbone.model.layers.2.mlp.router.W": torch.zeros(8, 4),
        "backbone.model.layers.2.mlp.router.ema_load": torch.ones(8) * 0.125,
        "backbone.model.layers.2.mlp.router.lambda_val": torch.tensor(0.46),
        "backbone.model.layers.2.mlp.router.lambda_initialized": torch.tensor(True),
        "backbone.model.layers.2.mlp.router.num_steps": torch.tensor(13000),
    }

    aligned = _align_orig_mod(ckpt_sd, target)

    for k in ckpt_sd:
        new_k = k.replace("backbone.", "backbone._orig_mod.", 1)
        assert new_k in aligned, f"Expected {new_k} in aligned keys"
        assert k not in aligned
