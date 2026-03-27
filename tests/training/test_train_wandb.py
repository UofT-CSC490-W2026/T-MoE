import sys

from omegaconf import OmegaConf

from scripts.train import init_wandb

def test_init_wandb_defaults_to_online_when_env_is_disabled(monkeypatch):

    class _FakeRun:

        url = "https://wandb.example/run"

    class _FakeWandb:

        def __init__(self):

            self.init_kwargs = None

        def init(self, **kwargs):

            self.init_kwargs = kwargs

            return _FakeRun()

    fake_wandb = _FakeWandb()

    cfg = OmegaConf.create(

        {

            "experiment_name": "demo-train",

            "logging": {

                "enabled": True,

            },

        }

    )

    monkeypatch.setenv("WANDB_MODE", "disabled")

    monkeypatch.setattr("scripts.train.is_main_process", lambda: True)

    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    init_wandb(cfg)

    assert fake_wandb.init_kwargs["project"] == "tmoe"

    assert fake_wandb.init_kwargs["name"] == "demo-train"

    assert fake_wandb.init_kwargs["mode"] == "online"

def test_init_wandb_respects_explicit_disabled_mode(monkeypatch):

    cfg = OmegaConf.create(

        {

            "experiment_name": "demo-train",

            "logging": {

                "enabled": True,

                "mode": "disabled",

            },

        }

    )

    monkeypatch.setattr("scripts.train.is_main_process", lambda: True)

    monkeypatch.delitem(sys.modules, "wandb", raising=False)

    init_wandb(cfg)
