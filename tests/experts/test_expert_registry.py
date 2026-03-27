import pytest

import torch.nn as nn

from src.core import ExpertRegistry, RouterRegistry, ModelRegistry

from src.experts.lora import LoRAConfig

def test_expert_registry_populated():

    assert "gpt_neo_lora" in ExpertRegistry

    assert "gpt_neo_lora" in ExpertRegistry.list()

def test_expert_registry_get():

    expert_cls = ExpertRegistry.get("gpt_neo_lora")

    config = LoRAConfig(hidden_dim=768, rank=16)

    expert = expert_cls(config)

    assert hasattr(expert, "c_fc")

    assert hasattr(expert, "c_proj")

def test_expert_registry_invalid_name():

    with pytest.raises(KeyError, match="not found in experts registry"):

        ExpertRegistry.get("invalid_expert")

def test_router_registry_contains_all_types():

    for name in (

        "metabolic",

        "standard",

        "topk",

        "switch",

        "deepseek",

        "expert_choice",

    ):

        assert name in RouterRegistry, f"RouterRegistry missing: {name}"

def test_router_registry_get_returns_class():

    cls = RouterRegistry.get("metabolic")

    assert issubclass(cls, nn.Module)

def test_model_registry_contains_gptneo():

    assert "gpt_neo" in ModelRegistry

def test_registry_contains_dunder():

    assert ("metabolic" in RouterRegistry) is True

    assert ("nonexistent" in RouterRegistry) is False

def test_registry_overwrite_warns():

    from src.core.registry import Registry

    reg = Registry("test_overwrite")

    reg.register("foo")(int)

    with pytest.warns(UserWarning, match="Overwriting"):

        reg.register("foo")(str)
