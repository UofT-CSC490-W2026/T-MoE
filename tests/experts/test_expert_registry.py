"""Test registry integration."""

import pytest
from src.core import ExpertRegistry
from src.experts.lora_mlp import LoRAConfig


def test_expert_registry_populated():
    """Test that GPTNeoLoRAExpert is registered."""
    assert "gpt_neo_lora" in ExpertRegistry
    assert ExpertRegistry.list() == ["gpt_neo_lora"]


def test_expert_registry_get():
    """Test that registry returns correct expert class."""
    expert_cls = ExpertRegistry.get("gpt_neo_lora")
    config = LoRAConfig(hidden_dim=768, intermediate_dim=3072, rank=16)
    expert = expert_cls(config)

    assert hasattr(expert, "fc1")
    assert hasattr(expert, "fc2")
    assert hasattr(expert, "activation")


def test_expert_registry_invalid_name():
    """Test that invalid expert name raises KeyError."""
    with pytest.raises(KeyError, match="not found in experts registry"):
        ExpertRegistry.get("invalid_expert")
