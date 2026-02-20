import pytest
from src.core import ExpertRegistry
from src.experts.lora import LoRAConfig


def test_expert_registry_populated():
    """Test that GPTNeoLoRAExpert is registered."""
    assert "gpt_neo_lora" in ExpertRegistry
    assert "gpt_neo_lora" in ExpertRegistry.list()


def test_expert_registry_get():
    """Test that registry returns correct expert class."""
    expert_cls = ExpertRegistry.get("gpt_neo_lora")
    config = LoRAConfig(hidden_dim=768, rank=16)
    expert = expert_cls(config)

    assert hasattr(expert, "c_fc")
    assert hasattr(expert, "c_proj")
    assert hasattr(expert, "act")


def test_expert_registry_invalid_name():
    """Test that invalid expert name raises KeyError."""
    with pytest.raises(KeyError, match="not found in experts registry"):
        ExpertRegistry.get("invalid_expert")
