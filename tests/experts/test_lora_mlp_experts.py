"""
Unit tests for LoRA MLP expert implementations.
"""

import pytest
import torch
import torch.nn as nn
from src.experts.lora_layer import LoRALayer
from src.experts.lora_mlp import LoRAConfig
from src.experts.gpt_neo import GPTNeoLoRAExpert


@pytest.fixture
def lora_config():
    return LoRAConfig(
        hidden_dim=768,
        intermediate_dim=3072,
        rank=16,
        alpha=16,
        init_scale=0.01,
    )


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestLoRALayer:
    """Test suite for LoRALayer."""

    def test_initialization(self, lora_config):
        """Test LoRALayer initializes correctly."""
        layer = LoRALayer(
            in_features=lora_config.hidden_dim,
            out_features=lora_config.intermediate_dim,
            rank=lora_config.rank,
            alpha=lora_config.alpha,
        )

        assert layer.in_features == lora_config.hidden_dim
        assert layer.out_features == lora_config.intermediate_dim
        assert layer.rank == lora_config.rank

        # LoRA B should be zero-initialized
        assert torch.all(layer.lora_B.weight == 0)

    def test_load_from_linear(self, lora_config, device):
        """Test loading weights from a pretrained linear layer."""
        layer = LoRALayer(
            in_features=lora_config.hidden_dim,
            out_features=lora_config.intermediate_dim,
            rank=lora_config.rank,
            alpha=lora_config.alpha,
        ).to(device)

        # Create a pretrained linear layer
        pretrained = nn.Linear(lora_config.hidden_dim, lora_config.intermediate_dim).to(
            device
        )
        nn.init.uniform_(pretrained.weight, -0.1, 0.1)

        # Load weights
        layer.load_from_linear(pretrained)

        # Verify base weights are frozen
        assert layer.base_weight is not None
        assert layer.base_weight.requires_grad is False

        # Verify LoRA adapters are trainable
        assert layer.lora_A.weight.requires_grad is True
        assert layer.lora_B.weight.requires_grad is True

    def test_forward_identity_at_init(self, lora_config, device):
        """Test that output matches base layer at initialization (B=0)."""
        layer = LoRALayer(
            in_features=lora_config.hidden_dim,
            out_features=lora_config.intermediate_dim,
            rank=lora_config.rank,
            alpha=lora_config.alpha,
        ).to(device)

        pretrained = nn.Linear(lora_config.hidden_dim, lora_config.intermediate_dim).to(
            device
        )
        nn.init.uniform_(pretrained.weight, -0.1, 0.1)

        layer.load_from_linear(pretrained)

        # Test forward pass
        x = torch.randn(10, lora_config.hidden_dim).to(device)

        with torch.no_grad():
            lora_out = layer(x)
            base_out = pretrained(x)

        # Should be identical since LoRA B is zero
        assert torch.allclose(lora_out, base_out, atol=1e-6)

    def test_dimension_validation(self, lora_config):
        """Test dimension mismatch raises ValueError."""
        layer = LoRALayer(
            in_features=lora_config.hidden_dim,
            out_features=lora_config.intermediate_dim,
            rank=lora_config.rank,
            alpha=lora_config.alpha,
        )

        # Wrong input dimension
        bad_linear = nn.Linear(lora_config.hidden_dim + 1, lora_config.intermediate_dim)
        with pytest.raises(ValueError, match="Input dim mismatch"):
            layer.load_from_linear(bad_linear)

        # Wrong output dimension
        bad_linear = nn.Linear(lora_config.hidden_dim, lora_config.intermediate_dim + 1)
        with pytest.raises(ValueError, match="Output dim mismatch"):
            layer.load_from_linear(bad_linear)


class TestGPTNeoLoRAExpert:
    """Test suite for GPTNeoLoRAExpert."""

    def test_initialization(self, lora_config):
        """Test expert initializes with correct structure."""
        expert = GPTNeoLoRAExpert(lora_config)

        assert hasattr(expert, "fc1")
        assert hasattr(expert, "fc2")
        assert hasattr(expert, "activation")

        # Verify layer dimensions
        assert expert.fc1.in_features == lora_config.hidden_dim
        assert expert.fc1.out_features == lora_config.intermediate_dim
        assert expert.fc2.in_features == lora_config.intermediate_dim
        assert expert.fc2.out_features == lora_config.hidden_dim

    def test_load_from_gpt_neo_mlp(self, lora_config, device):
        """Test loading weights from GPT-Neo MLP structure."""
        expert = GPTNeoLoRAExpert(lora_config).to(device)

        # Mock GPT-Neo MLP structure
        class MockMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.c_fc = nn.Linear(
                    lora_config.hidden_dim, lora_config.intermediate_dim
                )
                self.c_proj = nn.Linear(
                    lora_config.intermediate_dim, lora_config.hidden_dim
                )

        mlp = MockMLP().to(device)
        expert.load_from_mlp(mlp)

        # Verify base weights loaded and frozen
        assert expert.fc1.base_weight is not None
        assert expert.fc2.base_weight is not None
        assert expert.fc1.base_weight.requires_grad is False
        assert expert.fc2.base_weight.requires_grad is False

    def test_forward_matches_original_mlp(self, lora_config, device):
        """Test expert output matches original MLP at initialization."""
        expert = GPTNeoLoRAExpert(lora_config).to(device)

        # Create mock MLP
        class MockMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.c_fc = nn.Linear(
                    lora_config.hidden_dim, lora_config.intermediate_dim
                )
                self.c_proj = nn.Linear(
                    lora_config.intermediate_dim, lora_config.hidden_dim
                )
                try:
                    from transformers.activations import NewGELUActivation

                    self.act = NewGELUActivation()
                except ImportError:
                    self.act = nn.GELU()

            def forward(self, x):
                x = self.c_fc(x)
                x = self.act(x)
                x = self.c_proj(x)
                return x

        mlp = MockMLP().to(device)
        expert.load_from_mlp(mlp)

        # Test forward pass
        x = torch.randn(5, 10, lora_config.hidden_dim).to(device)

        with torch.no_grad():
            expert_out = expert(x)
            mlp_out = mlp(x)

        # Should match since LoRA adapters start at zero
        assert torch.allclose(expert_out, mlp_out, atol=1e-5)

    def test_parameter_count(self, lora_config):
        """Test trainable parameter count calculation."""
        expert = GPTNeoLoRAExpert(lora_config)

        # Expected: 2 layers, each with (in + out) * rank params
        # fc1: (768 + 3072) * 16 = 61,440
        # fc2: (3072 + 768) * 16 = 61,440
        # Total: 122,880
        expected = (
            2
            * (lora_config.hidden_dim + lora_config.intermediate_dim)
            * lora_config.rank
        )

        assert expert.get_param_count() == expected

    def test_only_lora_trainable(self, lora_config, device):
        """Test that only LoRA parameters are trainable after loading."""
        expert = GPTNeoLoRAExpert(lora_config).to(device)

        class MockMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.c_fc = nn.Linear(
                    lora_config.hidden_dim, lora_config.intermediate_dim
                )
                self.c_proj = nn.Linear(
                    lora_config.intermediate_dim, lora_config.hidden_dim
                )

        mlp = MockMLP().to(device)
        expert.load_from_mlp(mlp)

        # Count trainable parameters
        trainable_params = sum(
            p.numel() for p in expert.parameters() if p.requires_grad
        )
        expected_trainable = expert.get_param_count()

        assert trainable_params == expected_trainable

    def test_missing_mlp_attributes_raises_error(self, lora_config):
        """Test that loading from invalid MLP raises ValueError."""
        expert = GPTNeoLoRAExpert(lora_config)

        # MLP without correct structure
        bad_mlp = nn.Linear(lora_config.hidden_dim, lora_config.intermediate_dim)

        with pytest.raises(ValueError, match="c_fc"):
            expert.load_from_mlp(bad_mlp)
