import pytest
import torch
from torch import nn
from src.layers.tmoe import TMoELayer
from src.experts.base import BaseExpert


class MockExpert(BaseExpert):
    """Simple MLP expert for testing."""

    def __init__(self, hidden_dim: int, **kwargs):
        # BaseExpert expects a config object, but TMoELayer passes hidden_dim directly.
        # We pass a dummy config to satisfy BaseExpert's signature.
        super().__init__(config={"hidden_dim": hidden_dim})
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@pytest.fixture
def hidden_dim():
    return 64


@pytest.fixture
def valid_kwargs(hidden_dim):
    return {
        "hidden_dim": hidden_dim,
        "num_experts": 4,
        "top_k": 2,
        "expert_class": MockExpert,
        "router_type": "metabolic",  # Use metabolic since standard is not registered
    }


@pytest.fixture
def tmoe_layer(valid_kwargs):
    return TMoELayer(**valid_kwargs)


def test_initialization_valid(valid_kwargs):
    """Test that layer initializes correctly with valid arguments."""
    layer = TMoELayer(**valid_kwargs)
    assert layer.num_experts == 4
    assert layer.top_k == 2
    assert layer.hidden_dim == valid_kwargs["hidden_dim"]
    assert len(layer.experts) == 4
    assert isinstance(layer.experts[0], MockExpert)


def test_initialization_invalid():
    """Test validation logic in __init__."""
    # Invalid hidden dim
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        TMoELayer(hidden_dim=0, num_experts=4)

    # Invalid num_experts
    with pytest.raises(ValueError, match="num_experts must be positive"):
        TMoELayer(hidden_dim=64, num_experts=0)

    # Invalid top_k
    with pytest.raises(ValueError, match="top_k must be in"):
        TMoELayer(hidden_dim=64, num_experts=4, top_k=5)

    with pytest.raises(ValueError, match="top_k must be in"):
        TMoELayer(hidden_dim=64, num_experts=4, top_k=0)


def test_set_experts_validation(hidden_dim):
    """Test that set_experts raises ValueError (not AssertionError) for mismatched expert count."""
    layer = TMoELayer(hidden_dim=hidden_dim, num_experts=4, expert_class=None)

    # Create wrong number of experts
    wrong_experts = nn.ModuleList([MockExpert(hidden_dim) for _ in range(3)])

    # Should raise ValueError, not AssertionError (which can be disabled with -O)
    with pytest.raises(ValueError, match="Expected 4 experts, got 3"):
        layer.set_experts(wrong_experts)


def test_experts_initialization_check(hidden_dim):
    """Test RuntimeError when experts are not provided."""
    layer = TMoELayer(hidden_dim=hidden_dim, num_experts=4, expert_class=None)
    x = torch.randn(2, 4, hidden_dim)

    with pytest.raises(RuntimeError, match="Experts not initialized"):
        layer(x)


def test_input_validation(tmoe_layer, hidden_dim):
    """Test input shape validation during forward pass."""
    # Wrong dimensions (2D instead of 3D)
    x_2d = torch.randn(4, hidden_dim)
    with pytest.raises(ValueError, match="Expected 3D input"):
        tmoe_layer(x_2d)

    # Wrong hidden dim
    x_wrong_dim = torch.randn(2, 4, hidden_dim + 1)
    with pytest.raises(ValueError, match="Input hidden_dim mismatch"):
        tmoe_layer(x_wrong_dim)


def test_forward_shape_and_type(tmoe_layer, hidden_dim):
    """Test forward pass output shape and return type."""
    batch, seq = 2, 8
    x = torch.randn(batch, seq, hidden_dim)

    # Standard call (return_metrics=False)
    output, metrics = tmoe_layer(x, return_metrics=False)
    assert isinstance(output, torch.Tensor)
    assert metrics is None
    assert output.shape == x.shape

    # With metrics
    output, metrics = tmoe_layer(x, return_metrics=True)
    assert isinstance(output, torch.Tensor)
    assert isinstance(metrics, dict)
    assert output.shape == x.shape


def test_backward_pass(tmoe_layer, hidden_dim):
    """Test that gradients flow through the layer."""
    x = torch.randn(2, 4, hidden_dim, requires_grad=True)
    output, _ = tmoe_layer(x)

    loss = output.mean()
    loss.backward()

    # Check gradients on input
    assert x.grad is not None
    assert torch.norm(x.grad) > 0

    # Check gradients on experts
    for expert in tmoe_layer.experts:
        # At least some experts should have received gradients
        # Note: If routing is deterministic and sparse, some might get zero grad,
        # but the aggregate should have grad
        has_grad = False
        for param in expert.parameters():
            if param.grad is not None:
                has_grad = True
                break
        if has_grad:
            break
    else:
        pytest.fail("No gradients flowed to any expert parameters")


def test_parallel_vs_sequential_numerical_equivalence(valid_kwargs, hidden_dim):
    """Test that parallel and sequential implementations produce numerically close outputs."""
    torch.manual_seed(42)

    # Create two layers with identical initialization
    layer_seq = TMoELayer(**valid_kwargs, use_parallel=False)
    layer_par = TMoELayer(**valid_kwargs, use_parallel=True)

    # Copy weights to ensure identical initialization (strict=False to handle router metadata)
    layer_par.load_state_dict(layer_seq.state_dict(), strict=False)

    # Test input
    x = torch.randn(2, 8, hidden_dim)

    # Forward pass with no noise for deterministic routing
    with torch.no_grad():
        output_seq, _ = layer_seq(x, noise_std=0.0)
        output_par, _ = layer_par(x, noise_std=0.0)

    # Should be numerically close (allowing for floating point differences)
    assert torch.allclose(
        output_seq, output_par, rtol=1e-5, atol=1e-6
    ), f"Max diff: {(output_seq - output_par).abs().max()}"


def test_parallel_vs_sequential_gradient_equivalence(valid_kwargs, hidden_dim):
    """Test that parallel and sequential implementations produce matching gradients."""
    torch.manual_seed(42)

    # Create two layers with identical initialization
    layer_seq = TMoELayer(**valid_kwargs, use_parallel=False)
    layer_par = TMoELayer(**valid_kwargs, use_parallel=True)

    # Copy weights to ensure identical initialization (strict=False to handle router metadata)
    layer_par.load_state_dict(layer_seq.state_dict(), strict=False)

    # Test input (requires_grad for gradient checking)
    x_seq = torch.randn(2, 8, hidden_dim, requires_grad=True)
    x_par = x_seq.clone().detach().requires_grad_(True)

    # Forward pass with no noise for deterministic routing
    output_seq, _ = layer_seq(x_seq, noise_std=0.0)
    output_par, _ = layer_par(x_par, noise_std=0.0)

    # Backward pass with same loss
    loss_seq = output_seq.sum()
    loss_par = output_par.sum()

    loss_seq.backward()
    loss_par.backward()

    # Check input gradients match
    assert torch.allclose(
        x_seq.grad, x_par.grad, rtol=1e-4, atol=1e-5
    ), f"Input grad max diff: {(x_seq.grad - x_par.grad).abs().max()}"

    # Check expert parameter gradients match
    for (name_seq, param_seq), (name_par, param_par) in zip(
        layer_seq.named_parameters(), layer_par.named_parameters()
    ):
        assert name_seq == name_par
        if param_seq.grad is not None and param_par.grad is not None:
            assert torch.allclose(
                param_seq.grad, param_par.grad, rtol=1e-4, atol=1e-5
            ), f"Param {name_seq} grad max diff: {(param_seq.grad - param_par.grad).abs().max()}"


def test_non_contiguous_input_sequential(tmoe_layer, hidden_dim):
    """Test that sequential forward handles non-contiguous tensors (via reshape)."""
    batch, seq = 2, 8

    # Create non-contiguous tensor via transpose operations
    x = torch.randn(seq, batch, hidden_dim).transpose(0, 1)
    assert not x.is_contiguous(), "Test setup failed: tensor should be non-contiguous"

    # Should not raise an error (reshape handles non-contiguous)
    output, _ = tmoe_layer(x)
    assert output.shape == (batch, seq, hidden_dim)


def test_non_contiguous_input_parallel(valid_kwargs, hidden_dim):
    """Test that parallel forward handles non-contiguous tensors (via reshape)."""
    layer = TMoELayer(**valid_kwargs, use_parallel=True)
    batch, seq = 2, 8

    # Create non-contiguous tensor via transpose operations
    x = torch.randn(seq, batch, hidden_dim).transpose(0, 1)
    assert not x.is_contiguous(), "Test setup failed: tensor should be non-contiguous"

    # Should not raise an error (reshape handles non-contiguous)
    output, _ = layer(x)
    assert output.shape == (batch, seq, hidden_dim)


def test_non_contiguous_gradient_flow(valid_kwargs, hidden_dim):
    """Test that gradients flow correctly through non-contiguous inputs."""
    layer = TMoELayer(**valid_kwargs, use_parallel=False)
    batch, seq = 2, 8

    # Create non-contiguous tensor
    x = torch.randn(seq, batch, hidden_dim, requires_grad=True).transpose(0, 1)
    assert not x.is_contiguous()

    # Retain grad on non-leaf tensor to check gradient flow
    x.retain_grad()

    output, _ = layer(x)
    loss = output.sum()
    loss.backward()

    # Gradient should exist and be non-zero
    assert x.grad is not None
    assert torch.norm(x.grad) > 0


def test_extra_repr(tmoe_layer):
    """Test string representation."""
    repr_str = str(tmoe_layer)
    assert "hidden_dim=64" in repr_str
    assert "num_experts=4" in repr_str
    assert "top_k=2" in repr_str


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_consistency(valid_kwargs):
    """Test layer behavior on GPU."""
    device = torch.device("cuda")
    layer = TMoELayer(**valid_kwargs).to(device)
    x = torch.randn(2, 4, valid_kwargs["hidden_dim"], device=device)

    output, _ = layer(x)
    assert output.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_no_gpu_sync_in_parallel_forward(valid_kwargs):
    """Test that parallel forward doesn't trigger unnecessary GPU→CPU syncs."""
    device = torch.device("cuda")
    layer = TMoELayer(**valid_kwargs, use_parallel=True).to(device)
    x = torch.randn(2, 8, valid_kwargs["hidden_dim"], device=device)

    # This is a smoke test - if .tolist() was called, it would cause a sync
    # We can't directly detect syncs, but we can verify it runs without errors
    # and check that intermediate tensors stay on GPU
    with torch.no_grad():
        output, _ = layer(x)

    assert output.device.type == "cuda"
    # If there were GPU syncs, performance would be degraded, but that's hard to test
    # The main goal is ensuring the code path doesn't crash
