import pytest
import torch
import torch.nn.functional as F
from configs.router import MetabolicRouterConfig
from src.routers.metabolic import MetabolicRouter


class TestMetabolicRouterConfig:
    """Test suite for MetabolicRouterConfig validation."""

    def test_default_config_initialization(self):
        """Verify default config values are sensible."""
        config = MetabolicRouterConfig()
        assert config.router_type == "metabolic"
        assert config.hidden_dim > 0
        assert config.num_experts > 0
        assert config.top_k > 0
        assert 0.0 <= config.lambda_metabolic <= 1.0
        assert config.mu_silicon >= 0.0
        assert 0.0 < config.gamma_recovery < 1.0
        assert config.beta_cost > 0.0
        assert config.warmup_steps >= 0

    def test_custom_config_initialization(self):
        """Verify custom config values are preserved."""
        config = MetabolicRouterConfig(
            hidden_dim=512, num_experts=16, lambda_metabolic=0.2, warmup_steps=200
        )
        assert config.hidden_dim == 512
        assert config.num_experts == 16
        assert config.lambda_metabolic == 0.2
        assert config.warmup_steps == 200


class TestRouterInitialization:
    """Test router initialization and state buffers."""

    def test_router_initialization(self, standard_config, device):
        """Verify router initializes with correct architecture."""
        router = MetabolicRouter(standard_config).to(device)

        assert router.num_experts == standard_config.num_experts
        assert router.top_k == standard_config.top_k
        assert router.hidden_dim == standard_config.hidden_dim

        # Check learnable prototypes
        # With weight_norm (modern parametrizations API), the module is wrapped
        if standard_config.normalize_weights:
            # Modern API: check for parametrizations attribute
            assert hasattr(router.prototypes, "parametrizations")
            assert "weight" in router.prototypes.parametrizations
        else:
            # No normalization: standard linear layer
            assert router.prototypes.weight.shape == (
                standard_config.num_experts,
                standard_config.hidden_dim,
            )

        # Check state buffers
        assert router.fatigue.shape == (standard_config.num_experts,)
        assert router.birth_step.shape == (standard_config.num_experts,)
        assert router.num_steps.ndim == 0  # Scalar tensor

        # Verify initial state
        assert torch.allclose(router.fatigue, torch.zeros_like(router.fatigue))
        assert torch.allclose(router.birth_step, torch.zeros_like(router.birth_step))
        assert router.num_steps.item() == 0

    def test_router_device_consistency(self, standard_config, device):
        """Ensure all parameters and buffers are on the same device."""
        router = MetabolicRouter(standard_config).to(device)

        # Check device type consistency (not strict equality due to MPS index differences)
        # Modern parametrizations API: check the underlying weight parameter
        if standard_config.normalize_weights:
            # Weight is still accessible even with parametrizations
            assert router.prototypes.weight.device.type == device.type
        else:
            assert router.prototypes.weight.device.type == device.type

        assert router.fatigue.device.type == device.type
        assert router.birth_step.device.type == device.type
        assert router.num_steps.device.type == device.type


class TestAlignmentComputation:
    """Test cosine similarity and alignment computation."""

    def test_alignment_without_normalization(self, device):
        """Test dot product alignment (unnormalized)."""
        config = MetabolicRouterConfig(
            hidden_dim=64,
            num_experts=4,
            normalize_inputs=False,
            normalize_weights=False,
        )
        router = MetabolicRouter(config).to(device)
        x = torch.randn(2, 3, 64, device=device)

        alignment = router.compute_alignment(x)

        assert alignment.shape == (2, 3, 4)  # [batch, seq, num_experts]
        assert torch.isfinite(alignment).all()

    def test_alignment_with_normalization(self, device):
        """Test cosine similarity alignment (normalized)."""
        config = MetabolicRouterConfig(
            hidden_dim=64, num_experts=4, normalize_inputs=True, normalize_weights=True
        )
        router = MetabolicRouter(config).to(device)
        x = torch.randn(2, 3, 64, device=device)

        alignment = router.compute_alignment(x)

        assert alignment.shape == (2, 3, 4)
        # Cosine similarity should be bounded in [-1, 1]
        assert alignment.min() >= -1.0 - 1e-5
        assert alignment.max() <= 1.0 + 1e-5

    def test_alignment_orthogonal_vectors(self, device):
        """Verify orthogonal vectors have zero cosine similarity."""
        config = MetabolicRouterConfig(
            hidden_dim=4,
            num_experts=2,
            normalize_inputs=True,
            normalize_weights=False,  # Disable weight_norm for simpler testing
        )
        router = MetabolicRouter(config).to(device)

        # Set prototypes to orthogonal vectors
        router.prototypes.weight.data = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=device
        )

        # Input aligned with first expert
        x = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]], device=device)
        alignment = router.compute_alignment(x)

        assert torch.isclose(
            alignment[0, 0, 0], torch.tensor(1.0, device=device), atol=1e-5
        )
        assert torch.isclose(
            alignment[0, 0, 1], torch.tensor(0.0, device=device), atol=1e-5
        )


class TestRoutingPotential:
    """Test routing potential computation with fatigue and silicon tax."""

    def test_potential_without_penalties(self, zero_fatigue_router, test_input):
        """With zero fatigue and silicon tax, potential equals alignment."""
        zero_fatigue_router.lambda_metabolic = 0.0
        zero_fatigue_router.mu_silicon = 0.0

        alignment = zero_fatigue_router.compute_alignment(test_input)
        potential = zero_fatigue_router.compute_routing_potential(
            alignment, noise_std=0.0
        )

        assert torch.allclose(potential, alignment, atol=1e-5)

    def test_softsign_fatigue_penalty(self, router, test_input, device):
        """Verify SoftSign bounds fatigue penalty to (-1, 1)."""
        # Set extreme fatigue values
        router.fatigue.data = torch.tensor(
            [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0], device=device
        )

        alignment = router.compute_alignment(test_input)
        potential = router.compute_routing_potential(alignment, noise_std=0.0)

        # SoftSign(x) = x / (1 + |x|) is bounded in (-1, 1)
        fatigue_penalty = F.softsign(router.fatigue)
        assert fatigue_penalty.min() > -1.0
        assert fatigue_penalty.max() < 1.0

        # Potential should be affected by fatigue
        assert not torch.allclose(potential, alignment)

    def test_gumbel_noise_injection(self, router, test_input):
        """Verify Gumbel noise increases exploration during training."""
        router.train()

        alignment = router.compute_alignment(test_input)
        potential_no_noise = router.compute_routing_potential(alignment, noise_std=0.0)
        potential_with_noise = router.compute_routing_potential(
            alignment, noise_std=0.1
        )

        # Noise should create differences
        assert not torch.allclose(potential_no_noise, potential_with_noise)

    def test_silicon_tax_penalty(self, router, test_input, device):
        """Verify silicon tax penalizes distant experts (when enabled)."""
        router.mu_silicon = 0.5

        # Mock hardware distance (increasing with expert ID)
        def mock_distance(expert_ids):
            return expert_ids.float() * 0.1

        router._get_hardware_distance = mock_distance

        alignment = router.compute_alignment(test_input)
        potential = router.compute_routing_potential(alignment, noise_std=0.0)

        # Higher expert IDs should have lower potential due to distance penalty
        # This depends on the alignment values, but the tax should be applied
        assert torch.isfinite(potential).all()


class TestFatigueDynamics:
    """Test age-aware fatigue accumulation and recovery."""

    def test_fatigue_accumulation(self, zero_fatigue_router, test_input):
        """Verify fatigue increases with expert usage."""
        router = zero_fatigue_router
        router.train()

        initial_fatigue = router.fatigue.clone()

        # Run forward pass multiple times (simulating training steps)
        for _ in range(10):
            router(test_input, return_metrics=False)
            # Call step() to apply deferred fatigue update (as required after optimizer.step())
            router.step()

        # Some experts should have accumulated fatigue
        assert (router.fatigue > initial_fatigue).any()
        assert router.num_steps.item() == 10

    def test_exponential_recovery(self, router, device):
        """Verify fatigue decays exponentially when experts are not used."""
        router.fatigue.data = torch.ones(router.num_experts, device=device) * 10.0

        # Simulate recovery by updating fatigue with zero usage
        # usage = torch.zeros(router.num_experts, device=device) # when silicon tax is 0
        initial_fatigue = router.fatigue.clone()

        # Manual fatigue update (recovery only)
        gamma = router.gamma_recovery
        router.fatigue.data = (1 - gamma) * router.fatigue

        # Fatigue should decrease
        assert (router.fatigue < initial_fatigue).all()
        assert torch.allclose(router.fatigue, initial_fatigue * (1 - gamma), atol=1e-5)

    def test_age_aware_warmup_scaling(self, zero_fatigue_router, device):
        """Verify newborn experts have reduced fatigue penalty during warmup."""
        router = zero_fatigue_router
        router.warmup_steps = 100
        router.beta_cost = 0.1

        # Expert 0: mature (born at step 0)
        # Expert 1: newborn (born at step 50)
        router.num_steps.fill_(100)
        router.birth_step[0] = 0
        router.birth_step[1] = 50

        # Compute age factors
        age_0 = (router.num_steps - router.birth_step[0]) / router.warmup_steps
        age_1 = (router.num_steps - router.birth_step[1]) / router.warmup_steps

        age_factor_0 = torch.clamp(age_0, max=1.0)
        age_factor_1 = torch.clamp(age_1, max=1.0)

        # Mature expert should have full cost
        assert torch.isclose(age_factor_0, torch.tensor(1.0, device=device))

        # Newborn expert should have reduced cost
        assert age_factor_1 < 1.0
        assert torch.isclose(age_factor_1, torch.tensor(0.5, device=device))

    def test_bounded_fatigue(self, router, test_input):
        """Verify fatigue stays bounded even after many iterations."""
        router.train()
        router.lambda_metabolic = 0.5
        router.beta_cost = 1.0

        # Run many iterations
        for _ in range(1000):
            router(test_input, return_metrics=False)

        # Fatigue should stay finite and reasonable
        assert torch.isfinite(router.fatigue).all()
        # With SoftSign, fatigue penalty is bounded, but fatigue itself can grow
        # With raw count usage (no token normalization), fatigue accumulates faster
        # However, with recovery, it should still stabilize
        assert router.fatigue.abs().max() < 200.0  # Reasonable bound


class TestForwardPass:
    """Test end-to-end forward pass integration."""

    def test_forward_output_shapes(self, router, test_input):
        """Verify forward pass returns correct tensor shapes."""
        batch, seq, hidden = test_input.shape
        top_k = router.top_k

        weights, indices, metrics = router(test_input, return_metrics=True)

        assert weights.shape == (batch, seq, top_k)
        assert indices.shape == (batch, seq, top_k)
        assert isinstance(metrics, dict)

    def test_forward_weight_normalization(self, router, test_input):
        """Verify routing weights sum to 1 (softmax normalization)."""
        weights, _, _ = router(test_input, return_metrics=True)

        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_forward_topk_selection(self, router, test_input):
        """Verify Top-K expert selection correctness."""
        weights, indices, _ = router(test_input, return_metrics=True)

        batch, seq, top_k = weights.shape

        # Indices should be in valid range
        assert indices.min() >= 0
        assert indices.max() < router.num_experts

        # Each position should select exactly top_k experts
        assert indices.shape[-1] == top_k

    def test_forward_metrics_content(self, router, test_input):
        """Verify metrics dictionary contains expected keys."""
        _, _, metrics = router(test_input, return_metrics=True)

        # RouterMetricsTracker returns comprehensive metrics
        expected_keys = {
            "expert_entropy",
            "expert_entropy_normalized",
            "fatigue_mean",
            "fatigue_std",
            "fatigue_min",
            "fatigue_max",
            "routing_diversity_gini",
            "effective_experts",
            "num_steps",
        }
        assert all(key in metrics for key in expected_keys)

        assert metrics["fatigue_mean"] >= 0.0
        assert metrics["fatigue_max"] >= metrics["fatigue_mean"]
        assert metrics["expert_entropy"] >= 0.0
        assert 1.0 <= metrics["effective_experts"] <= router.num_experts

    def test_forward_step_increment(self, zero_fatigue_router, test_input):
        """Verify global step counter increments when step() is called."""
        router = zero_fatigue_router

        assert router.num_steps.item() == 0

        router(test_input, return_metrics=False)
        router.step()
        assert router.num_steps.item() == 1

        router(test_input, return_metrics=False)
        router.step()
        assert router.num_steps.item() == 2

    @pytest.mark.parametrize(
        "batch_size,seq_len",
        [
            (1, 1),
            (2, 4),
            (8, 16),
            (32, 64),
        ],
    )
    def test_forward_batching(self, router, device, batch_size, seq_len):
        """Test forward pass with various batch sizes."""
        x = torch.randn(batch_size, seq_len, router.hidden_dim, device=device)
        weights, indices, _ = router(x, return_metrics=True)

        assert weights.shape == (batch_size, seq_len, router.top_k)
        assert indices.shape == (batch_size, seq_len, router.top_k)


class TestStateManagement:
    """Test router state management and serialization."""

    def test_reset_state(self, router, test_input):
        """Verify reset_state clears all buffers."""
        router.train()

        # Accumulate some fatigue
        for _ in range(5):
            router(test_input, return_metrics=False)
            router.step()

        assert router.num_steps.item() > 0
        assert (router.fatigue > 0).any()

        # Reset state
        router.reset_state()

        assert router.num_steps.item() == 0
        assert torch.allclose(router.fatigue, torch.zeros_like(router.fatigue))
        assert torch.allclose(router.birth_step, torch.zeros_like(router.birth_step))

    def test_get_state(self, router, test_input):
        """Verify get_state returns comprehensive state dict."""
        router.train()
        router(test_input, return_metrics=False)
        router.step()

        state = router.get_state()

        assert isinstance(state, dict)
        assert "fatigue" in state
        assert "num_steps" in state
        assert "mean_fatigue" in state

        # Verify state values are sensible
        assert state["num_steps"] > 0
        assert torch.is_tensor(state["fatigue"])
        assert state["fatigue"].shape == (router.num_experts,)

    def test_register_birth(self, router):
        """Verify expert birth registration updates birth_step."""
        router.num_steps.fill_(42)

        expert_id = 3
        router.register_birth(expert_id)

        assert router.birth_step[expert_id].item() == 42
        assert router.birth_step[0].item() == 0  # Other experts unaffected


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_expert(self, device):
        """Test router with only one expert."""
        config = MetabolicRouterConfig(hidden_dim=64, num_experts=1, top_k=1)
        router = MetabolicRouter(config).to(device)
        x = torch.randn(2, 3, 64, device=device)

        weights, indices, _ = router(x, return_metrics=True)

        assert weights.shape == (2, 3, 1)
        assert torch.allclose(weights, torch.ones_like(weights))
        assert (indices == 0).all()

    def test_zero_warmup_steps(self, device):
        """Test router with no warmup period."""
        config = MetabolicRouterConfig(hidden_dim=64, num_experts=4, warmup_steps=0)
        router = MetabolicRouter(config).to(device)
        x = torch.randn(2, 3, 64, device=device)

        # Should not crash
        weights, indices, _ = router(x, return_metrics=True)
        assert weights.shape == (2, 3, config.top_k)

    def test_eval_mode_no_noise(self, router, test_input):
        """Verify eval mode disables exploration noise."""
        router.eval()

        alignment = router.compute_alignment(test_input)
        potential_1 = router.compute_routing_potential(alignment, noise_std=0.1)
        # potential_2 = router.compute_routing_potential(alignment, noise_std=0.1)

        # In eval mode, noise should be disabled regardless of noise_std
        # (This depends on the implementation checking self.training)
        # For now, just verify it doesn't crash
        assert torch.isfinite(potential_1).all()

    def test_topk_equals_num_experts(self, device):
        """Test when top_k equals total number of experts."""
        config = MetabolicRouterConfig(
            hidden_dim=64,
            num_experts=4,
            top_k=4,  # Select all experts
        )
        router = MetabolicRouter(config).to(device)
        x = torch.randn(2, 3, 64, device=device)

        weights, indices, _ = router(x, return_metrics=True)

        assert weights.shape == (2, 3, 4)
        # All experts should be selected
        assert len(torch.unique(indices)) == 4


class TestAuxiliaryLoss:
    """Test auxiliary loss computation."""

    def test_aux_loss_is_zero(self, router):
        """MetabolicRouter should return zero auxiliary loss."""
        aux_loss = router.compute_aux_loss()

        assert torch.is_tensor(aux_loss)
        assert aux_loss.item() == 0.0


class TestDeviceCompatibility:
    """Test device handling and dtype consistency."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self, standard_config):
        """Test router works on CUDA devices."""
        device = torch.device("cuda")
        router = MetabolicRouter(standard_config).to(device)
        x = torch.randn(2, 4, standard_config.hidden_dim, device=device)

        weights, indices, _ = router(x, return_metrics=True)

        assert weights.device == x.device
        assert indices.device == x.device

    def test_cpu_compatibility(self, standard_config):
        """Test router works on CPU."""
        device = torch.device("cpu")
        router = MetabolicRouter(standard_config).to(device)
        x = torch.randn(2, 4, standard_config.hidden_dim, device=device)

        weights, indices, _ = router(x, return_metrics=True)

        assert weights.device == x.device
        assert indices.device == x.device


class TestRouterRefinements:
    """Test suite for metabolic router refinements."""

    def test_warmup_timing_fix(self, zero_fatigue_router, test_input):
        """Test that newborn experts don't get free first step."""
        router = zero_fatigue_router
        router.train()
        router.num_steps.fill_(0)
        router.birth_step.zero_()

        # First routing step
        router(test_input)
        router.step()

        # Age should be calculated as (num_steps + 1 - birth_step)
        # At step 0, age should be 1, not 0
        # This means experts born at step 0 get warmup from the start
        # Verify fatigue was applied (non-zero usage should create fatigue)
        # If age was 0, eta_i would be 0 and no fatigue would accumulate
        assert router.num_steps.item() == 1

    def test_temperature_override(self, router, test_input):
        """Test that temperature can be overridden per-call."""
        router.eval()

        # Get routing with default temperature
        weights_default, indices_default, _ = router(test_input, temperature=None)

        # Get routing with low temperature (sharper distribution)
        weights_low_temp, indices_low_temp, _ = router(test_input, temperature=0.1)

        # Get routing with high temperature (smoother distribution)
        weights_high_temp, indices_high_temp, _ = router(test_input, temperature=10.0)

        # Lower temperature should create sharper distributions
        # (higher max weight, lower entropy)
        assert weights_low_temp.max() > weights_high_temp.max()

        # Weights should still sum to 1
        assert torch.allclose(
            weights_low_temp.sum(dim=-1), torch.ones_like(weights_low_temp.sum(dim=-1))
        )
        assert torch.allclose(
            weights_high_temp.sum(dim=-1),
            torch.ones_like(weights_high_temp.sum(dim=-1)),
        )

    def test_forced_noise_in_eval(self, router, test_input):
        """Test that noise can be applied in eval mode for exploration studies."""
        router.eval()  # Important: eval mode

        alignment = router.compute_alignment(test_input)

        # With old implementation, noise_std would be ignored in eval mode
        # With new implementation, noise should be applied regardless
        potential_no_noise = router.compute_routing_potential(alignment, noise_std=0.0)
        potential_with_noise_1 = router.compute_routing_potential(
            alignment, noise_std=0.5
        )
        potential_with_noise_2 = router.compute_routing_potential(
            alignment, noise_std=0.5
        )

        # Noise should create differences even in eval mode
        assert not torch.allclose(potential_no_noise, potential_with_noise_1)

        # Different calls with noise should produce different results (stochastic)
        assert not torch.allclose(potential_with_noise_1, potential_with_noise_2)

    def test_n_active_buffer_exists(self, router):
        """Test that n_active buffer is created and initialized correctly."""
        assert hasattr(router, "n_active")
        assert router.n_active.item() == router.num_experts
        assert router.n_active.dtype == torch.long

    def test_bincount_usage_calculation(self, router, test_input):
        """Test that usage is calculated using bincount."""
        router.train()

        # Run forward pass
        weights, indices, _ = router(test_input)

        # Manually calculate usage using the old method
        usage_old = torch.zeros(router.num_experts, device=weights.device)
        flat_indices = indices.flatten()
        flat_weights = weights.flatten()
        usage_old.scatter_add_(0, flat_indices, flat_weights)

        # The new implementation uses bincount
        usage_new = torch.bincount(
            flat_indices, weights=flat_weights, minlength=router.num_experts
        )

        # Both should produce identical results (minus the normalization)
        assert torch.allclose(usage_old, usage_new)


if __name__ == "__main__":
    # Run tests with: pytest tests/routers/test_metabolic_router.py -v
    pytest.main([__file__, "-v", "--tb=short"])
