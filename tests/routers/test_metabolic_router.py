import torch
import torch.nn.functional as F
from src.configs.router import MetabolicRouterConfig
from src.routers.metabolic import MetabolicRouter


class TestMetabolicRouterConfig:
    def test_default_config_initialization(self):
        config = MetabolicRouterConfig(hidden_dim=512)
        assert config.router_type == "metabolic"
        assert config.hidden_dim > 0
        assert config.num_experts > 0
        assert config.top_k > 0
        assert config.lambda_metabolic >= 0.0
        assert config.gamma_recovery >= 0.0


class TestAlignment:
    def test_alignment_is_bounded_cosine_similarity(self, standard_config, device):
        """Alignment is pure cosine similarity ∈ [-1, 1]."""
        router = MetabolicRouter(standard_config).to(device)
        router.eval()
        x = torch.randn(2, 4, standard_config.hidden_dim, device=device)
        alignment = router.compute_alignment(x)
        assert alignment.min() >= -1.0 - 1e-5
        assert alignment.max() <= 1.0 + 1e-5

    def test_no_learnable_magnitude(self, standard_config, device):
        """No prototype_magnitude — g_i=1 fixed."""
        router = MetabolicRouter(standard_config).to(device)
        param_names = [name for name, _ in router.named_parameters()]
        assert not any("magnitude" in n for n in param_names)


class TestFatiguePenalty:
    def test_penalty_grows_with_raw_fatigue(self, standard_config, device):
        """Raw F_i: penalty scales linearly with fatigue (no SoftSign ceiling)."""
        router = MetabolicRouter(standard_config).to(device)
        router.lambda_metabolic = 1.0

        x = torch.randn(1, 1, standard_config.hidden_dim, device=device)
        alignment = router.compute_alignment(x)

        # Low fatigue
        router.fatigue.data = torch.zeros(standard_config.num_experts, device=device)
        router.fatigue.data[0] = 0.5
        potential_low = router.compute_routing_potential(
            alignment, noise_std=0.0, lambda_scale=1.0
        )

        # High fatigue
        router.fatigue.data[0] = 2.0
        potential_high = router.compute_routing_potential(
            alignment, noise_std=0.0, lambda_scale=1.0
        )

        # Expert 0 should be suppressed more with higher fatigue
        assert potential_high[0, 0, 0] < potential_low[0, 0, 0]

    def test_overloaded_expert_is_suppressed(self, standard_config, device):
        """Expert with positive fatigue has lower routing potential."""
        router = MetabolicRouter(standard_config).to(device)
        router.lambda_metabolic = 1.0
        router.fatigue.data = torch.zeros(standard_config.num_experts, device=device)
        router.fatigue.data[0] = 3.0  # very overloaded

        x = torch.randn(1, 1, standard_config.hidden_dim, device=device)

        # Force expert 0 to have positive alignment for a clear test
        with torch.no_grad():
            router.gate.weight[0] = F.normalize(x.squeeze(), p=2, dim=-1)

        alignment = router.compute_alignment(x)
        potential_with_penalty = router.compute_routing_potential(
            alignment, noise_std=0.0, lambda_scale=1.0
        )
        potential_no_penalty = router.compute_routing_potential(
            alignment, noise_std=0.0, lambda_scale=0.0
        )

        # Expert 0 should have lower potential when it's overloaded
        assert potential_with_penalty[0, 0, 0] < potential_no_penalty[0, 0, 0]

    def test_warmup_gives_zero_penalty(self, standard_config, device):
        """At lambda_scale=0 (warmup), fatigue has no effect — uniform penalty."""
        router = MetabolicRouter(standard_config).to(device)
        router.fatigue.data = torch.randn(standard_config.num_experts, device=device)

        x = torch.randn(1, 1, standard_config.hidden_dim, device=device)
        alignment = router.compute_alignment(x)

        # During warmup: lambda_scale=0 → no fatigue penalty applied
        potential_warmup = router.compute_routing_potential(
            alignment, noise_std=0.0, lambda_scale=0.0
        )
        assert torch.allclose(potential_warmup, alignment, atol=1e-5)

    def test_underused_expert_gets_bonus(self, standard_config, device):
        """Expert with negative fatigue (underused) gets a routing bonus."""
        router = MetabolicRouter(standard_config).to(device)
        router.lambda_metabolic = 1.0
        router.fatigue.data = torch.zeros(standard_config.num_experts, device=device)
        router.fatigue.data[0] = -2.0  # very underused

        x = torch.randn(1, 1, standard_config.hidden_dim, device=device)
        alignment = router.compute_alignment(x)

        potential_with = router.compute_routing_potential(
            alignment, noise_std=0.0, lambda_scale=1.0
        )
        potential_without = router.compute_routing_potential(
            alignment, noise_std=0.0, lambda_scale=0.0
        )

        # Underused expert gets a bonus (penalty is negative → potential increases)
        assert potential_with[0, 0, 0] > potential_without[0, 0, 0]


class TestFatigueDynamics:
    def test_fatigue_accumulates_with_usage(self, zero_fatigue_router, test_input):
        router = zero_fatigue_router
        router.train()
        router.beta_cost = 1.0
        router.gamma_recovery = 0.0

        for _ in range(5):
            router(test_input)
            router.step()

        assert (router.fatigue != 0).any()
        assert router.num_steps.item() == 5

    def test_fatigue_zero_sum(self, zero_fatigue_router, test_input):
        """Σ F_i should stay near 0 (differential fatigue is zero-sum)."""
        router = zero_fatigue_router
        router.train()

        for _ in range(10):
            router(test_input)
            router.step()

        assert abs(router.fatigue.sum().item()) < 1e-4

    def test_fatigue_recovers(self, router, device):
        router.fatigue.data = torch.ones(router.num_experts, device=device) * 2.0
        initial = router.fatigue.clone()
        router.gamma_recovery = 0.1

        with torch.no_grad():
            router.fatigue.mul_(1 - router.gamma_recovery)

        assert (router.fatigue < initial).all()


class TestForwardPass:
    def test_forward_output_shapes(self, router, test_input):
        batch, seq, hidden = test_input.shape
        weights, indices, metrics = router(test_input, return_metrics=True)

        assert weights.shape == (batch, seq, router.top_k)
        assert indices.shape == (batch, seq, router.top_k)
        assert torch.allclose(
            weights.sum(dim=-1),
            torch.ones(batch, seq, device=weights.device),
            atol=1e-5,
        )
        assert indices.min() >= 0
        assert indices.max() < router.num_experts

    def test_step_increments_counter(self, zero_fatigue_router, test_input):
        router = zero_fatigue_router
        router(test_input)
        assert router._usage_pending is True

        router.step()
        assert router.num_steps == 1
        assert router._usage_pending is False

    def test_no_aux_loss(self, router):
        assert router.compute_aux_loss().item() == 0.0


class TestConfidenceMetrics:
    def test_confidence_metrics_present(self, router, test_input):
        router.train()
        _, _, metrics = router(test_input, return_metrics=True)
        assert "router_confidence_mean" in metrics
        assert "top1_dominance" in metrics

    def test_confidence_in_valid_range(self, router, test_input):
        router.eval()
        _, _, metrics = router(test_input, return_metrics=True)
        assert 0 < metrics["router_confidence_mean"] <= 1.0

    def test_top1_dominance_equals_one_for_topk1(self, device):
        config = MetabolicRouterConfig(hidden_dim=64, num_experts=4, top_k=1)
        router = MetabolicRouter(config).to(device)
        router.eval()
        x = torch.randn(2, 4, 64, device=device)
        _, _, metrics = router(x, return_metrics=True)
        assert abs(metrics["top1_dominance"] - 1.0) < 1e-5
