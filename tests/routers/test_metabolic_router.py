import torch

import torch.nn.functional as F

import math

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

        assert config.tau_specialization > 0.0

        assert config.F_scale > 0.0

    def test_v6_params_present(self):
        config = MetabolicRouterConfig(hidden_dim=64)

        assert hasattr(config, "tau_specialization")

        assert hasattr(config, "F_scale")


class TestAlignment:
    def test_alignment_is_bounded_cosine_similarity(self, standard_config, device):
        router = MetabolicRouter(standard_config).to(device)

        router.eval()

        x = torch.randn(2, 4, standard_config.hidden_dim, device=device)

        alignment = router.compute_alignment(x)

        assert alignment.min() >= -1.0 - 1e-5

        assert alignment.max() <= 1.0 + 1e-5

    def test_no_learnable_magnitude(self, standard_config, device):
        router = MetabolicRouter(standard_config).to(device)

        param_names = [name for name, _ in router.named_parameters()]

        assert not any("magnitude" in n for n in param_names)


class TestTanhPenalty:
    def test_penalty_bounded_by_lambda(self, standard_config, device):
        router = MetabolicRouter(standard_config).to(device)

        router.lambda_metabolic = 1.0

        x = torch.randn(1, 1, standard_config.hidden_dim, device=device)

        alignment = router.compute_alignment(x)

        router.fatigue.data = torch.full(
            (standard_config.num_experts,), 1000.0, device=device
        )

        potential = router.compute_routing_potential(
            alignment, noise_std=0.0, lambda_scale=1.0
        )

        assert potential.min().item() >= -2.0 - 1e-4

    def test_penalty_grows_with_fatigue_until_saturation(self, standard_config, device):
        router = MetabolicRouter(standard_config).to(device)

        router.lambda_metabolic = 1.0

        x = torch.randn(1, 1, standard_config.hidden_dim, device=device)

        alignment = router.compute_alignment(x)

        potentials = []

        for f_val in [0.0, 0.5, 2.0, 10.0, 100.0]:
            router.fatigue.data = torch.zeros(
                standard_config.num_experts, device=device
            )

            router.fatigue.data[0] = f_val

            p = router.compute_routing_potential(
                alignment, noise_std=0.0, lambda_scale=1.0
            )

            potentials.append(p[0, 0, 0].item())

        for i in range(len(potentials) - 1):
            assert potentials[i] >= potentials[i + 1] - 1e-5

        assert potentials[0] - potentials[-1] <= 2.0 + 1e-4

    def test_overloaded_expert_is_suppressed(self, standard_config, device):
        router = MetabolicRouter(standard_config).to(device)

        router.lambda_metabolic = 1.0

        router.fatigue.data = torch.zeros(standard_config.num_experts, device=device)

        router.fatigue.data[0] = 3.0

        x = torch.randn(1, 1, standard_config.hidden_dim, device=device)

        with torch.no_grad():
            router.gate.weight[0] = F.normalize(x.squeeze(), p=2, dim=-1)

        alignment = router.compute_alignment(x)

        with_penalty = router.compute_routing_potential(
            alignment, noise_std=0.0, lambda_scale=1.0
        )

        no_penalty = router.compute_routing_potential(
            alignment, noise_std=0.0, lambda_scale=0.0
        )

        assert with_penalty[0, 0, 0] < no_penalty[0, 0, 0]

    def test_warmup_gives_zero_penalty(self, standard_config, device):
        router = MetabolicRouter(standard_config).to(device)

        router.fatigue.data = torch.randn(standard_config.num_experts, device=device)

        x = torch.randn(1, 1, standard_config.hidden_dim, device=device)

        alignment = router.compute_alignment(x)

        potential_warmup = router.compute_routing_potential(
            alignment, noise_std=0.0, lambda_scale=0.0
        )

        assert torch.allclose(potential_warmup, alignment, atol=1e-5)

    def test_zero_fatigue_no_penalty(self, standard_config, device):
        router = MetabolicRouter(standard_config).to(device)

        router.fatigue.data = torch.zeros(standard_config.num_experts, device=device)

        x = torch.randn(1, 1, standard_config.hidden_dim, device=device)

        alignment = router.compute_alignment(x)

        potential = router.compute_routing_potential(
            alignment, noise_std=0.0, lambda_scale=1.0
        )

        assert torch.allclose(potential, alignment, atol=1e-5)

    def test_tanh_correct_value(self, device):
        config = MetabolicRouterConfig(
            hidden_dim=4,
            num_experts=2,
            top_k=1,
            lambda_metabolic=1.0,
            F_scale=1.0,
            warmup_steps=0,
        )

        router = MetabolicRouter(config).to(device)

        router.fatigue.data = torch.tensor([1.0, 0.0], device=device)

        x = torch.randn(1, 1, 4, device=device)

        alignment = router.compute_alignment(x)

        potential = router.compute_routing_potential(
            alignment, noise_std=0.0, lambda_scale=1.0
        )

        expected_0 = alignment[0, 0, 0] - math.tanh(1.0)

        assert abs(potential[0, 0, 0].item() - expected_0) < 1e-4

        assert abs(potential[0, 0, 1].item() - alignment[0, 0, 1].item()) < 1e-4


class TestFatigueDynamics:
    def test_one_sided_update_underused_experts_dont_accumulate(
        self, standard_config, device
    ):
        router = MetabolicRouter(standard_config).to(device)

        router.gamma_recovery = 0.0

        router.beta_cost = 1.0

        router.tau_specialization = 2.0

        router.fatigue.data = torch.zeros(standard_config.num_experts, device=device)

        N = standard_config.num_experts

        uniform_usage = torch.full((N,), 1.0 / N, device=device)

        router.update_fatigue(uniform_usage)

        assert router.fatigue.sum().item() == 0.0

    def test_one_sided_update_overloaded_expert_accumulates(
        self, standard_config, device
    ):
        router = MetabolicRouter(standard_config).to(device)

        router.gamma_recovery = 0.0

        router.beta_cost = 1.0

        router.tau_specialization = 1.0

        router.fatigue.data = torch.zeros(standard_config.num_experts, device=device)

        N = standard_config.num_experts

        overloaded = torch.zeros(N, device=device)

        overloaded[0] = 1.0

        router.update_fatigue(overloaded)

        assert router.fatigue[0].item() > 0.0

        assert router.fatigue[1:].sum().item() == 0.0

    def test_tau_free_zone_exact_threshold(self, standard_config, device):
        router = MetabolicRouter(standard_config).to(device)

        router.gamma_recovery = 0.0

        router.beta_cost = 1.0

        router.tau_specialization = 2.0

        router.fatigue.data = torch.zeros(standard_config.num_experts, device=device)

        N = standard_config.num_experts

        tau = 2.0

        usage = torch.full((N,), 0.0, device=device)

        usage[0] = tau / N

        usage[1:] = (1.0 - tau / N) / (N - 1)

        router.update_fatigue(usage)

        assert router.fatigue[0].item() == 0.0

    def test_fatigue_non_negative(self, standard_config, device):
        router = MetabolicRouter(standard_config).to(device)

        router.gamma_recovery = 0.5

        router.fatigue.data = (
            torch.ones(standard_config.num_experts, device=device) * 0.01
        )

        N = standard_config.num_experts

        usage = torch.zeros(N, device=device)

        for _ in range(20):
            router.update_fatigue(usage)

        assert (router.fatigue >= 0.0).all()

    def test_fatigue_accumulates_over_steps(self, zero_fatigue_router, test_input):
        router = zero_fatigue_router

        router.train()

        router.beta_cost = 1.0

        router.gamma_recovery = 0.0

        router.tau_specialization = 0.0

        for _ in range(5):
            router(test_input)

            router.step()

        assert (router.fatigue != 0).any()

        assert router.num_steps.item() == 5

    def test_fatigue_recovers_when_idle(self, router, device):
        router.fatigue.data = torch.ones(router.num_experts, device=device) * 2.0

        initial = router.fatigue.clone()

        router.gamma_recovery = 0.1

        usage = torch.zeros(router.num_experts, device=device)

        router.update_fatigue(usage)

        assert (router.fatigue < initial).all()

    def test_fraction_penalised_metric(self, standard_config, device):
        router = MetabolicRouter(standard_config).to(device)

        router.tau_specialization = 2.0

        N = standard_config.num_experts

        usage = torch.full((N,), 1.0 / N, device=device)

        router.update_fatigue(usage)

        assert router._last_fraction_penalised == 0.0

        spike = torch.zeros(N, device=device)

        spike[0] = 1.0

        router.update_fatigue(spike)

        assert router._last_fraction_penalised > 0.0


class TestForwardPass:
    def test_forward_output_shapes(self, router, test_input):
        batch, seq, hidden = test_input.shape

        weights, indices, metrics = router(test_input, return_metrics=True)

        N = batch * seq

        assert weights.shape == (N, router.num_experts)

        assert indices is None

        row_sums = weights.sum(dim=-1)

        assert torch.allclose(row_sums, torch.ones(N, device=weights.device), atol=1e-5)

        assert (weights > 0).sum(dim=-1).eq(router.top_k).all()

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
        config = MetabolicRouterConfig(
            hidden_dim=64,
            num_experts=4,
            top_k=1,
            tau_specialization=2.0,
            F_scale=0.5,
        )

        router = MetabolicRouter(config).to(device)

        router.eval()

        x = torch.randn(2, 4, 64, device=device)

        _, _, metrics = router(x, return_metrics=True)

        assert abs(metrics["top1_dominance"] - 1.0) < 1e-5


class TestGetState:
    def test_get_state_keys(self, router):
        state = router.get_state()

        for key in (
            "fatigue",
            "num_steps",
            "mean_fatigue",
            "max_fatigue",
            "min_fatigue",
            "lambda_eff",
            "fatigue_tanh_mean",
            "fairshare",
            "fraction_penalised",
        ):
            assert key in state, f"Missing key: {key}"

    def test_fairshare_value(self, standard_config, device):
        router = MetabolicRouter(standard_config).to(device)

        state = router.get_state()

        expected = standard_config.tau_specialization / standard_config.num_experts

        assert abs(state["fairshare"] - expected) < 1e-6

    def test_lambda_eff_zero_at_start(self, standard_config, device):
        router = MetabolicRouter(standard_config).to(device)

        router.train()

        router._step_count = 0

        state = router.get_state()

        assert state["lambda_eff"] == 0.0

    def test_lambda_eff_full_after_warmup(self, standard_config, device):
        router = MetabolicRouter(standard_config).to(device)

        router._step_count = standard_config.warmup_steps

        state = router.get_state()

        assert abs(state["lambda_eff"] - standard_config.lambda_metabolic) < 1e-6

    def test_fatigue_tanh_mean_bounded(self, standard_config, device):
        router = MetabolicRouter(standard_config).to(device)

        router.fatigue.data = (
            torch.randn(standard_config.num_experts, device=device) * 100
        )

        state = router.get_state()

        assert -1.0 <= state["fatigue_tanh_mean"] <= 1.0
