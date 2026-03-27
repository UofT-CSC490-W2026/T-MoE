import torch
import torch.nn.functional as F
import pytest
from src.configs.router import StressCorrectedRouterConfig
from src.routers.stress_corrected import StressCorrectedRouter


@pytest.fixture
def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_router(hidden_dim=64, num_experts=4, top_k=2, temperature=1.0, device="cpu"):
    cfg = StressCorrectedRouterConfig(
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        temperature=temperature,
        noise_std=0.0,
    )

    return StressCorrectedRouter(cfg).to(device)


def make_input(batch=2, seq=4, hidden=64, device="cpu"):
    return torch.randn(batch, seq, hidden, device=device)


def run_steps(router, x, n):
    for _ in range(n):
        router(x, record_usage=True)
        router.step()


CALIB_STEP = 200


def make_router_calib(calib_step=CALIB_STEP, device="cpu", **kwargs):
    cfg = StressCorrectedRouterConfig(
        hidden_dim=kwargs.get("hidden_dim", 64),
        num_experts=kwargs.get("num_experts", 4),
        top_k=kwargs.get("top_k", 2),
        temperature=1.0,
        noise_std=0.0,
        lambda_calib_step=calib_step,
    )

    return StressCorrectedRouter(cfg).to(device)


class TestForwardShapes:
    def test_output_shapes(self, device):
        router = make_router(hidden_dim=64, num_experts=4, top_k=2, device=device)
        x = make_input(batch=3, seq=6, hidden=64, device=device)
        weights, indices, _ = router(x)
        N = 3 * 6
        assert weights.shape == (N, 4)
        assert indices is None
        assert (weights > 0).sum(dim=-1).eq(2).all()

    def test_weights_sum_to_one(self, device):
        router = make_router(device=device)
        x = make_input(batch=4, seq=8, device=device)
        weights, _, _ = router(x)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_weights_nonneg(self, device):
        router = make_router(device=device)
        x = make_input(device=device)
        weights, _, _ = router(x)
        assert (weights >= 0).all()

    def test_metrics_returned_when_requested(self, device):
        router = make_router(device=device)
        x = make_input(device=device)
        _, _, metrics = router(x, return_metrics=True)
        assert metrics is not None


class TestLambdaCalibration:
    def test_lambda_stays_default_before_calib_step(self, device):
        router = make_router_calib(device=device)
        router.train()
        x = make_input(device=device)
        for _ in range(CALIB_STEP - 1):
            router(x, record_usage=True)
            router.step()
            assert not router.lambda_initialized.item()
            assert router.lambda_val.item() == pytest.approx(0.1)

    def test_lambda_initializes_at_calib_step(self, device):
        router = make_router_calib(device=device)
        router.train()
        x = make_input(device=device)
        run_steps(router, x, CALIB_STEP - 1)
        assert not router.lambda_initialized.item()
        router(x, record_usage=True)
        router.step()
        assert router.lambda_initialized.item()

    def test_lambda_not_initialized_twice(self, device):
        router = make_router_calib(device=device)
        router.train()
        x = make_input(device=device)
        run_steps(router, x, CALIB_STEP + 10)
        val = router.lambda_val.item()
        run_steps(router, x, 10)
        assert router.lambda_val.item() == pytest.approx(val)

    def test_lambda_capped_at_5(self, device):
        router = make_router_calib(device=device)
        router.train()
        x = make_input(device=device)
        run_steps(router, x, CALIB_STEP + 1)
        assert router.lambda_val.item() <= 5.0 + 1e-6

    def test_lambda_positive(self, device):
        router = make_router_calib(device=device)
        router.train()
        x = make_input(device=device)
        run_steps(router, x, CALIB_STEP + 1)
        assert router.lambda_val.item() > 0.0

    def test_step_without_forward_no_crash(self, device):
        router = make_router_calib(calib_step=10, device=device)
        router.train()
        for _ in range(15):
            router.step()
        assert not router.lambda_initialized.item()


class TestOneSidedPenalty:
    def test_zero_penalty_at_fair_share(self, device):
        router = make_router(num_experts=4, device=device)
        router.eval()
        fair = 1.0 / 4
        router.ema_load.fill_(fair)
        with torch.no_grad():
            penalty = (router.ema_load - fair).clamp(min=0.0)
        assert penalty.max().item() == pytest.approx(0.0, abs=1e-7), (
            "One-sided penalty must be zero when all experts are at fair share"
        )

    def test_penalty_only_for_overloaded(self, device):
        router = make_router(num_experts=4, device=device)
        router.eval()
        router.ema_load.copy_(torch.tensor([0.5, 0.1, 0.2, 0.2], device=device))
        fair = 1.0 / 4
        penalty = (router.ema_load - fair).clamp(min=0.0)
        assert penalty[0].item() > 0.0, "Overloaded expert should have positive penalty"
        assert penalty[1].item() == pytest.approx(0.0, abs=1e-7), (
            "Underloaded expert should have zero penalty"
        )

    def test_overloaded_expert_selected_less(self, device):
        router = make_router(num_experts=4, top_k=1, temperature=0.5, device=device)
        router.eval()
        with torch.no_grad():
            router.ema_load.copy_(torch.tensor([0.7, 0.1, 0.1, 0.1], device=device))
            direction = F.normalize(torch.randn(64, device=device), dim=-1)
            router.W[0] = direction.clone()
            router.W[1] = direction.clone()
            router.W[2] = F.normalize(torch.randn(64, device=device), dim=-1)
            router.W[3] = F.normalize(torch.randn(64, device=device), dim=-1)
            router.lambda_val.fill_(2.0)
        x = direction.unsqueeze(0).unsqueeze(0)
        count_0, count_1 = 0, 0
        for _ in range(20):
            weights, _, _ = router(x)
            selected = weights[0].nonzero().squeeze(-1).tolist()
            if 0 in selected:
                count_0 += 1
            if 1 in selected:
                count_1 += 1
        assert count_1 > count_0, (
            f"Underloaded expert 1 should win over overloaded expert 0 with top_k=1. "
            f"count_0={count_0}, count_1={count_1}"
        )

    def test_output_weights_use_cosine_only(self, device):
        router = make_router(temperature=1.0, device=device)
        router.eval()
        x = make_input(device=device)
        weights, _, _ = router(x)
        with torch.no_grad():
            W_norm = F.normalize(router.W, dim=-1)
            x_norm = F.normalize(x, dim=-1)
            cos_sim = x_norm @ W_norm.T
            topk_cos, topk_idx = cos_sim.topk(router.top_k, dim=-1)
            expected_sparse = F.softmax(topk_cos / router.temperature, dim=-1)
            N = x.shape[0] * x.shape[1]
            expected = torch.zeros(N, router.num_experts, device=device)
            expected.scatter_(1, topk_idx.view(N, -1), expected_sparse.view(N, -1))
        assert torch.allclose(weights, expected, atol=1e-5), (
            "Output weights must equal softmax(cos/τ) regardless of load penalty"
        )


class TestTauAnnealing:
    def _make_annealing_router(
        self, tau_start=1.0, tau_final=0.1, anneal_steps=100, device="cpu"
    ):
        cfg = StressCorrectedRouterConfig(
            hidden_dim=64,
            num_experts=4,
            top_k=2,
            temperature=tau_start,
            tau_final=tau_final,
            tau_anneal_steps=anneal_steps,
            noise_std=0.0,
        )
        return StressCorrectedRouter(cfg).to(device)

    def test_tau_starts_at_temperature(self, device):
        router = self._make_annealing_router(device=device)
        assert router._current_tau() == pytest.approx(router.temperature)

    def test_tau_decreases_over_steps(self, device):
        router = self._make_annealing_router(device=device)
        router.train()
        x = make_input(device=device)
        tau_0 = router._current_tau()
        run_steps(router, x, 50)
        tau_50 = router._current_tau()
        run_steps(router, x, 50)
        tau_100 = router._current_tau()
        assert tau_50 < tau_0, "τ must decrease during annealing"
        assert tau_100 < tau_50, "τ must keep decreasing until tau_anneal_steps"

    def test_tau_clamps_at_tau_final(self, device):
        router = self._make_annealing_router(
            tau_start=1.0, tau_final=0.1, anneal_steps=50, device=device
        )
        router.train()
        x = make_input(device=device)
        run_steps(router, x, 100)
        assert router._current_tau() == pytest.approx(0.1, abs=1e-6)

    def test_no_annealing_when_disabled(self, device):
        cfg = StressCorrectedRouterConfig(
            hidden_dim=64,
            num_experts=4,
            top_k=2,
            temperature=0.5,
            tau_final=0.1,
            tau_anneal_steps=0,
            noise_std=0.0,
        )
        router = StressCorrectedRouter(cfg).to(device)
        router.train()
        x = make_input(device=device)
        run_steps(router, x, 200)
        assert router._current_tau() == pytest.approx(0.5)

    def test_no_annealing_when_tau_final_equals_temperature(self, device):
        cfg = StressCorrectedRouterConfig(
            hidden_dim=64,
            num_experts=4,
            top_k=2,
            temperature=0.5,
            tau_final=0.5,
            tau_anneal_steps=100,
            noise_std=0.0,
        )
        router = StressCorrectedRouter(cfg).to(device)
        router.train()
        x = make_input(device=device)
        run_steps(router, x, 100)
        assert router._current_tau() == pytest.approx(0.5)

    def test_tau_affects_output_weight_sharpness(self, device):
        router_sharp = self._make_annealing_router(
            tau_start=1.0, tau_final=0.1, anneal_steps=10, device=device
        )
        router_sharp.train()
        x = make_input(device=device)
        run_steps(router_sharp, x, 20)
        router_sharp.eval()
        router_flat = make_router(temperature=1.0, device=device)
        router_flat.eval()
        with torch.no_grad():
            router_sharp.W.copy_(router_flat.W)
        x_test = make_input(batch=4, seq=8, device=device)
        weights_sharp, _, _ = router_sharp(x_test)
        weights_flat, _, _ = router_flat(x_test)
        entropy_sharp = (
            -(weights_sharp * weights_sharp.clamp(min=1e-10).log())
            .sum(-1)
            .mean()
            .item()
        )
        entropy_flat = (
            -(weights_flat * weights_flat.clamp(min=1e-10).log()).sum(-1).mean().item()
        )
        assert entropy_sharp < entropy_flat, (
            "Lower τ must produce sharper (lower-entropy) output weights"
        )


class TestWelford:
    def test_welford_accumulates_after_forward(self, device):
        router = make_router(device=device)
        router.train()
        x = make_input(device=device)
        n_before = router.welford_n.sum().item()
        for _ in range(5):
            router(x, record_usage=True)
        n_after = router.welford_n.sum().item()
        assert n_after > n_before, (
            "Welford_n must increase after training forward passes"
        )

    def test_welford_not_in_logit(self, device):
        router = make_router(device=device)
        router.eval()
        with torch.no_grad():
            router.welford_n.fill_(1000.0)
            router.welford_mu.fill_(0.3)
            router.welford_M2.fill_(10.0)
        x = make_input(device=device)
        _, idx_with, _ = router(x)
        with torch.no_grad():
            router.welford_n.zero_()
            router.welford_mu.zero_()
            router.welford_M2.zero_()
        _, idx_without, _ = router(x)
        w_with, _, _ = router(x)
        router.welford_n.zero_()
        router.welford_mu.zero_()
        router.welford_M2.zero_()
        w_without, _, _ = router(x)
        assert torch.allclose(w_with, w_without, atol=1e-6), (
            "Welford state must not affect routing decisions (metrics-only)"
        )


class TestNoiseAnnealing:
    def _make_annealing_router(self, noise_std=0.1, anneal_steps=100, device="cpu"):
        cfg = StressCorrectedRouterConfig(
            hidden_dim=64,
            num_experts=4,
            top_k=2,
            temperature=1.0,
            noise_std=noise_std,
            noise_anneal_steps=anneal_steps,
        )
        return StressCorrectedRouter(cfg).to(device)

    def test_noise_std_starts_at_config_value(self, device):
        router = self._make_annealing_router(device=device)
        assert router._current_noise_std() == pytest.approx(router.noise_std)

    def test_noise_std_decreases_over_steps(self, device):
        router = self._make_annealing_router(device=device)
        router.train()
        x = make_input(device=device)
        n0 = router._current_noise_std()
        run_steps(router, x, 50)
        n50 = router._current_noise_std()
        run_steps(router, x, 50)
        n100 = router._current_noise_std()
        assert n50 < n0, "noise_std must decrease during annealing"
        assert n100 < n50, "noise_std must keep decreasing until noise_anneal_steps"

    def test_noise_std_reaches_zero_at_anneal_steps(self, device):
        router = self._make_annealing_router(anneal_steps=50, device=device)
        router.train()
        x = make_input(device=device)
        run_steps(router, x, 100)
        assert router._current_noise_std() == pytest.approx(0.0, abs=1e-7)

    def test_no_annealing_when_disabled(self, device):
        cfg = StressCorrectedRouterConfig(
            hidden_dim=64,
            num_experts=4,
            top_k=2,
            temperature=1.0,
            noise_std=0.05,
            noise_anneal_steps=0,
        )
        router = StressCorrectedRouter(cfg).to(device)
        router.train()
        x = make_input(device=device)
        run_steps(router, x, 200)
        assert router._current_noise_std() == pytest.approx(0.05)

    def test_noise_std_in_get_custom_metrics(self, device):
        router = self._make_annealing_router(
            noise_std=0.1, anneal_steps=100, device=device
        )
        router.train()
        x = make_input(device=device)
        run_steps(router, x, 50)
        router.eval()
        weights, _, _ = router(x, return_metrics=True)
        metrics = router.get_custom_metrics(None, weights)
        assert "noise_std" in metrics
        assert metrics["noise_std"] == pytest.approx(router._noise_std)


class TestPrototypeInit:
    def test_initialize_prototypes_changes_W(self, device):
        router = make_router(hidden_dim=64, num_experts=4, device=device)
        W_before = router.W.data.clone()
        activations = torch.randn(200, 64, device=device)
        router.initialize_prototypes_from_data(activations)
        assert not torch.allclose(router.W.data, W_before), (
            "W must change after prototype initialization"
        )

    def test_initialized_prototypes_are_unit_normalized(self, device):
        router = make_router(hidden_dim=64, num_experts=4, device=device)
        activations = torch.randn(200, 64, device=device)
        router.initialize_prototypes_from_data(activations)
        norms = router.W.data.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
            "All prototype rows must be unit-normalized"
        )

    def test_initialized_prototypes_improve_cosine_similarity(self, device):
        torch.manual_seed(0)
        hidden_dim, num_experts = 64, 4
        cluster_dirs = F.normalize(
            torch.randn(num_experts, hidden_dim, device=device), dim=-1
        )
        activations = []
        for i in range(num_experts):
            tokens = cluster_dirs[i].unsqueeze(0) + 0.1 * torch.randn(
                50, hidden_dim, device=device
            )
            activations.append(tokens)
        activations = torch.cat(activations, dim=0)
        router = make_router(
            hidden_dim=hidden_dim, num_experts=num_experts, device=device
        )
        router.initialize_prototypes_from_data(activations, n_iter=30)
        W_norm = F.normalize(router.W.data, dim=-1)
        x_norm = F.normalize(activations, dim=-1)
        cos_sim = (x_norm @ W_norm.T).max(dim=-1).values
        mean_cos = cos_sim.mean().item()
        assert mean_cos > 0.3, (
            f"Initialized prototypes should align with clustered data (got {mean_cos:.3f})"
        )

    def test_initialize_prototypes_no_crash_with_exact_k_tokens(self, device):
        router = make_router(hidden_dim=64, num_experts=4, device=device)
        activations = torch.randn(4, 64, device=device)
        router.initialize_prototypes_from_data(activations, n_iter=5)

    def test_kmeans_init_all_experts_assigned(self, device):
        from src.routers.stress_corrected import _kmeans_init

        torch.manual_seed(1)
        k = 4
        activations = torch.cat(
            [torch.randn(50, 32, device=device) + i * 10.0 for i in range(k)], dim=0
        )
        centroids = _kmeans_init(activations, k=k, n_iter=20)
        assert centroids.shape == (k, 32)
        norms = centroids.norm(dim=-1)
        assert (norms > 0).all(), (
            "All centroids must be assigned for well-separated clusters"
        )
