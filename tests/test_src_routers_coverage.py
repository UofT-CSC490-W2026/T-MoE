import pytest

import torch

from unittest.mock import MagicMock


def test_base_router_step_noop():
    from src.routers.base import BaseRouter

    class ConcreteRouter(BaseRouter):
        def forward(self, x, **kw):
            pass

        def compute_aux_loss(self):
            return torch.tensor(0.0)

    cfg = MagicMock()

    cfg.num_experts = 4

    cfg.top_k = 2

    cfg.hidden_dim = 64

    r = ConcreteRouter(cfg)

    r.step()


def test_base_router_reset_state():
    from src.routers.base import BaseRouter

    class ConcreteRouter(BaseRouter):
        def forward(self, x, **kw):
            pass

        def compute_aux_loss(self):
            return torch.tensor(0.0)

    cfg = MagicMock()

    cfg.num_experts = 4

    cfg.top_k = 2

    cfg.hidden_dim = 64

    r = ConcreteRouter(cfg)

    r.reset_state()


def test_base_router_clear_aux_state():
    from src.routers.base import BaseRouter

    class ConcreteRouter(BaseRouter):
        def forward(self, x, **kw):
            pass

        def compute_aux_loss(self):
            return torch.tensor(0.0)

    cfg = MagicMock()

    cfg.num_experts = 4

    cfg.top_k = 2

    cfg.hidden_dim = 64

    r = ConcreteRouter(cfg)

    r.clear_aux_state()


def test_base_router_get_state():
    from src.routers.base import BaseRouter

    class ConcreteRouter(BaseRouter):
        def forward(self, x, **kw):
            pass

        def compute_aux_loss(self):
            return torch.tensor(0.0)

    cfg = MagicMock()

    cfg.num_experts = 4

    cfg.top_k = 2

    cfg.hidden_dim = 64

    r = ConcreteRouter(cfg)

    state = r.get_state()

    assert state == {}


def _make_standard_router(hidden=64, num_experts=4, top_k=2, use_aux=False):
    from src.routers.standard import StandardRouter

    from src.configs.router import StandardRouterConfig

    cfg = StandardRouterConfig(
        hidden_dim=hidden, num_experts=num_experts, top_k=top_k, use_aux_loss=use_aux
    )

    return StandardRouter(cfg)


def test_standard_router_forward_eval():
    r = _make_standard_router()

    r.eval()

    x = torch.randn(2, 4, 64)

    weights, indices, metrics = r(x)

    assert weights.shape == (8, 4)

    assert metrics is None


def test_standard_router_forward_with_metrics():
    r = _make_standard_router()

    x = torch.randn(2, 4, 64)

    weights, indices, metrics = r(x, return_metrics=True)

    assert metrics is not None

    assert "eff_E_hard" in metrics


def test_standard_router_aux_loss_not_training():
    r = _make_standard_router(use_aux=True)

    r.eval()

    loss = r.compute_aux_loss()

    assert loss.item() == 0.0


def test_standard_router_aux_loss_no_cache():
    r = _make_standard_router(use_aux=True)

    r.train()

    loss = r.compute_aux_loss()

    assert loss.item() == 0.0


def test_standard_router_aux_loss_with_cache():
    r = _make_standard_router(use_aux=True)

    r.train()

    x = torch.randn(2, 4, 64)

    r(x)

    loss = r.compute_aux_loss()

    assert isinstance(loss.item(), float)


def test_standard_router_clear_aux_state():
    r = _make_standard_router(use_aux=True)

    r.train()

    x = torch.randn(2, 4, 64)

    r(x)

    r.clear_aux_state()

    assert r._last_probs is None


def test_standard_router_get_state():
    r = _make_standard_router()

    assert r.get_state() == {}


def test_standard_router_compute_aux_loss_full():
    r = _make_standard_router(use_aux=True)

    r.train()

    x = torch.randn(4, 8, 64)

    r(x)

    loss = r.compute_aux_loss()

    assert loss.item() >= 0.0


def _make_deepseek_router(hidden=64, num_experts=4, top_k=2, use_sigmoid=False):
    from src.routers.deepseek import DeepSeekRouter

    from src.configs.router import DeepSeekRouterConfig

    cfg = DeepSeekRouterConfig(
        hidden_dim=hidden, num_experts=num_experts, top_k=top_k, use_sigmoid=use_sigmoid
    )

    return DeepSeekRouter(cfg)


def test_deepseek_router_forward():
    r = _make_deepseek_router()

    x = torch.randn(2, 4, 64)

    weights, _, metrics = r(x)

    assert weights.shape == (8, 4)


def test_deepseek_router_forward_sigmoid():
    r = _make_deepseek_router(use_sigmoid=True)

    x = torch.randn(2, 4, 64)

    weights, _, metrics = r(x)

    assert weights.shape == (8, 4)


def test_deepseek_router_forward_with_metrics():
    r = _make_deepseek_router()

    x = torch.randn(2, 4, 64)

    weights, _, metrics = r(x, return_metrics=True)

    assert metrics is not None

    assert "eff_E_hard" in metrics


def test_deepseek_router_forward_with_noise():
    r = _make_deepseek_router()

    r.train()

    r.noise_std = 0.1

    x = torch.randn(2, 4, 64)

    weights, _, _ = r(x)

    assert weights.shape == (8, 4)


def test_deepseek_router_step():
    r = _make_deepseek_router()

    r.train()

    x = torch.randn(2, 4, 64)

    r(x)

    r.step()

    assert not r._usage_pending


def test_deepseek_router_step_no_pending():
    r = _make_deepseek_router()

    r.step()


def test_deepseek_router_reset_state():
    r = _make_deepseek_router()

    r.train()

    x = torch.randn(2, 4, 64)

    r(x)

    r.reset_state()

    assert not r._usage_pending

    assert r.bias.abs().sum().item() == 0.0


def test_deepseek_router_get_state():
    r = _make_deepseek_router()

    state = r.get_state()

    assert "bias" in state

    assert "mean_bias" in state


def test_deepseek_router_compute_aux_loss():
    r = _make_deepseek_router()

    loss = r.compute_aux_loss()

    assert loss.item() == 0.0


def test_deepseek_router_record_usage_false():
    r = _make_deepseek_router()

    r.train()

    x = torch.randn(2, 4, 64)

    r(x, record_usage=False)

    assert not r._usage_pending


def test_deepseek_router_sync_usage_not_distributed():
    r = _make_deepseek_router()

    r._sync_usage_distributed()


def _make_metabolic_router(hidden=64, num_experts=4, top_k=2):
    from src.routers.metabolic import MetabolicRouter

    from src.configs.router import MetabolicRouterConfig

    cfg = MetabolicRouterConfig(hidden_dim=hidden, num_experts=num_experts, top_k=top_k)

    return MetabolicRouter(cfg)


def test_metabolic_router_top_k_exceeds_experts():
    from src.routers.metabolic import MetabolicRouter

    from src.configs.router import MetabolicRouterConfig

    with pytest.raises(ValueError, match="top_k"):
        MetabolicRouter(MetabolicRouterConfig(hidden_dim=64, num_experts=4, top_k=5))


def test_metabolic_router_compute_alignment():
    r = _make_metabolic_router()

    x = torch.randn(2, 4, 64)

    alignment = r.compute_alignment(x)

    assert alignment.shape == (2, 4, 4)


def test_metabolic_router_compute_routing_potential_no_lambda():
    r = _make_metabolic_router()

    r.lambda_metabolic = 0.0

    alignment = torch.randn(2, 4, 4)

    potential = r.compute_routing_potential(alignment)

    assert torch.allclose(potential, alignment)


def test_metabolic_router_compute_routing_potential_with_noise():
    r = _make_metabolic_router()

    alignment = torch.randn(2, 4, 4)

    potential = r.compute_routing_potential(alignment, noise_std=0.1)

    assert potential.shape == alignment.shape


def test_metabolic_router_forward_eval():
    r = _make_metabolic_router()

    r.eval()

    x = torch.randn(2, 4, 64)

    weights, indices, metrics = r(x)

    assert weights.shape == (8, 4)

    assert indices is None


def test_metabolic_router_forward_with_temperature():
    r = _make_metabolic_router()

    x = torch.randn(2, 4, 64)

    weights, _, _ = r(x, temperature=0.5)

    assert weights.shape == (8, 4)


def test_metabolic_router_forward_with_noise():
    r = _make_metabolic_router()

    x = torch.randn(2, 4, 64)

    weights, _, _ = r(x, noise_std=0.1)

    assert weights.shape == (8, 4)


def test_metabolic_router_forward_warmup():
    r = _make_metabolic_router()

    r.warmup_steps = 100

    r.train()

    x = torch.randn(2, 4, 64)

    weights, _, _ = r(x)

    assert weights.shape == (8, 4)


def test_metabolic_router_update_fatigue():
    r = _make_metabolic_router()

    usage = torch.ones(4) * 0.5

    r.update_fatigue(usage)

    assert r.fatigue.sum().item() >= 0.0


def test_metabolic_router_state_dict_load():
    r = _make_metabolic_router()

    r.train()

    x = torch.randn(2, 4, 64)

    r(x)

    r.step()

    sd = r.state_dict()

    assert "_metabolic_metadata" in sd

    r2 = _make_metabolic_router()

    r2.load_state_dict(sd)

    assert r2.num_steps.item() == r.num_steps.item()


def test_metabolic_router_load_state_dict_mismatch_warns():
    r = _make_metabolic_router()

    sd = r.state_dict()

    sd["_metabolic_metadata"] = {"num_steps": 5, "lambda_metabolic": 999.0}

    with pytest.warns(UserWarning, match="mismatch"):
        r.load_state_dict(sd)


def test_metabolic_router_load_state_dict_no_metadata():
    r = _make_metabolic_router()

    sd = r.state_dict()

    sd_clean = {k: v for k, v in sd.items() if k != "_metabolic_metadata"}

    r2 = _make_metabolic_router()

    r2.load_state_dict(sd_clean)


def test_metabolic_router_sync_usage_not_distributed():
    r = _make_metabolic_router()

    r._sync_usage_distributed()


def test_metabolic_router_get_state():
    r = _make_metabolic_router()

    state = r.get_state()

    assert "fatigue" in state

    assert "lambda_eff" in state


def test_metabolic_router_compute_aux_loss():
    r = _make_metabolic_router()

    loss = r.compute_aux_loss()

    assert loss.item() == 0.0


def _make_stress_router(hidden=64, num_experts=4, top_k=2):
    from src.routers.stress_corrected import StressCorrectedRouter

    from src.configs.router import StressCorrectedRouterConfig

    cfg = StressCorrectedRouterConfig(
        hidden_dim=hidden, num_experts=num_experts, top_k=top_k
    )

    return StressCorrectedRouter(cfg)


def test_stress_router_top_k_exceeds_experts():
    from src.routers.stress_corrected import StressCorrectedRouter

    from src.configs.router import StressCorrectedRouterConfig

    with pytest.raises(ValueError, match="top_k"):
        StressCorrectedRouter(
            StressCorrectedRouterConfig(hidden_dim=64, num_experts=4, top_k=5)
        )


def test_stress_router_forward_eval():
    r = _make_stress_router()

    r.eval()

    x = torch.randn(2, 4, 64)

    weights, _, metrics = r(x)

    assert weights.shape == (8, 4)


def test_stress_router_forward_train_with_noise():
    r = _make_stress_router()

    r.train()

    r.noise_std = 0.1

    x = torch.randn(2, 4, 64)

    weights, _, _ = r(x)

    assert weights.shape == (8, 4)


def test_stress_router_forward_with_metrics():
    r = _make_stress_router()

    x = torch.randn(2, 4, 64)

    weights, _, metrics = r(x, return_metrics=True)

    assert metrics is not None


def test_stress_router_forward_record_usage_false():
    r = _make_stress_router()

    r.train()

    x = torch.randn(2, 4, 64)

    r(x, record_usage=False)

    assert r._pending_count_n.item() == 0


def test_stress_router_step():
    r = _make_stress_router()

    r.train()

    x = torch.randn(2, 4, 64)

    r(x)

    r.step()

    assert r.num_steps.item() == 1


def test_stress_router_step_lambda_calibration():
    r = _make_stress_router()

    r.lambda_calib_step = 1

    r.train()

    x = torch.randn(2, 4, 64)

    r(x)

    r.step()

    assert r._lambda_init_done


def test_stress_router_step_no_pending():
    r = _make_stress_router()

    r.step()

    assert r.num_steps.item() == 1


def test_stress_router_reset_state():
    r = _make_stress_router()

    r.train()

    x = torch.randn(2, 4, 64)

    r(x)

    r.step()

    r.reset_state()

    assert r.num_steps.item() == 0

    assert not r._lambda_init_done


def test_stress_router_reset_welford():
    r = _make_stress_router()

    r.welford_n.fill_(100.0)

    r.reset_welford()

    assert r.welford_n.sum().item() == 0.0


def test_stress_router_get_state():
    r = _make_stress_router()

    state = r.get_state()

    assert "lambda_val" in state

    assert "ema_load" in state


def test_stress_router_compute_aux_loss():
    r = _make_stress_router()

    loss = r.compute_aux_loss()

    assert loss.item() == 0.0


def test_stress_router_clear_aux_state():
    r = _make_stress_router()

    r.clear_aux_state()


def test_stress_router_current_tau_no_anneal():
    r = _make_stress_router()

    r.tau_anneal_steps = 0

    tau = r._current_tau()

    assert tau == r.temperature


def test_stress_router_current_tau_with_anneal():
    r = _make_stress_router()

    r.tau_anneal_steps = 1000

    r.tau_final = 0.1

    r.num_steps.fill_(500)

    tau = r._current_tau()

    assert r.tau_final < tau < r.temperature


def test_stress_router_current_noise_std_no_anneal():
    r = _make_stress_router()

    r.noise_anneal_steps = 0

    std = r._current_noise_std()

    assert std == r.noise_std


def test_stress_router_current_noise_std_with_anneal():
    r = _make_stress_router()

    r.noise_anneal_steps = 1000

    r.noise_std = 0.1

    r.num_steps.fill_(500)

    std = r._current_noise_std()

    assert 0.0 < std < 0.1


def test_stress_router_calibrate_lambda():
    r = _make_stress_router()

    cos_sims = torch.randn(100, 4)

    r._calibrate_lambda(cos_sims)

    assert r.lambda_initialized.item()

    assert r.lambda_val.item() > 0.0


def test_stress_router_initialize_prototypes():
    r = _make_stress_router()

    activations = torch.randn(100, 64)

    r.initialize_prototypes_from_data(activations, n_iter=5)

    norms = torch.nn.functional.normalize(r.W, dim=-1).norm(dim=-1)

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_stress_router_welford_variance():
    r = _make_stress_router()

    r.welford_n.fill_(10.0)

    r.welford_M2.fill_(5.0)

    var = r._welford_variance()

    assert var.shape == (4,)


def test_stress_router_get_custom_metrics():
    r = _make_stress_router()

    x = torch.randn(2, 4, 64)

    weights, _, _ = r(x)

    metrics = r.get_custom_metrics(None, weights)

    assert "lambda_val" in metrics

    assert "ema_load_mean" in metrics


def test_stress_router_get_custom_metrics_with_indices():
    r = _make_stress_router()

    indices = torch.randint(0, 4, (2, 4, 2))

    weights = torch.randn(8, 4).abs()

    metrics = r.get_custom_metrics(indices, weights)

    assert "eff_E_hard" in metrics


def test_stress_router_sync_not_distributed():
    r = _make_stress_router()

    r._sync_pending_counts_distributed()

    r._sync_ema_load_distributed()

    r._sync_lambda_distributed()


def test_stress_router_step_with_lambda_already_done():
    r = _make_stress_router()

    r._lambda_init_done = True

    r.lambda_initialized.fill_(True)

    r.train()

    x = torch.randn(2, 4, 64)

    r(x)

    r.step()

    assert r.num_steps.item() == 1


def test_stress_router_step_lambda_calib_no_pending_cos():
    r = _make_stress_router()

    r.lambda_calib_step = 1

    r._lambda_init_done = False

    r._pending_cos_sims = []

    r.train()

    x = torch.randn(2, 4, 64)

    r(x, record_usage=False)

    r._pending_cos_sims = []

    r.num_steps.fill_(0)

    r.step()


def test_kmeans_init():
    from src.routers.stress_corrected import _kmeans_init

    activations = torch.randn(50, 64)

    centroids = _kmeans_init(activations, k=4, n_iter=5)

    assert centroids.shape == (4, 64)


def test_kmeans_init_too_few_tokens():
    from src.routers.stress_corrected import _kmeans_init

    activations = torch.randn(3, 64)

    with pytest.raises(ValueError, match="need at least"):
        _kmeans_init(activations, k=4)


def test_stress_router_update_welford():
    r = _make_stress_router()

    r.train()

    x = torch.randn(2, 4, 64)

    r(x)

    assert r.welford_n.sum().item() > 0


def test_stress_router_welford_no_active():
    r = _make_stress_router()

    import torch.nn.functional as F

    x_norm = torch.zeros(2, 4, 64)

    topk_idx = torch.zeros(2, 4, 2, dtype=torch.long)

    W_norm = F.normalize(r.W, dim=-1)

    r._update_welford(x_norm, topk_idx, W_norm)


def test_stress_router_sync_welford_not_distributed():
    r = _make_stress_router()

    r._sync_welford_distributed()


def test_stress_router_read_ema_load():
    r = _make_stress_router()

    load = r._read_ema_load()

    assert load.shape == (4,)


def test_stress_router_forward_with_temperature_override():
    r = _make_stress_router()

    x = torch.randn(2, 4, 64)

    weights, _, _ = r(x, temperature=0.1)

    assert weights.shape == (8, 4)


def test_stress_router_forward_with_noise_override():
    r = _make_stress_router()

    r.train()

    x = torch.randn(2, 4, 64)

    weights, _, _ = r(x, noise_std=0.5)

    assert weights.shape == (8, 4)


def test_metabolic_router_record_usage_accumulate():
    r = _make_metabolic_router()

    r.train()

    x = torch.randn(2, 4, 64)

    r(x)

    r(x)

    assert r._usage_pending

    assert r._pending_tokens.item() > 0


def test_metabolic_router_step_no_pending():
    r = _make_metabolic_router()

    r.step()

    assert r.num_steps.item() == 0


def test_metabolic_router_forward_no_warmup():
    r = _make_metabolic_router()

    r.warmup_steps = 0

    r.train()

    x = torch.randn(2, 4, 64)

    weights, _, _ = r(x)

    assert weights.shape == (8, 4)


def test_deepseek_router_step_overloaded_underloaded():
    r = _make_deepseek_router()

    r.train()

    r._pending_usage_sum = torch.tensor([8.0, 0.0, 0.0, 0.0])

    r._pending_tokens.fill_(8)

    r._usage_pending = True

    r.step()

    assert r.bias[0].item() < 0.0

    assert r.bias[1].item() > 0.0


def test_deepseek_router_record_usage_accumulate():
    r = _make_deepseek_router()

    r.train()

    x = torch.randn(2, 4, 64)

    r(x)

    r(x)

    assert r._pending_tokens.item() > 0


def test_expert_choice_router_nan_guard():
    from src.routers.expert_choice import ExpertChoiceRouter

    from src.configs.router import ExpertChoiceRouterConfig

    cfg = ExpertChoiceRouterConfig(hidden_dim=4, num_experts=2, top_k=1)

    router = ExpertChoiceRouter(cfg)

    with torch.no_grad():
        router.gate.weight.fill_(float("nan"))

    x = torch.randn(2, 4, 4)

    weights, _, _ = router(x)

    assert torch.isfinite(weights).all()


def test_expert_choice_router_compute_aux_loss():
    from src.routers.expert_choice import ExpertChoiceRouter

    from src.configs.router import ExpertChoiceRouterConfig

    cfg = ExpertChoiceRouterConfig(hidden_dim=4, num_experts=2, top_k=1)

    router = ExpertChoiceRouter(cfg)

    loss = router.compute_aux_loss()

    assert loss.item() == 0.0


def test_standard_router_clear_aux_state_explicit():
    from src.routers.standard import StandardRouter

    from src.configs.router import StandardRouterConfig

    cfg = StandardRouterConfig(hidden_dim=64, num_experts=4, top_k=2, use_aux_loss=True)

    router = StandardRouter(cfg)

    router.train()

    x = torch.randn(2, 4, 64)

    router(x)

    assert router._last_probs is not None

    router.clear_aux_state()

    assert router._last_probs is None

    assert router._last_indices is None

    assert router._last_weights is None


def test_stress_router_get_custom_metrics_none_both():
    from src.routers.stress_corrected import StressCorrectedRouter

    from src.configs.router import StressCorrectedRouterConfig

    cfg = StressCorrectedRouterConfig(hidden_dim=64, num_experts=4, top_k=2)

    router = StressCorrectedRouter(cfg)

    metrics = router.get_custom_metrics(indices=None, weights=None)

    assert "eff_E_hard" in metrics


def test_stress_router_update_welford_no_active():
    from src.routers.stress_corrected import StressCorrectedRouter

    from src.configs.router import StressCorrectedRouterConfig

    cfg = StressCorrectedRouterConfig(hidden_dim=4, num_experts=2, top_k=1)

    router = StressCorrectedRouter(cfg)

    x_norm = torch.randn(1, 1, 4)

    topk_idx = torch.zeros(1, 1, 1, dtype=torch.long)

    W_norm = torch.randn(2, 4)

    router._update_welford(x_norm, topk_idx, W_norm)

    assert router.welford_n[0].item() >= 0.0
