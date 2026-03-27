import torch


def test_frozen_mlp_forward():

    from scripts.check_routing_signal import frozen_mlp_forward

    x = torch.randn(4, 768)

    fc_w = torch.randn(3072, 768)

    fc_b = torch.zeros(3072)

    proj_w = torch.randn(768, 3072)

    proj_b = torch.zeros(768)

    out = frozen_mlp_forward(x, fc_w, fc_b, proj_w, proj_b)

    assert out.shape == (4, 768)


def test_routing_spread():

    from scripts.check_routing_signal import routing_spread

    import torch.nn.functional as F

    x_flat = F.normalize(torch.randn(16, 768), dim=-1)

    W_norm = F.normalize(torch.randn(8, 768), dim=-1)

    spread = routing_spread(x_flat, W_norm)

    assert isinstance(spread, float)

    assert spread >= 0.0


def test_run_diagnostic():

    from scripts.check_routing_signal import run_diagnostic

    ratio = run_diagnostic()

    assert isinstance(ratio, float)

    assert ratio > 0.0
