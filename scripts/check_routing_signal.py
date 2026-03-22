"""
Diagnostic: does routing on base_out = frozen_MLP(x) give a stronger signal than routing on x?

Metric: ratio = mean_token(std_expert(cos(base_out, W))) / mean_token(std_expert(cos(x, W)))
Threshold: ratio > 1.2 → recommend switching to base_out routing.

Runs without a checkpoint (random init is fine — we're comparing discriminability of
x vs base_out against the same W, not routing quality per se).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers.activations import NewGELUActivation  # noqa: E402

from src.routers.stress_corrected import StressCorrectedRouter  # noqa: E402
from src.configs import StressCorrectedRouterConfig  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- config matching v7 -------------------------------------------------------
HIDDEN_DIM = 768  # GPT-Neo 125M
INTERMEDIATE_DIM = 3072  # 4 × 768
NUM_EXPERTS = 8
TOP_K = 2
MOE_LAYER_INDICES = [1, 3, 5, 7, 9, 11]
NUM_LAYERS = len(MOE_LAYER_INDICES)

BATCH = 4
SEQ_LEN = 64  # short — we only need the signal ratio, not real token stats


def frozen_mlp_forward(
    x: torch.Tensor,
    fc_w: torch.Tensor,
    fc_b: torch.Tensor,
    proj_w: torch.Tensor,
    proj_b: torch.Tensor,
) -> torch.Tensor:
    """GPT-Neo frozen MLP: GELU activation."""
    act = NewGELUActivation()
    h = F.linear(x, fc_w, fc_b)
    h = act(h)
    return F.linear(h, proj_w, proj_b)


def routing_spread(x_flat: torch.Tensor, W_norm: torch.Tensor) -> float:
    """
    x_flat: [T, D] normalized tokens
    W_norm: [N, D] normalized prototypes
    Returns mean over tokens of std over experts of cosine scores.
    """
    cos = x_flat @ W_norm.T  # [T, N]
    return cos.std(dim=-1).mean().item()


@torch.no_grad()
def run_diagnostic():
    results = []

    for layer_num in range(NUM_LAYERS):
        # Build a fresh router (random W)
        router_cfg = StressCorrectedRouterConfig(
            hidden_dim=HIDDEN_DIM,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            temperature=0.5,
            tau_final=0.12,
            tau_anneal_steps=10000,
            noise_std=0.0,
            noise_anneal_steps=0,
            eps=1e-3,
            ema_alpha=0.01,
            lambda_calib_step=600,
        )
        router = StressCorrectedRouter(router_cfg).to(DEVICE).eval()
        W_norm = F.normalize(router.W, dim=-1)  # [N, D]

        # Build one GPTNeoLoRAMLP to get realistic frozen weights
        # We use random weights (no pretrained checkpoint needed for this diagnostic).
        # Create random frozen weights matching GPT-Neo MLP dimensions
        fc_w = torch.randn(INTERMEDIATE_DIM, HIDDEN_DIM, device=DEVICE) * 0.02
        fc_b = torch.zeros(INTERMEDIATE_DIM, device=DEVICE)
        proj_w = torch.randn(HIDDEN_DIM, INTERMEDIATE_DIM, device=DEVICE) * 0.02
        proj_b = torch.zeros(HIDDEN_DIM, device=DEVICE)

        # Synthetic input: simulate residual stream activations
        # Use layer-norm-like scale: hidden states are roughly unit-norm in practice
        x = torch.randn(BATCH, SEQ_LEN, HIDDEN_DIM, device=DEVICE)
        x = F.layer_norm(x, [HIDDEN_DIM])  # mimic post-LN distribution

        x_flat = x.reshape(-1, HIDDEN_DIM)  # [T, D]

        # base_out = frozen_MLP(x)
        base_out_flat = frozen_mlp_forward(x_flat, fc_w, fc_b, proj_w, proj_b)

        # Normalize for cosine similarity
        x_norm = F.normalize(x_flat, dim=-1)
        base_out_norm = F.normalize(base_out_flat, dim=-1)

        std_x = routing_spread(x_norm, W_norm)
        std_base = routing_spread(base_out_norm, W_norm)
        ratio = std_base / (std_x + 1e-12)

        results.append((layer_num, std_x, std_base, ratio))

    # Print table
    print(f"\n{'Layer':<8}{'std_x':<12}{'std_base':<12}{'ratio':<8}")
    print("-" * 40)
    for layer_num, std_x, std_base, ratio in results:
        idx = MOE_LAYER_INDICES[layer_num]
        print(f"L{idx:<7}{std_x:<12.4f}{std_base:<12.4f}{ratio:<8.3f}")

    mean_std_x = sum(r[1] for r in results) / NUM_LAYERS
    mean_std_base = sum(r[2] for r in results) / NUM_LAYERS
    mean_ratio = sum(r[3] for r in results) / NUM_LAYERS

    print("-" * 40)
    print(f"{'Mean':<8}{mean_std_x:<12.4f}{mean_std_base:<12.4f}{mean_ratio:<8.3f}")

    threshold = 1.2
    verdict = "IS" if mean_ratio > threshold else "IS NOT"
    print(
        f"\nVerdict: ratio = {mean_ratio:.3f} → route on base_out {verdict} recommended (threshold: {threshold})"
    )

    return mean_ratio


if __name__ == "__main__":
    run_diagnostic()
