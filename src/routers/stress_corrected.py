# =============================================================================
# TODO: GMR (Global Mean Projection) — v8b experiment
# =============================================================================
#
# Motivation:
#   On fineweb-edu, gini stays ~0.05-0.10 at all stages (vs 0.388 on wikitext).
#   Root cause: diverse corpus → near-uniform token distribution in hidden space
#   → cosine similarities between x and all W_i are ~equal → SPAR enforces
#   perfect balance (eff_E→8) with no semantic differentiation.
#   Fix: route in corpus-residual space instead of raw hidden space.
#
# Formulation:
#   x_proj = x - (x · v_global) * v_global      # project out global mean dir
#   v_global updated via EMA of batch means (no hyperparameter, no grad)
#   All cosine sims computed on x_proj instead of x
#
# Files to change:
#   src/routers/stress_corrected.py   ← here
#   src/configs/router.py             ← add gmr_enabled: bool = False, gmr_beta: float = 0.999
#   experiments/gptneo_125m_stress_v8b-fineweb.yaml  ← add gmr_enabled: true
#
# Implementation steps:
#
#   1. StressCorrectedRouterConfig: add two fields
#        gmr_enabled: bool = False
#        gmr_beta: float = 0.999     # EMA decay for v_global (no hyperparameter exposure needed)
#
#   2. StressCorrectedRouter.__init__: register v_global buffer
#        self.register_buffer('v_global', torch.zeros(config.hidden_dim, dtype=torch.float32))
#        self.register_buffer('v_global_initialized', torch.tensor(False))
#
#   3. Add _update_v_global(x_flat: Tensor) -> None
#        with torch.no_grad():
#            batch_mean = x_flat.mean(dim=0).float()
#            batch_mean_norm = F.normalize(batch_mean, dim=-1)
#            if not self.v_global_initialized:
#                self.v_global.copy_(batch_mean_norm)
#                self.v_global_initialized.fill_(True)
#            else:
#                self.v_global.mul_(self.gmr_beta).add_(batch_mean_norm * (1 - self.gmr_beta))
#                self.v_global.copy_(F.normalize(self.v_global, dim=-1))
#
#   4. Add _sync_v_global_distributed() — call in step() after EMA update
#        dist.all_reduce(self.v_global, op=dist.ReduceOp.AVG)
#        self.v_global.copy_(F.normalize(self.v_global, dim=-1))
#
#   5. In forward(): project x before F.normalize
#        if self.config.gmr_enabled:
#            self._update_v_global(x_flat)
#            proj = (x_flat.float() @ self.v_global.unsqueeze(-1)).squeeze(-1)
#            x_for_routing = x_flat - proj.unsqueeze(-1) * self.v_global
#        else:
#            x_for_routing = x_flat
#        x_norm = F.normalize(x_for_routing.to(self.W.dtype), dim=-1)
#        # output weights still use original x (not x_proj) so gradient flows correctly
#
#   6. In initialize_prototypes_from_data(): apply same projection before k-means
#        so prototypes are initialized in the same residual space
#
# Key properties:
#   - v_global is updated out-of-graph (no_grad) → zero new hyperparameters
#   - At init, v_global=0 → projection is identity → bit-identical to v7/v8a
#   - DDP: sync v_global via all_reduce(AVG) + renormalize in step()
#   - Output weights w_i computed from original x (not x_proj) → gradient
#     flows through softmax to W_i correctly
#
# Run after v8a completes. Config: gptneo_125m_stress_v8b-fineweb.yaml
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional

from src.configs import StressCorrectedRouterConfig
from src.core import RouterRegistry
from src.routers.base import BaseRouter
from src.metrics import RouterMetricsTracker
from src.project_types import RouterType

MIN_TEMPERATURE = 1e-3


def _kmeans_init(activations: torch.Tensor, k: int, n_iter: int = 20) -> torch.Tensor:
    """K-means on unit-normalized activations. Returns [k, D] centroids (not normalized)."""
    N, D = activations.shape
    if N < k:
        raise ValueError(f"_kmeans_init: need at least k={k} tokens, got N={N}")
    idx = torch.randperm(N, device=activations.device)[:k]
    centroids = activations[idx].clone()
    for _ in range(n_iter):
        norms_x = F.normalize(activations, dim=-1)
        norms_c = F.normalize(centroids, dim=-1)
        assignments = (norms_x @ norms_c.T).argmax(dim=-1)  # [N]
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=activations.device)
        new_centroids.scatter_add_(
            0, assignments.unsqueeze(-1).expand(-1, D), activations
        )
        counts.scatter_add_(0, assignments, torch.ones(N, device=activations.device))
        mask = counts > 0
        new_centroids[mask] /= counts[mask].unsqueeze(-1)
        new_centroids[~mask] = centroids[~mask]  # keep old centroid for empty clusters
        centroids = new_centroids
    return centroids


@RouterRegistry.register(RouterType.STRESS_CORRECTED.value)
class StressCorrectedRouter(BaseRouter):
    r"""
    SPAR Router: one-sided adaptive load penalty, no auxiliary loss.

    Selection logit:
        z_i(x,t) = cos(x, W_i) - λ · max(0, L_i(t) - 1/N)

        cos(x, W_i)          — cosine similarity: directional, scale-invariant
        max(0, L_i - 1/N)    — one-sided penalty: zero at equilibrium, positive
                               only for overloaded experts. Zero-sum differential:
                               Σ_i max(0, L_i - 1/N) concentrates on overloaded
                               experts only, leaving underloaded ones unpenalised.
        λ                    — auto-calibrated once at step lambda_calib_step
                               (default: warmup_steps + 200, post-LR-warmup):
                               λ = min(σ_cos / mean(L), 5.0)
                               A 1-sigma routing variation equals a 1× fair-share
                               penalty. No manual tuning required.

    Output weight:
        w_i = softmax(cos(x, W_i) / τ_t)   over top-k selected experts only

        τ_t anneals linearly: temperature → tau_final over tau_anneal_steps optimizer steps.
        Selection and weighting are factored: selection carries the load signal;
        weighting reflects alignment quality only. τ < 1 sharpens the distribution.

    Load update (EMA, after each optimizer step):
        L_i(t) = (1-α) · L_i(t-1) + α · U_i(t)
        U_i(t) = fraction of token-expert assignments routed to expert i.
        Synced across DDP ranks every step via all_reduce(AVG).

    Welford statistics (tracked for metrics only, not used in logit):
        Per-expert mean/variance of cosine distances — exposes alignment quality
        in WandB without affecting routing.
    """

    def __init__(self, config: StressCorrectedRouterConfig):
        super().__init__(config)

        if config.top_k > config.num_experts:
            raise ValueError(
                f"top_k ({config.top_k}) cannot exceed num_experts ({config.num_experts})"
            )

        # NOTE: eps was originally used for the Stress CV denominator (removed
        # in SPAR clean — mu_stress=0).  F.normalize uses PyTorch's default
        # eps=1e-12.  Kept for config compatibility.
        self.eps = config.eps
        self.temperature = config.temperature
        self.noise_std = config.noise_std
        self.ema_alpha = config.ema_alpha
        self.lambda_calib_step = config.lambda_calib_step
        self.tau_final = config.tau_final
        self.tau_anneal_steps = config.tau_anneal_steps
        self.noise_anneal_steps = config.noise_anneal_steps

        # Prototype directions — normalized in forward().
        self.W = nn.Parameter(
            F.normalize(torch.randn(config.num_experts, config.hidden_dim), dim=-1)
        )

        # EMA load — fraction of assignments per expert, sums to 1.
        # Initialized to fair share; updated each optimizer step.
        self.register_buffer(
            "ema_load", torch.ones(config.num_experts) / config.num_experts
        )

        # Lambda — calibrated once at lambda_calib_step, then fixed.
        self.register_buffer("lambda_val", torch.tensor(1.0))
        self.register_buffer("lambda_initialized", torch.tensor(False))
        self._lambda_init_done: bool = False  # compile-safe Python mirror

        self.register_buffer("num_steps", torch.tensor(0, dtype=torch.long))
        self._tau: float = (
            config.temperature
        )  # compile-safe Python float, updated in step()
        self._noise_std: float = (
            config.noise_std
        )  # compile-safe Python float, updated in step()

        # Pending count accumulator — raw assignment fractions summed across all
        # forward passes in a gradient-accumulation window, applied as a single
        # EMA step in step() to avoid multi-forward EMA compounding.
        self.register_buffer("_pending_counts", torch.zeros(config.num_experts))
        self.register_buffer("_pending_count_n", torch.tensor(0, dtype=torch.long))

        # Cosine accumulator for λ calibration — collects cos_sim tensors from
        # all forwards in the current grad-accum window so calibration uses the
        # full optimizer step's data (not just the last microbatch).
        self._pending_cos_sims: list = []

        # Welford state — metrics only, not used in routing logit.
        self.register_buffer(
            "welford_n", torch.zeros(config.num_experts, dtype=torch.float32)
        )
        self.register_buffer(
            "welford_mu", torch.zeros(config.num_experts, dtype=torch.float32)
        )
        self.register_buffer(
            "welford_M2", torch.zeros(config.num_experts, dtype=torch.float32)
        )

        self.metrics_tracker = RouterMetricsTracker(self)

    @torch.no_grad()
    def initialize_prototypes_from_data(
        self, activations: torch.Tensor, n_iter: int = 30
    ) -> None:
        """Initialize W from k-means centroids of actual layer activations.
        Call once before training begins, after collecting a representative batch.
        activations: [N_tokens, hidden_dim] — raw hidden states at this MoE layer.
        DDP: caller is responsible for broadcasting the result across ranks.
        """
        centroids = _kmeans_init(activations.float(), self.num_experts, n_iter)
        self.W.data.copy_(F.normalize(centroids, dim=-1).to(self.W.dtype))

    @torch.no_grad()
    def _update_welford(
        self, x_norm: torch.Tensor, topk_idx: torch.Tensor, W_norm: torch.Tensor
    ) -> None:
        B, S, D = x_norm.shape
        x_flat = x_norm.reshape(-1, D)  # [BS, D]
        alignments = x_flat @ W_norm.T  # [BS, E]
        distances = 1.0 - alignments  # [BS, E], ∈ [0, 2]

        # Binary mask: 1 if expert was selected for this token, 0 otherwise.
        mask = torch.zeros(B * S, self.num_experts, device=x_flat.device)
        mask.scatter_(1, topk_idx.reshape(-1, self.top_k), 1.0)  # [BS, E]

        # Vectorized weighted Welford over all experts simultaneously.
        # w_sum[E]: total weight routed to each expert this batch.
        w_sum = mask.sum(dim=0)  # [E]
        active = w_sum > 1e-8  # [E] bool

        if not active.any():
            return

        # Work only in fp32 for numerical stability.
        w = mask.float()  # [BS, E]
        d = distances.float()  # [BS, E]

        # Welford update:  n_new = n + w_sum
        #   mu_new = mu + sum(w * (d - mu)) / n_new
        #   M2_new = M2 + sum(w * delta_pre * delta_post)
        n_new = self.welford_n + w_sum  # [E]
        delta_pre = d - self.welford_mu  # [BS, E]  broadcast mu over tokens
        mu_update = (w * delta_pre).sum(dim=0) / n_new.clamp(min=1e-8)  # [E]
        new_mu = self.welford_mu + mu_update
        delta_post = d - new_mu  # [BS, E]
        m2_update = (w * delta_pre * delta_post).sum(dim=0)  # [E]

        # Apply updates only for active experts (those with tokens this batch).
        self.welford_n = torch.where(active, n_new, self.welford_n)
        self.welford_mu = torch.where(active, new_mu, self.welford_mu)
        self.welford_M2 = torch.where(
            active, self.welford_M2 + m2_update, self.welford_M2
        )

    def _welford_variance(self) -> torch.Tensor:
        return self.welford_M2 / self.welford_n.clamp(min=1.0)

    def _current_tau(self) -> float:
        if self.tau_anneal_steps <= 0 or self.tau_final >= self.temperature:
            return self.temperature
        frac = min(self.num_steps.item() / self.tau_anneal_steps, 1.0)
        return self.temperature + frac * (self.tau_final - self.temperature)

    def _current_noise_std(self) -> float:
        if self.noise_anneal_steps <= 0:
            return self.noise_std
        frac = min(self.num_steps.item() / self.noise_anneal_steps, 1.0)
        return self.noise_std * (1.0 - frac)

    @torch.no_grad()
    def _calibrate_lambda(self, cos_sim_flat: torch.Tensor) -> None:
        """λ = min(σ_cos / mean(L), 5.0). Calibrates once at lambda_calib_step.

        Floor: σ_cos is clamped to ≥ 1e-4 to prevent λ=0 in degenerate cases
        (e.g., all cosines identical after random init in high-D). A zero λ
        would permanently disable the load penalty with no recovery path.
        """
        sigma_cos = cos_sim_flat.std().clamp(min=1e-4)
        self.lambda_val.fill_((sigma_cos * self.num_experts).clamp(max=5.0).item())
        self.lambda_initialized.fill_(True)

    @torch._dynamo.disable
    def _read_ema_load(self) -> torch.Tensor:
        """Clone ema_load outside the compiled graph to avoid AOT version-mismatch."""
        return self.ema_load.clone()

    @torch._dynamo.disable
    def _update_load_and_welford(
        self,
        topk_idx: torch.Tensor,
        x_norm: torch.Tensor,
        W_norm: torch.Tensor,
        cos_sim: torch.Tensor,
        B: int,
        S: int,
    ) -> None:
        """Update EMA load, Welford stats, and cosine accumulators.
        Disabled from Dynamo to prevent version conflicts."""
        with torch.no_grad():
            counts = torch.zeros(self.num_experts, device=topk_idx.device)
            counts.scatter_add_(
                0,
                topk_idx.flatten().clamp(min=0),
                torch.ones(topk_idx.numel(), device=topk_idx.device),
            )
            counts.div_(counts.sum().clamp(min=1e-8))
            self._pending_counts.add_(counts)
            self._pending_count_n.add_(1)
            self._update_welford(x_norm, topk_idx, W_norm)

            # Accumulate cosines for λ calibration (pre-calibration only).
            if not self._lambda_init_done:
                self._pending_cos_sims.append(cos_sim.reshape(-1, self.num_experts))

    def forward(
        self,
        x: torch.Tensor,
        return_metrics: bool = False,
        record_usage: bool = True,
        noise_std: Optional[float] = None,
        temperature: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        B, S, D = x.shape
        x_norm = F.normalize(x, dim=-1)
        W_norm = F.normalize(self.W, dim=-1).to(x.dtype)

        cos_sim = x_norm @ W_norm.T  # [B, S, E]

        # Gumbel exploration noise
        noise = 0.0
        used_noise = noise_std if noise_std is not None else self._noise_std
        if self.training and used_noise > 0:
            u = torch.empty_like(cos_sim).uniform_(1e-10, 1.0 - 1e-10)
            noise = used_noise * (-torch.log(-torch.log(u)))

        # SPAR selection logit: cosine - λ·max(0, L_i - 1/N)
        ema_load = self._read_ema_load()
        fair_share = 1.0 / self.num_experts
        logits = (
            cos_sim - self.lambda_val * (ema_load - fair_share).clamp(min=0.0) + noise
        )

        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)  # [B, S, k]

        # Output weights: cosine-only softmax (factored from selection)
        tau = max(
            temperature if temperature is not None else self._tau, MIN_TEMPERATURE
        )
        topk_cos = cos_sim.gather(-1, topk_idx)
        output_weights = F.softmax(topk_cos / tau, dim=-1)  # [B, S, k]

        if self.training and record_usage:
            # Detach all tensors at the call site so the Dynamo-resumed graph
            # after the @disable boundary always sees requires_grad=False.
            # All three args are metrics-only inside _update_load_and_welford;
            # no gradient is needed or used there.
            self._update_load_and_welford(
                topk_idx.detach(),
                x_norm.detach(),
                W_norm.detach(),
                cos_sim.detach(),
                B,
                S,
            )

        metrics = None
        if return_metrics:
            metrics = self.metrics_tracker.compute_all_metrics(topk_idx, output_weights)

        return output_weights, topk_idx, metrics

    def step(self) -> None:
        # Sync Python bool from buffer (safe after checkpoint loads).
        if self.lambda_initialized.item():
            self._lambda_init_done = True

        self.num_steps += 1
        self._tau = self._current_tau()
        self._noise_std = self._current_noise_std()

        # Apply one EMA step from counts accumulated across all forward passes
        # in this gradient-accumulation window. This ensures α=0.01 per optimizer
        # step regardless of how many layers × microbatches called forward().
        n = self._pending_count_n.item()
        if n > 0:
            with torch.no_grad():
                avg_counts = self._pending_counts / n
                self.ema_load.mul_(1 - self.ema_alpha).add_(avg_counts * self.ema_alpha)
                self._pending_counts.zero_()
                self._pending_count_n.zero_()

        if (
            self.num_steps.item() == self.lambda_calib_step
            and not self._lambda_init_done
        ):
            if self._pending_cos_sims:
                pending = torch.cat(self._pending_cos_sims, dim=0)
                self._calibrate_lambda(pending)
            self._lambda_init_done = True
            # ALL ranks must call this together — no branching.
            # If pending was empty on this rank, lambda stays at 1.0 and participates
            # in the AVG so other ranks' calibrated value is preserved.
            self._sync_lambda_distributed()

        # Clear cosine accumulator each step — bounds memory to one grad-accum window.
        self._pending_cos_sims.clear()

        # EMA load sync — critical for routing correctness across DDP ranks.
        self._sync_ema_load_distributed()
        # Welford is metrics-only: per-rank divergence is acceptable.
        # Removed from step() to avoid 18 all_gather ops/step and the seqnum
        # drift that caused the NCCL deadlock.

    @torch.no_grad()
    def _sync_ema_load_distributed(self) -> None:
        try:
            import torch.distributed as dist
        except ImportError:
            return
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        dist.all_reduce(self.ema_load, op=dist.ReduceOp.AVG)

    @torch.no_grad()
    def _sync_lambda_distributed(self) -> None:
        try:
            import torch.distributed as dist
        except ImportError:
            return
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        dist.all_reduce(self.lambda_val, op=dist.ReduceOp.AVG)

    @torch.no_grad()
    def _sync_welford_distributed(self) -> None:
        """Parallel Welford all-reduce (Chan et al. 1979) for cross-rank metrics consistency.

        NOT called in step() — doing so caused an NCCL sequence-number deadlock because
        all_gather requires all ranks to call it in the same order, but MoE layers are
        not always visited by all ranks at the same time under gradient accumulation.

        Call at eval time only (all ranks participate simultaneously):
            for moe_layer in moe_layers.values():
                moe_layer.router._sync_welford_distributed()
        """
        try:
            import torch.distributed as dist
        except ImportError:
            return
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return

        world_size = dist.get_world_size()
        n_list = [torch.zeros_like(self.welford_n) for _ in range(world_size)]
        mu_list = [torch.zeros_like(self.welford_mu) for _ in range(world_size)]
        M2_list = [torch.zeros_like(self.welford_M2) for _ in range(world_size)]

        dist.all_gather(n_list, self.welford_n)
        dist.all_gather(mu_list, self.welford_mu)
        dist.all_gather(M2_list, self.welford_M2)

        # Vectorized parallel Welford combine (Chan et al. 1979).
        # Stack to [world_size, E] so all ranks are reduced in one pass.
        ns = torch.stack(n_list, dim=0)  # [W, E]
        mus = torch.stack(mu_list, dim=0)  # [W, E]
        M2s = torch.stack(M2_list, dim=0)  # [W, E]

        combined_n = ns[0].clone()
        combined_mu = mus[0].clone()
        combined_M2 = M2s[0].clone()

        # Left-fold across ranks — sequential dependency requires iterating,
        # but all per-expert ops are now vectorized across the E dimension.
        for i in range(1, world_size):
            n_b = ns[i]
            mu_b = mus[i]
            M2_b = M2s[i]
            n_new = combined_n + n_b
            delta = mu_b - combined_mu
            safe_n = n_new.clamp(min=1.0)
            combined_mu = (combined_n * combined_mu + n_b * mu_b) / safe_n
            combined_M2 = combined_M2 + M2_b + delta.pow(2) * combined_n * n_b / safe_n
            combined_n = n_new

        self.welford_n.copy_(combined_n)
        self.welford_mu.copy_(combined_mu)
        self.welford_M2.copy_(combined_M2)

    def get_custom_metrics(
        self, indices: torch.Tensor, weights: torch.Tensor
    ) -> Dict[str, Any]:
        metrics = {}

        # Welford alignment quality (metrics only — not in routing logit)
        var = self._welford_variance()
        metrics["welford_mu_mean"] = self.welford_mu.mean().item()
        metrics["welford_var_mean"] = var.mean().item()
        metrics["welford_n_min"] = self.welford_n.min().item()

        # Load signal
        metrics["ema_load_mean"] = self.ema_load.mean().item()
        metrics["ema_load_max"] = self.ema_load.max().item()
        metrics["ema_load_std"] = self.ema_load.std().item()
        metrics["lambda_val"] = self.lambda_val.item()
        metrics["tau"] = self._current_tau()
        metrics["noise_std"] = self._noise_std

        # Per-expert
        metrics["ema_load_per_expert"] = self.ema_load.cpu().float().numpy().tolist()

        # Hard assignment counts
        hard = torch.zeros(self.num_experts, device=indices.device, dtype=torch.float32)
        hard.scatter_add_(
            0,
            indices.flatten().clamp(min=0),
            torch.ones(indices.numel(), device=indices.device),
        )
        hard = hard / hard.sum().clamp(min=1e-8)
        metrics["eff_E_hard"] = (1.0 / (hard**2).sum().clamp(min=1e-8)).item()

        return metrics

    def compute_aux_loss(self) -> torch.Tensor:
        """Always zero — one-sided load penalty IS the balancing mechanism."""
        return torch.tensor(0.0, device=self.W.device)

    def clear_aux_state(self) -> None:
        pass

    def reset_state(self) -> None:
        with torch.no_grad():
            self.ema_load.fill_(1.0 / self.num_experts)
            self.lambda_val.fill_(1.0)
            self.lambda_initialized.fill_(False)
            self.welford_n.zero_()
            self.welford_mu.zero_()
            self.welford_M2.zero_()
            self.num_steps.zero_()
            self._pending_counts.zero_()
            self._pending_count_n.zero_()
        self._lambda_init_done = False
        self._tau = self.temperature
        self._noise_std = self.noise_std
        self._pending_cos_sims = []

    def reset_welford(self) -> None:
        """Reset Welford accumulators.

        Call periodically (e.g., each eval interval) to prevent fp32 precision
        degradation when welford_n grows large (~10^8 after 19k steps at
        batch=32, seq=1024).  Safe because Welford is metrics-only and does
        not affect routing.
        """
        with torch.no_grad():
            self.welford_n.zero_()
            self.welford_mu.zero_()
            self.welford_M2.zero_()

    def get_state(self) -> Dict[str, Any]:
        return {
            "num_steps": self.num_steps.item(),
            "lambda_val": self.lambda_val.item(),
            "ema_load": self.ema_load.clone(),
            "ema_load_std": self.ema_load.std().item(),
            "welford_n": self.welford_n.clone(),
            "welford_mu": self.welford_mu.clone(),
        }
