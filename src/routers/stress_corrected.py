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
    """K-means++ on unit-normalized activations. Returns [k, D] centroids (unnormalized)."""
    N, D = activations.shape
    if N < k:
        raise ValueError(f"_kmeans_init: need at least k={k} tokens, got N={N}")
    idx = torch.randperm(N, device=activations.device)[:k]
    centroids = activations[idx].clone()
    for _ in range(n_iter):
        norms_x = F.normalize(activations, dim=-1)
        norms_c = F.normalize(centroids, dim=-1)
        assignments = (norms_x @ norms_c.T).argmax(dim=-1)
        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=activations.device)
        new_centroids.scatter_add_(
            0, assignments.unsqueeze(-1).expand(-1, D), activations
        )
        counts.scatter_add_(0, assignments, torch.ones(N, device=activations.device))
        mask = counts > 0
        new_centroids[mask] /= counts[mask].unsqueeze(-1)
        new_centroids[~mask] = centroids[~mask]
        centroids = new_centroids
    return centroids


@RouterRegistry.register(RouterType.STRESS_CORRECTED.value)
class StressCorrectedRouter(BaseRouter):
    r"""
    SPAR Router — symmetric load-corrected cosine routing, no auxiliary loss.

    Selection:   z_i(x) = cos(x, W_i) - λ·(L_i - 1/N)
    Output wt:   w_i = softmax(cos(x, W_i) / τ)  over top-k selected experts
    Lambda:      λ = min(σ_cos · N, 5.0), calibrated once post-warmup
                 σ_cos = within-token inter-expert std (averaged across tokens)
    Load EMA:    L_i(t) = (1-α)·L_i(t-1) + α·U_i(t)
                 U_i = soft output weight fraction (not hard count) — at τ=0.1
                 the dominant expert gets ~2× its selection-frequency share, so
                 soft tracking prevents penalty under-correction.
    """

    def __init__(self, config: StressCorrectedRouterConfig):
        super().__init__(config)

        if config.top_k > config.num_experts:
            raise ValueError(
                f"top_k ({config.top_k}) cannot exceed num_experts ({config.num_experts})"
            )

        self.eps = config.eps
        self.temperature = config.temperature
        self.noise_std = config.noise_std
        self.ema_alpha = config.ema_alpha
        self.lambda_calib_step = config.lambda_calib_step
        self.tau_final = config.tau_final
        self.tau_anneal_steps = config.tau_anneal_steps
        self.noise_anneal_steps = config.noise_anneal_steps

        self.W = nn.Parameter(
            F.normalize(torch.randn(config.num_experts, config.hidden_dim), dim=-1)
        )
        self.register_buffer(
            "ema_load", torch.ones(config.num_experts) / config.num_experts
        )
        self.register_buffer("lambda_val", torch.tensor(config.lambda_init))
        self.register_buffer("lambda_initialized", torch.tensor(False))
        self.register_buffer("sigma_cos_at_calib", torch.tensor(0.0))
        self.register_buffer("num_steps", torch.tensor(0, dtype=torch.long))

        # Python mirrors of buffers — needed for torch.compile compatibility.
        self._lambda_init_done: bool = False
        self._tau: float = config.temperature
        self._noise_std: float = config.noise_std

        # Pending accumulators: counts and cos_sims are summed across all microbatches
        # in a grad-accum window, then applied as a single EMA step in step().
        # This prevents α from compounding across microbatches.
        self.register_buffer("_pending_counts", torch.zeros(config.num_experts))
        self.register_buffer("_pending_count_n", torch.tensor(0, dtype=torch.long))
        self._pending_cos_sims: list = []

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
        """Init W from k-means centroids of layer activations. Call before training.
        activations: [N_tokens, hidden_dim]. DDP: caller broadcasts across ranks.
        """
        centroids = _kmeans_init(activations.float(), self.num_experts, n_iter)
        self.W.data.copy_(F.normalize(centroids, dim=-1).to(self.W.dtype))

    @torch.no_grad()
    def _update_welford(self, cos_sim: torch.Tensor, topk_idx: torch.Tensor) -> None:
        """Online Welford update using only the top-k selected (token, expert) pairs.
        Operates on cos_sim already computed in forward() — no extra matmul.
        Uses scatter_add on [BS*k] pairs instead of a dense [BS, E] mask.
        """
        BS = cos_sim.shape[0] * cos_sim.shape[1]
        flat_idx = topk_idx.reshape(-1)  # [BS*k]
        cos_flat = cos_sim.reshape(BS, self.num_experts)
        sel_dist = (
            (1.0 - cos_flat.gather(1, topk_idx.reshape(BS, self.top_k)))
            .float()
            .reshape(-1)
        )

        w_sum = torch.zeros(
            self.num_experts, device=cos_sim.device, dtype=torch.float32
        )
        w_sum.scatter_add_(0, flat_idx.clamp(min=0), torch.ones_like(sel_dist))
        active = w_sum > 1e-8
        if not active.any():
            return

        mu_sel = self.welford_mu[flat_idx.clamp(min=0)]
        delta_pre = sel_dist - mu_sel

        n_new = self.welford_n + w_sum
        mu_num = torch.zeros(
            self.num_experts, device=cos_sim.device, dtype=torch.float32
        )
        mu_num.scatter_add_(0, flat_idx.clamp(min=0), delta_pre)
        new_mu = self.welford_mu + mu_num / n_new.clamp(min=1e-8)

        new_mu_sel = new_mu[flat_idx.clamp(min=0)]
        m2_update = torch.zeros(
            self.num_experts, device=cos_sim.device, dtype=torch.float32
        )
        m2_update.scatter_add_(
            0, flat_idx.clamp(min=0), delta_pre * (sel_dist - new_mu_sel)
        )

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
        """λ = min(σ_cos · N, 5.0). σ_cos clamped ≥ 1e-4 to prevent λ=0 at random init."""
        sigma_cos = cos_sim_flat.std(dim=-1).mean().clamp(min=1e-4)
        self.lambda_val.fill_((sigma_cos * self.num_experts).clamp(max=5.0).item())
        self.lambda_initialized.fill_(True)
        self.sigma_cos_at_calib.fill_(sigma_cos.item())

    @torch._dynamo.disable
    def _read_ema_load(self) -> torch.Tensor:
        # Clone outside compiled graph to avoid AOT version-mismatch on buffer reads.
        return self.ema_load.clone()

    @torch._dynamo.disable
    def _update_load_and_welford(
        self,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        cos_sim: torch.Tensor,
    ) -> None:
        # @dynamo.disable: this function mutates buffers; keeping it out of the
        # compiled graph avoids version-counter conflicts on _pending_counts.
        with torch.no_grad():
            counts = torch.zeros(self.num_experts, device=topk_idx.device)
            counts.scatter_add_(
                0,
                topk_idx.flatten().clamp(min=0),
                topk_weights.flatten().to(counts.dtype),
            )
            counts.div_(counts.sum().clamp(min=1e-8))
            self._pending_counts.add_(counts)
            self._pending_count_n.add_(1)
            self._update_welford(cos_sim, topk_idx)
            if not self._lambda_init_done:
                self._pending_cos_sims.append(
                    cos_sim.reshape(-1, self.num_experts).cpu()
                )

    @torch._dynamo.disable
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

        noise = 0.0
        used_noise = noise_std if noise_std is not None else self._noise_std
        if self.training and used_noise > 0:
            u = torch.empty_like(cos_sim).uniform_(1e-10, 1.0 - 1e-10)
            noise = used_noise * (-torch.log(-torch.log(u)))

        ema_load = self._read_ema_load()
        logits = cos_sim - self.lambda_val * (ema_load - 1.0 / self.num_experts) + noise
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)  # [B, S, k]

        tau = max(
            temperature if temperature is not None else self._tau, MIN_TEMPERATURE
        )
        topk_cos = cos_sim.gather(-1, topk_idx)
        topk_weights = F.softmax(topk_cos / tau, dim=-1)  # [B, S, k]

        # out-of-place scatter preserves autograd graph through topk_weights → W
        topk_idx_flat = topk_idx.view(-1, self.top_k)
        topk_weights_flat = topk_weights.view(-1, self.top_k)
        expert_weights = torch.zeros(
            B * S,
            self.num_experts,
            dtype=topk_weights_flat.dtype,
            device=topk_weights_flat.device,
        )
        expert_weights = expert_weights.scatter(1, topk_idx_flat, topk_weights_flat)

        if self.training and record_usage:
            self._update_load_and_welford(
                topk_idx.detach(), topk_weights.detach(), cos_sim.detach()
            )

        metrics = None
        if return_metrics:
            metrics = self.metrics_tracker.compute_all_metrics(topk_idx, topk_weights)

        return expert_weights, None, metrics

    def step(self) -> None:
        if self.lambda_initialized.item():
            self._lambda_init_done = True

        self.num_steps += 1
        self._tau = self._current_tau()
        self._noise_std = self._current_noise_std()

        # Sync pending counts across DDP ranks before EMA update so all ranks
        # apply the same globally-pooled load signal. Must be called unconditionally
        # (no n>0 guard) to avoid collective rank mismatch.
        with torch.no_grad():
            self._sync_pending_counts_distributed()
            n_global = self._pending_count_n.item()
            if n_global > 0:
                avg_counts = self._pending_counts / self._pending_count_n.float()
                self.ema_load.mul_(1 - self.ema_alpha).add_(avg_counts * self.ema_alpha)
            self._pending_counts.zero_()
            self._pending_count_n.zero_()

        if (
            self.num_steps.item() >= self.lambda_calib_step
            and not self._lambda_init_done
        ):
            if self._pending_cos_sims:
                pending = torch.cat(self._pending_cos_sims, dim=0).to(self.W.device)
                self._calibrate_lambda(pending)
                self._lambda_init_done = True
                # Broadcast from rank 0 — AVG would corrupt λ if any rank had empty cos_sims.
                self._sync_lambda_distributed()
            # If cos_sims empty (checkpoint resume before calib step), retry next step.

        self._pending_cos_sims.clear()
        self._sync_ema_load_distributed()

    @torch.no_grad()
    def _sync_pending_counts_distributed(self) -> None:
        try:
            import torch.distributed as dist
        except ImportError:
            return
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            return
        dist.all_reduce(self._pending_counts, op=dist.ReduceOp.SUM)
        dist.all_reduce(self._pending_count_n, op=dist.ReduceOp.SUM)

    @torch.no_grad()
    def _sync_ema_load_distributed(self) -> None:
        # Safety net after resume — primary sync is via _sync_pending_counts_distributed.
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
        dist.broadcast(self.lambda_val, src=0)
        dist.broadcast(self.sigma_cos_at_calib, src=0)

    @torch.no_grad()
    def _sync_welford_distributed(self) -> None:
        """Parallel Welford combine (Chan et al. 1979) across DDP ranks.

        NOT called in step() — all_gather requires all ranks to call in the same
        order, which is not guaranteed under gradient accumulation (caused NCCL
        deadlock). Call at eval time only when all ranks participate simultaneously.
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

        ns = torch.stack(n_list, dim=0)
        mus = torch.stack(mu_list, dim=0)
        M2s = torch.stack(M2_list, dim=0)
        combined_n, combined_mu, combined_M2 = (
            ns[0].clone(),
            mus[0].clone(),
            M2s[0].clone(),
        )

        for i in range(1, world_size):
            n_new = combined_n + ns[i]
            delta = mus[i] - combined_mu
            safe_n = n_new.clamp(min=1.0)
            combined_mu = (combined_n * combined_mu + ns[i] * mus[i]) / safe_n
            combined_M2 = (
                combined_M2 + M2s[i] + delta.pow(2) * combined_n * ns[i] / safe_n
            )
            combined_n = n_new

        self.welford_n.copy_(combined_n)
        self.welford_mu.copy_(combined_mu)
        self.welford_M2.copy_(combined_M2)

    def get_custom_metrics(
        self, indices: Optional[torch.Tensor], weights: torch.Tensor
    ) -> Dict[str, Any]:
        var = self._welford_variance()
        metrics = {
            "welford_mu_mean": self.welford_mu.mean().item(),
            "welford_var_mean": var.mean().item(),
            "welford_n_min": self.welford_n.min().item(),
            "ema_load_mean": self.ema_load.mean().item(),
            "ema_load_max": self.ema_load.max().item(),
            "ema_load_std": self.ema_load.std().item(),
            "lambda_val": self.lambda_val.item(),
            "sigma_cos_at_calib": self.sigma_cos_at_calib.item(),
            "tau": self._current_tau(),
            "noise_std": self._noise_std,
            "ema_load_per_expert": self.ema_load.cpu().float().numpy().tolist(),
        }

        if weights is not None and weights.dim() == 2:
            hard = (weights > 0).float().sum(dim=0)
        elif indices is not None:
            hard = torch.zeros(
                self.num_experts, device=indices.device, dtype=torch.float32
            )
            hard.scatter_add_(
                0,
                indices.flatten().clamp(min=0),
                torch.ones(indices.numel(), device=indices.device),
            )
        else:
            hard = torch.ones(self.num_experts)
        hard = hard / hard.sum().clamp(min=1e-8)
        metrics["eff_E_hard"] = (1.0 / (hard**2).sum().clamp(min=1e-8)).item()
        return metrics

    def compute_aux_loss(self) -> torch.Tensor:
        return self.W.new_zeros(1).squeeze()

    def clear_aux_state(self) -> None:
        pass

    def reset_state(self) -> None:
        with torch.no_grad():
            self.ema_load.fill_(1.0 / self.num_experts)
            self.lambda_val.fill_(self.config.lambda_init)
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
        """Reset Welford accumulators. Call periodically to prevent fp32 overflow
        when welford_n grows large (~10^8 after 19k steps). Safe — metrics only.
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
