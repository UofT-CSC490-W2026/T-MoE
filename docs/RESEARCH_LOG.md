## SPAR Formulation Decisions

- **[architecture]** One-sided zero-sum penalty `max(0, L_i - 1/N)` chosen over raw `-λ·L_i`.
  **result:** At equilibrium all penalties cancel to zero — routing is governed purely by cosine
  similarity. Raw EMA penalty imposes a constant floor that wastes signal budget.
  **note:** This is the core theoretical property of SPAR: the penalty disappears when load balance
  is achieved, unlike auxiliary loss which applies a permanent gradient perturbation.

- **[architecture]** Factored output weights: `w_i = softmax(cos(x, W_i) / τ)` over selected set
  only, separate from the selection logit `z_i = cos(x,W_i) - λ·max(0, L_i - 1/N)`.
  **result:** W learns from cosine-quality signal only (through `output_weights`). Load penalty
  steers selection without contributing gradient to W — gradient sparsity is intentional.
  **note:** Gradient does not flow through `topk_idx` (argmax is non-differentiable). This clean
  separation of selection mechanism and adaptation signal is the factored design principle.

- **[architecture]** `mu_stress=0` (set permanently): stress term `μ·CV(1-cos)` was tested at
  μ=0.5 and μ=0.1 before removal.
  **result:** μ=0.5 caused eff_E collapse 7.5→5.8, gini spike 0.11→0.41 (structural Welford
  bias: dominant experts have more observations → inflated CV → double-suppression). μ=0.1 had
  zero effect — load penalty dominates at any μ≤0.1. Same eff_E=5.8 equilibrium in both cases.
  **note:** Removing mu_stress was the single change that raised conf from 0.527 to 0.562 and
  prevented eff_E collapse. Stress CV is retained as a logged diagnostic only (not in logit).

- **[architecture]** EMA load (α=0.01) chosen over fatigue buffers (the earlier MetabolicRouter
  approach). EMA has a natural 100-step forgetting window; fatigue buffers required 6
  `dist.all_reduce` calls per step. SPAR uses 1 all_reduce per step.
  **result:** SPAR DDP overhead reduced to 1 `all_reduce` per step vs MetabolicRouter's 6.
  **note:** The EMA sum-to-1 invariant (proven: `Σ L_i(t) = (1-α)·1 + α·1 = 1`) means fair share
  is exactly `1/N` at all times, making the calibration formula `λ = min(σ_cos · N, 5.0)`.

- **[architecture]** λ auto-calibration: `λ = min(σ_cos / mean_L, 5.0)` fired once at
  `lambda_calib_step`. Since `mean_L = 1/N` always, simplifies to `min(σ_cos · N, 5.0)`.
  **result:** Penalty scale adapts to dimensionality — in D=768, σ_cos≈0.036 → λ≈0.29; the
  calibration matches penalty magnitude to routing signal magnitude automatically.
  **note:** `sigma_cos` floored at `1e-4` to prevent permanent λ=0 in degenerate random init.
  `_pending_cos_sims` accumulates across grad-accum window so calibration uses full optimizer
  step data (not just the last microbatch).

---

## Key Bug Discoveries

- **[bug]** `clamp(min=0)` on stress term in pre-fix StressCorrectedRouter zeroed gradients for
  ~50% of selected experts (those below stress threshold).
  **result:** Routing collapse on affected experts; fixed by removing the clamp entirely and
  shifting to the factored output-weight formulation.
  **note:** This was the most impactful early bug — masked as "stress working correctly" since
  penalized experts did receive fewer tokens.

- **[bug]** `num_steps` in early SPAR counted forward passes not optimizer steps. λ calibration
  fired at optimizer step ~12 instead of step 50 (with grad_accum=4). EMA had not converged.
  **result:** λ was calibrated on unconverged load estimates → calibrated value was noisy.
  **note:** Fixed by incrementing `num_steps` in `step()` (called once per optimizer step), not
  in `forward()`.

- **[bug]** `lambda_init=1.0` in Qwen2 v1 config. Calibrated λ for fineweb-edu is ~0.1, so the
  pre-calibration penalty was 10× too strong.
  **result:** Steps 0–999: eff_E climbed 5.4→7.6, gini fell 0.469→0.18 — artificial balance
  suppressing routing signal. At step 1000 calibration: λ drops 1.0→0.1 discontinuously.
  eff_E crashed 7.5→6.2, gini spiked 0.18→0.374 in one step (pent-up routing preference released).
  **note:** Spike magnitude (Δgini=+0.194, Δeff_E=−1.3) proportional to suppressed differentiation
  over 1000 steps. Fixed in v3: `lambda_init=0.1`. LM loss descent unaffected throughout — early
  decoupling of routing quality and LM quality confirmed.

- **[bug]** Welford buffers initialized in bf16 in early SPAR runs. Variance accumulation lost
  precision for small per-token cosine distance values.
  **result:** `welford_mu` and `welford_var` metrics were unreliable in affected runs.
  **note:** Fixed: all registered buffers use fp32 explicitly.

- **[bug]** DDP Welford sync (`_sync_welford_distributed`) caused SEQNUM drift and potential NCCL
  deadlock in early iterations (3 `all_gather` calls per step).
  **result:** Removed from `step()`. Welford is per-rank only — metrics not synced across DDP
  ranks. Per-rank Welford is sufficient for logging; routing decisions use `ema_load` (synced).
  **note:** This was the right tradeoff: Welford is metrics-only, never feeds back into routing.

- **[bug]** `_pending_cos_sim` (single tensor) was overwritten each forward in grad-accum windows.
  Only the last microbatch's cosines were used for λ calibration.
  **result:** Minimal practical impact (σ_cos is stable across microbatches from the same
  distribution), but mathematically imprecise.
  **note:** Fixed: `_pending_cos_sims` list accumulates across the grad-accum window.

---

## SPAR Wikitext-103 Baseline (first full run)

`gptneo_125m_spar_wikitext`, 2026-03-18

- **[experiment]** GPT-Neo 125M, 8 experts, top-k=2, 6 MoE layers [1,3,5,7,9,11], rank=16,
  τ=0.5, α=0.01, λ_calib_step=200, wikitext-103, 5000 steps, A100:4, 108 min.
  **result:** val_ppl=22.9 (val_loss=3.1333). eff_E=6.1 (stable 6.0–6.2 from step 2000),
  gini=0.388 (peaked 0.424 at step 1000, declined), conf=0.562. 1 `all_reduce` per step.
  **note:** Statistically tied with metabolic_v4 (val_loss=3.1246) and standard_v2 (3.1212).
  The SPAR wikitext-103 baseline. Task-loss parity at 50% higher effective expert utilization.

- **[observation]** PPL trajectory: 89% of gain in first 10% of steps (51.2→25.9 over steps
  0–500, then 25.9→22.9 over steps 500–5000). conf growing monotonically, not plateaued at 5000.
  **result:** eff_E settling at 6.1/8 rather than 8/8 is expected — SPAR prevents monopolisation,
  not enforces uniformity. Post-peak gini decline (0.424→0.388) is the EMA fixed-point forming.
  **note:** LM loss plateau 3.29–3.32 from step 1500 onward while val loss kept declining — no
  overfitting gap. LoRA dropout (0.05) + routing noise provides sufficient regularization.

---

## SOTA Comparison: SPAR vs DeepSeek V3 (2026-03-22)

- **[observation]** DeepSeek V3 bias correction vs SPAR EMA penalty:

  | Property | SPAR | DeepSeek |
  |----------|------|----------|
  | Penalty on underloaded | Zero (one-sided max) | Negative (sign pushes b_i up) |
  | Update smoothness | Continuous EMA (α=0.01) | Discrete (sign ±γ) |
  | Scale sensitivity | Auto-calibrated λ | Fixed γ (manual tuning) |
  | Steady-state | L_i = 1/N → penalty = 0 | b_i drifts without bound |
  | Gradient interaction | Out-of-graph (detached ema_load) | Out-of-graph |

  **result:** SPAR v1 beats DeepSeek V3 on wikitext103_ppl (10.99 vs 12.20, Qwen2-1.5B).
  DeepSeek edges SPAR v1 on lm_harness avg by +0.45%.
  **note:** SPAR's one-sided EMA penalty is a continuous, auto-calibrated generalization of
  DeepSeek V3's discrete bias correction. The one-sided property is the key differentiator:
  DeepSeek's sign function pushes underloaded experts' biases up AND overloaded down — a global
  shift that confounds the routing signal. SPAR only penalizes overloaded experts.

---

## Voronoi Constraint / Perfect Load Paradox (2026-03-19)

- **[finding]** Solved for Δcos from conf+τ at two checkpoints (v7-fineweb):
  - Step 1775: τ=0.43, conf=0.522 → Δcos = **0.038**
  - Step 8000: τ=0.196, conf=0.537 → Δcos = **0.029**
  Prototypes moving closer, not further apart, despite τ annealing.
  **result:** eff_E=8.0 forces prototypes to Voronoi cell boundaries (equidistant from adjacent
  tokens by definition). At a Voronoi boundary, Δcos → 0 structurally.
  **note:** Perfect load balance and high conf are in fundamental tension. You cannot have both.
  conf ceiling at eff_E=8.0 is ~0.54–0.58 in D=768 — a mathematical consequence of the primary
  claim, not a failure mode. Report eff_E and gini; not conf.

- **[observation]** Confirmed on both wikitext-103 (v7 step 8000, gini=0.046) and fineweb-edu
  (v8a step 12075, gini=0.047–0.096). The pathology is structural to SPAR at full capacity, not
  corpus-specific. SharedBaseLoRA did not move gini on fineweb-edu (0.047–0.096 throughout 12k
  steps) — the problem is in routing space, not adapter architecture.
  **result:** Fix must change how cosine similarities are computed. GMR (Global Mean Projection)
  planned for v8b: route in `x_proj = x - (x·v_global)v_global` residual space to create
  heterogeneous similarities even on diverse corpora.
  **note:** v8b is also the first bf16 run (dtype kwarg bug fixed: `torch_dtype=COMPUTE_DTYPE`
  was wrong kwarg in all prior runs, silently ignored; all v6/v7/v8a ran fp32). PPL comparisons
  v6/v7/v8a are internally consistent but not comparable to v8b.

---

## Qwen2-1.5B Scale-Up Findings (2026-03-22)

- **[scaling]** SPAR λ calibration at D=1536: `σ_cos ~ 1/sqrt(D) ~ 0.026` → `λ = min(0.026 * 8,
  5.0) = 0.207`. Penalty at max overload `L_i - 1/N = 0.05`: `0.207 * 0.05 = 0.010`. Cosine
  signal (Δcos ~ 0.026) is ~2.6× the penalty term.
  **result:** Penalty remains a perturbation, not a dominant term, at Qwen2 scale.
  **note:** Monitor eff_E at D=1536. If eff_E drops below 7.0, add a `lambda_floor` parameter:
  `λ = max(min(σ_cos * N, 5.0), lambda_floor)` with `lambda_floor=0.5`.

- **[architecture]** Qwen2 SwiGLU expert class (`qwen2_lora`) routes on 3 projections (gate,
  up, down). `F.normalize → cos_sim → logit` pipeline is architecture-agnostic (only depends on
  hidden_dim D). RMSNorm vs LayerNorm does not affect cosine routing: `cos(x, W_i)` is
  scale-invariant; only direction matters.
  **result:** No router changes needed for Qwen2. Expert class and tokenizer handling only.
  **note:** `intermediate_dim` must be set explicitly for SwiGLU (Qwen2-1.5B: 8960, not 4×1536).

- **[observation]** Under FSDP (required for 7B+): router buffers (`ema_load`, `lambda_val`,
  `welford_*`) are registered via `register_buffer` → replicated across ranks, not sharded.
  All-reduce in `step()` called after `optimizer.step()` — outside FSDP backward pass. Correct.
  **result:** FSDP compatibility confirmed by design for buffer-based router state.
  **note:** Router module should be in a separate FSDP wrap unit to prevent `W` and `ema_load`
  from being treated as part of an expert shard.

---

## Expert Symmetry Deadlock and B-init Fix (2026-03-19)

- **[architecture]** Standard LoRA init (B=0): all 8 experts produce identical outputs at step 0
  → router gradient is exactly zero (`∂L/∂z_j = 0` when all expert outputs are equal — proven
  algebraically via softmax Jacobian identity). Expert divergence from SGD noise is O(σ_SGD/√t).
  **result:** cos_sim=0.007–0.10 after 6400 steps on fineweb-edu — prototypes barely differentiate
  from random. conf=0.52 plateau = symptom of identical experts, not routing failure.
  **note:** `b_init_scale=0.01` breaks symmetry immediately. Each expert starts with a unique
  random delta (~0.08% of ||x||, safely below task-loss scale). Router gets nonzero gradient from
  step 0. k-means prototype init is the complementary fix for the W direction problem.

- **[ablation]** `b_init_scale=0.01` superseded by `init_from_data=true` (k-means prototype init,
  Experiment #17). K-means differentiates prototype directions W_i from step 0 — nonzero cosine
  similarity differences → nonzero router gradient immediately, without perturbed B.
  **result:** Experiment #17 confirmed b_init_scale had negligible benefit when k-means init is
  active. Current Qwen2 paper runs: `b_init_scale=0.0`, `init_from_data=true`.
  **note:** The deadlock is best resolved in routing space (W_i directions), not adapter space
  (B perturbation). k-means init addresses prototype direction divergence — the load-bearing fix
  for cosine routing.

- **[observation]** LoRA Without Regret (Schulman et al., Sep 2025): alpha=rank is optimal
  (scaling=1.0). Prior runs v6/v7/v8a used alpha=2×rank (scaling=2.0) — Adam compensates, so
  results are internally consistent, but alpha=rank removes a free variable.
  **result:** v8b: alpha 64→32 (rank=32), shared_base_alpha 16→8 (rank=8), batch_size 32→16
  (eff batch 64→32 — large batch penalizes LoRA more than FullFT).
  **note:** MLP-only LoRA validated by paper: rank-256 attention-only underperforms rank-128
  MLP-only at equal parameter count. SPAR's MLP-only placement is paper-optimal.

---

## Rejected Architectural Ideas (summary)

- **Expert Choice routing**: fatal — `combined = torch.zeros_like(x_flat)` silently zeros tokens
  selected by zero experts. Also: paper contradiction (hard capacity constraint ≠ SPAR's claim).

- **Prototype orthogonality regularization** `L_proto = α||W·W^T - I||²_F`: valid fixed point,
  expected conf improvement to 0.65–0.80. Fatal for paper: adds auxiliary loss, contradicting
  "zero auxiliary loss" claim. Defer to post-publication ablation.

- **Grassmannian routing** `p_i(x) = ||A_i·x̂||²/rank`: 55× worse discriminability at init
  (`std_expert ~ 6.5e-4` vs cosine's `1/sqrt(D) ~ 0.036`). Dual-use gradient conflict (A_i
  used for both routing and adaptation). Rejected.

- **Prototype LR multiplier** (3× base LR on W): prototypes outrun EMA load tracker (α=0.01,
  100-step lag) → stale penalties fire on wrong experts → eff_E dropped 7.6→7.0, gini volatile.
  Conf gain +0.005 does not justify eff_E loss −0.6. Fully reverted.
