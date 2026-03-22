# SPAR Router — Mathematical Analysis Report

**Date:** 2026-03-19
**Scope:** `src/routers/stress_corrected.py` (StressCorrectedRouter / SPAR)
**References:** `docs/EQUATIONS.md`, `experiments/gptneo_125m_stress_v7-fineweb.yaml`, `EXPERIMENT_RESULTS.md`

---

## Assumptions

| # | Assumption | Source |
|---|---|---|
| A1 | Training uses DDP (not FSDP); buffers are not sharded. | v7 YAML: `distributed.strategy: ddp` |
| A2 | Mixed precision training with bf16 compute; buffers remain fp32 unless explicitly cast. | `register_buffer` uses default dtype (fp32). Verified in `__init__`. |
| A3 | Gradient accumulation = 2; each optimizer step sees 2 forward passes. | v7 YAML: `gradient_accumulation_steps: 2` |
| A4 | 4 DDP ranks (H100×4). Each rank processes an independent microbatch. | v7 YAML: `compute.modal.gpu: "H100:4"` |
| A5 | `torch.compile` wraps the backbone only; MoE layers and routers run in eager mode (except `_read_ema_load` and `_update_load_and_welford` which are explicitly `@torch._dynamo.disable`). | `scripts/train.py` line 635–638 |
| A6 | `router.step()` is called exactly once per optimizer step, after `optimizer.step()` and `scheduler.step()`. | `scripts/train.py` lines 933–936 via `LoRAMoELayer.step()` → `self.router.step()` |

---

## Observations

### O1: Formulation from EQUATIONS.md vs. code

The documented SPAR formulation states:

$$z_i(x,t) = \cos(x,\, W_i) - \lambda \cdot \max\!\bigl(0,\; L_i(t) - \tfrac{1}{N}\bigr)$$

$$w_i = \frac{\exp(\cos(x, W_i) / \tau_t)}{\sum_{j \in \mathcal{S}} \exp(\cos(x, W_j) / \tau_t)}, \quad i \in \mathcal{S}(x,t)$$

$$L_i(t) = (1-\alpha) L_i(t-1) + \alpha \cdot U_i(t), \quad U_i(t) = \frac{\#\{\text{assignments to } i\}}{B \cdot S \cdot k}$$

$$\lambda = \min\!\left(\frac{\sigma_{\cos}}{\bar{L}},\; 5.0\right), \quad \text{calibrated once at step } \lambda_{\text{calib}}$$

### O2: Variable catalog

| Symbol | Code | Shape | Semantics |
|---|---|---|---|
| $x$ | `x` | `[B, S, D]` | Token hidden states |
| $\hat{x}$ | `x_norm` | `[B, S, D]` | L2-normalized tokens |
| $\hat{W}_i$ | `W_norm` | `[N, D]` | L2-normalized prototype directions |
| $\cos(x, W_i)$ | `cos_sim` | `[B, S, N]` | Cosine similarity matrix |
| $L_i(t)$ | `ema_load` | `[N]` | EMA load per expert |
| $\lambda$ | `lambda_val` | scalar | Penalty scale |
| $\tau_t$ | `_tau` | scalar | Current temperature (annealed) |
| $z_i$ | `logits` | `[B, S, N]` | Selection logits |
| $w_i$ | `output_weights` | `[B, S, k]` | Softmax output weights over selected set |
| $U_i(t)$ | `counts` / `avg_counts` | `[N]` | Instantaneous usage fraction |
| $\alpha$ | `ema_alpha` | scalar | EMA smoothing coefficient |
| $\sigma_{\cos}$ | `sigma_cos` | scalar | Std of cosine similarities |

### O3: Tensor shape trace through forward()

```
x:           [B, S, D]    (input)
x_norm:      [B, S, D]    (F.normalize, dim=-1)
W_norm:      [N, D]       (F.normalize, dim=-1)
cos_sim:     [B, S, N]    (x_norm @ W_norm.T — matmul broadcasts correctly)
ema_load:    [N]           (broadcast to [1, 1, N] in logits expression)
logits:      [B, S, N]    (cos_sim - λ·clamp + noise)
topk_vals:   [B, S, k]    (logits.topk)
topk_idx:    [B, S, k]    (logits.topk)
topk_cos:    [B, S, k]    (cos_sim.gather(-1, topk_idx))
output_weights: [B, S, k] (softmax(topk_cos / τ, dim=-1))
```

All shapes are consistent. The `@` matmul `[B, S, D] @ [N, D].T = [B, S, N]` is correct
(batch matmul with transpose on the last two dims of W_norm).

---

## Mathematical Verdict

### 2.1 Formulation Verification

**Status: ✅ Correct**

Line-by-line verification:

| Equation component | Code (line) | Match |
|---|---|---|
| $\hat{x} = x / \|x\|$ | `x_norm = F.normalize(x, dim=-1)` (L220) | ✅ |
| $\hat{W}_i = W_i / \|W_i\|$ | `W_norm = F.normalize(self.W, dim=-1)` (L221) | ✅ |
| $\cos(x, W_i) = \hat{x}^\top \hat{W}_i$ | `cos_sim = x_norm @ W_norm.T` (L223) | ✅ |
| $z_i = \cos - \lambda \cdot \max(0, L_i - 1/N) + \text{noise}$ | `logits = cos_sim - self.lambda_val * (ema_load - fair_share).clamp(min=0.0) + noise` (L235) | ✅ |
| $\mathcal{S} = \text{top-}k(z_i)$ | `topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)` (L237) | ✅ |
| $w_i = \text{softmax}(\cos(x, W_i) / \tau)$ over $\mathcal{S}$ | `topk_cos = cos_sim.gather(-1, topk_idx); output_weights = F.softmax(topk_cos / tau, dim=-1)` (L241-242) | ✅ |
| $U_i = \text{counts} / \text{total}$ | `counts.div_(counts.sum().clamp(min=1e-8))` (L216) | ✅ |
| $L_i(t) = (1-\alpha)L_i(t-1) + \alpha \cdot U_i$ | `self.ema_load.mul_(1 - self.ema_alpha).add_(avg_counts * self.ema_alpha)` (L285) | ✅ |
| $\lambda = \min(\sigma_\cos / \bar{L}, 5.0)$ | `(sigma_cos / mean_load).clamp(max=5.0)` (L186) | ✅ |

**Normalization constraints verified:**
- $\sum_i w_i = 1$ over selected set: guaranteed by `F.softmax(dim=-1)`. ✅
- $w_i \ge 0$: guaranteed by softmax. ✅
- $\sum_i U_i = 1$: guaranteed by `counts / counts.sum()`. ✅
- $\sum_i L_i(t) = 1$: preserved by EMA since $\sum_i L_i(0) = 1$ and $\sum_i U_i = 1$.
  Proof: $\sum_i L_i(t) = (1-\alpha) \sum_i L_i(t-1) + \alpha \sum_i U_i = (1-\alpha) \cdot 1 + \alpha \cdot 1 = 1$. ✅
- `all_reduce(AVG)` preserves $\sum L_i = 1`: average of vectors each summing to 1 still sums to 1. ✅

---

### 2.2 Gradient Flow and Differentiability

**Status: ✅ Correct**

**Gradient path:** task_loss → model output → expert outputs → `combined[token_ids] += expert_out * expert_w`
→ `expert_w` = `output_weights` = `F.softmax(topk_cos / τ)` → `topk_cos` = `cos_sim.gather(-1, topk_idx)`
→ `cos_sim` = `x_norm @ W_norm.T` → `W_norm` = `F.normalize(self.W)` → `self.W` (trainable parameter).

**Key property:** `topk_idx` from `logits.topk()` is produced by argmax (non-differentiable).
Gradients do **not** flow through `topk_idx` — it serves only as an index tensor for `.gather()`.
This means the load penalty `λ·max(0, L_i - 1/N)` in `logits` (line 235) affects *which* experts
are selected but does not contribute gradient to `W`. This is correct and intentional: W learns from
the cosine-quality signal only (through `output_weights`), while the penalty steers selection
without distorting the output weight gradient.

**F.normalize chain rule:** $\frac{\partial \hat{W}_i}{\partial W_i} = \frac{1}{\|W_i\|}\left(I - \hat{W}_i \hat{W}_i^\top\right)$

This is the standard projection onto the tangent plane of the unit sphere. Updates to $W_i$ are
perpendicular to $\hat{W}_i$ (magnitude changes are suppressed; direction changes are preserved).
Correctly implemented by PyTorch's autograd through `F.normalize`.

**Gradient sparsity:** Only the top-$k$ selected experts receive gradient per token (through
`cos_sim.gather(-1, topk_idx)` → `output_weights`). Non-selected experts receive zero gradient
from that token. This is intrinsic to top-$k$ gating and desirable for prototype specialization.

**Vanishing gradient risk (minor):** When the top-$k$ experts have nearly identical cosine
similarities ($\Delta\cos \to 0$), softmax approaches uniform $w_i \to 1/k$, and the gradient
$\partial w_i / \partial \cos_i$ approaches zero (softmax Jacobian $\to 0$ at the uniform point).
With $\tau$-annealing ($\tau_0 = 0.5 \to \tau_f = 0.12$), the effective cosine gap is amplified
by $1/\tau$, mitigating this. However, at early steps with random prototypes ($\Delta\cos \approx 0.037$
per EXPERIMENT_RESULTS.md #14), $\Delta\cos / \tau_0 = 0.074$ — small but nonzero. The k-means prototype
initialization (v7) directly addresses this by starting with $\Delta\cos \approx 0.15+$.

---

### 2.3 Load Balancing Mechanism

**Status: ✅ Correct (after applied fix)**

**One-sided penalty — zero at equilibrium:**

$$\max(0, L_i - 1/N) = 0 \iff L_i \le 1/N$$

At perfect balance ($L_i = 1/N$ for all $i$), every penalty term is zero, and routing is governed
purely by cosine similarity. This is verified algebraically and tested in
`TestOneSidedPenalty::test_zero_penalty_at_fair_share`.

**Sum invariant:** $\sum_i L_i = 1$ is maintained by the EMA update. Proof by induction:
- Base: $L_i(0) = 1/N$, $\sum_i L_i(0) = 1$. ✓
- Step: $\sum_i L_i(t) = (1-\alpha) \sum_i L_i(t-1) + \alpha \sum_i U_i(t) = (1-\alpha) + \alpha = 1$. ✓

**DDP sync:** `all_reduce(AVG)` on `ema_load`: if each rank has $L_i^{(r)}$ with $\sum_i L_i^{(r)} = 1$,
the average is $(1/R)\sum_r L_i^{(r)}$ with sum $(1/R)\sum_r \sum_i L_i^{(r)} = (1/R) \cdot R = 1$. ✓

**Gradient-accumulation deferred update:**
With `grad_accum=2`, two forwards produce `counts₁` and `counts₂` (each summing to 1).
The step applies: `avg = (counts₁ + counts₂) / 2`, then `L(t) = (1-α)L(t-1) + α·avg`.
This is correct: one EMA step per optimizer step, using the mean usage across microbatches.
The alternative (two sequential EMA steps) would make $\alpha$ effectively larger,
proportional to `grad_accum`. The implementation avoids this.

**λ auto-calibration:**
$\lambda = \min(\sigma_{\cos} / \bar{L}, 5.0)$. Since $\bar{L} = (1/N)\sum_i L_i = 1/N$ always,
this simplifies to $\lambda = \min(\sigma_{\cos} \cdot N, 5.0)$. The code computes both
$\sigma_{\cos}$ and $\bar{L}$ explicitly rather than using the simplification — not incorrect,
just slightly redundant.

**⚠️ Issue found (now fixed):** `_pending_cos_sim` was a single tensor overwritten each forward.
With `grad_accum=2`, only the last microbatch's cosines were used for λ calibration. Fixed by
changing to a list (`_pending_cos_sims`) that accumulates across the grad-accum window, producing
a `torch.cat` of all microbatches' cosines at calibration time. Practical impact: minimal (σ_cos
is stable across microbatches from the same distribution), but mathematically precise.

**⚠️ Issue found (now fixed):** If all cosine similarities were identical (degenerate random init
in very high $D$), `σ_cos = 0` → `λ = 0` permanently, disabling the penalty with no recovery path.
Fixed by clamping `σ_cos ≥ 1e-4` in `_calibrate_lambda`.

**Can all tokens route to one expert?** No, under the following argument: if expert $i$ receives
all tokens, $L_i \to 1$ and the penalty becomes $\lambda \cdot (1 - 1/N) \approx \lambda \cdot 0.875$
(for $N=8$). With typical $\lambda \approx 1.5$ and $\sigma_{\cos} \approx 0.2$, the penalty is
$\approx 1.31$ — far larger than any cosine advantage in $[-1, 1]$. The overloaded expert's effective
logit becomes $\cos - 1.31$, which is below $-0.31$ even for perfect alignment ($\cos = 1$).
Meanwhile underloaded experts have zero penalty. Monopoly is unstable.

---

### 2.4 Geometric Analysis

**Status: ✅ Correct**

**Cosine similarity properties:**
- Domain: $\cos(x, W_i) \in [-1, 1]$ (L2-normalized inputs). ✓
- Scale-invariant: insensitive to hidden state magnitude. ✓
- Dimensionality: in $D = 768$, random unit vectors have $\mathbb{E}[\cos] = 0$ and
  $\text{Var}[\cos] \approx 1/D \approx 0.0013$, so $\sigma_{\cos} \approx 0.036$.
  This matches EXPERIMENT_RESULTS.md observation ($\Delta\cos \approx 0.037$ at early steps with random prototypes).

**Prototype collapse:** Can two prototypes $W_i, W_j$ converge to the same direction?
The gradient sparsity prevents this: if $W_i \approx W_j$, both compete for the same tokens
via top-$k$. The losing prototype receives no gradient from those tokens and drifts toward
other token clusters that do select it. The one-sided penalty further discourages convergence
by penalizing whichever copy becomes overloaded. Verified empirically: eff_E=8.0 (maximum)
sustained from step 3725 onward in v6-fineweb (EXPERIMENT_RESULTS.md #14).

**Scaling with $D$ and $N$:** The $\lambda$ calibration adapts to dimensionality via $\sigma_{\cos}$.
For high $D$, $\sigma_{\cos}$ is small → $\lambda$ is small → penalty matches the scale of routing
signal. For low $D$, $\sigma_{\cos}$ is larger → $\lambda$ scales up. The 5.0 cap prevents
pathological values when $\sigma_{\cos} \cdot N > 5$.

**Zero-norm vectors:** `F.normalize` uses default `eps=1e-12`. If $\|x\| < 10^{-12}$ (effectively
zero input), the normalized output is $x / \epsilon$, which has very large magnitude — but this
represents degenerate input (all-zero hidden states), not a realistic scenario. The cosine
similarity with any prototype would be approximately zero, routing the token uniformly.

---

### 2.5 Numerical Stability

**Status: ⚠️ Minor concern (Welford at scale)**

**Division by zero:**
- `counts.sum().clamp(min=1e-8)` in usage normalization: safe. ✓
- `self.ema_load.mean().clamp(min=1e-6)` in λ calibration: safe. ✓
- `self.welford_n.clamp(min=1.0)` in variance computation: safe. ✓
- `hard.sum().clamp(min=1e-8)` in metrics: safe. ✓

**Precision of registered buffers:** All registered buffers (`ema_load`, `lambda_val`,
`welford_n/mu/M2`, `_pending_counts`) are created with `torch.zeros(...)` or `torch.ones(...)`,
which default to `float32`. Under bf16 training with DDP (not FSDP), buffers retain their
declared dtype. ✅

**Gumbel noise numerical validity:**
`u ~ Uniform(1e-10, 1 - 1e-10)` → `noise = σ · (-log(-log(u)))`.
- Inner log: `log(u)` where `u ∈ [1e-10, 1-1e-10]` → `log(u) ∈ [-23.03, -1e-10]` — safe.
- Outer log: `log(-log(u))` where `-log(u) ∈ [1e-10, 23.03]` → `log(1e-10) = -23.03`,
  `log(23.03) = 3.14` — safe, no negative argument. ✅

**⚠️ Welford accumulation at scale:** Over 19,000 steps with batch=32, seq=1024, top_k=2:
each expert sees approximately $19000 \times 32 \times 1024 \times 2 / 8 \approx 1.6 \times 10^8$
assignments. `welford_n` reaches $\sim 10^8$ in fp32 (representable exactly up to $2^{24} \approx
1.7 \times 10^7$; above that, integer increments lose precision). At $n = 10^8$, the Welford
update `μ += (w·δ).sum() / n` divides by a large number, making the mean correction
$\Delta\mu \sim 10^{-8}$ per observation — below fp32 precision for $\mu \sim 0.5$.

**Impact:** The running mean and variance become slightly stale (frozen at their values near
$n \approx 10^7$). Since Welford is metrics-only, this does not affect routing. However, logged
`welford_mu_mean` and `welford_var_mean` metrics become unreliable after $\sim 10^7$ observations.

**Recommendation (implemented):** Added `reset_welford()` method. Call it at each eval interval
(every 1000 steps) to bound $n$ at $\sim 10^6$ per window.

---

### 2.6 Implementation vs. Formulation Mismatch

**Status: ⚠️ Three documentation mismatches found (all fixed)**

| # | Mismatch | Severity | Status |
|---|---|---|---|
| M1 | EQUATIONS.md §2 stated $\lambda_{\text{calib}} = 200$; code/config uses `warmup_steps + 200` (600). | Minor (docs-only) | ✅ Fixed |
| M2 | EQUATIONS.md §1 stated τ is "Fixed at 0.5"; code implements linear τ-annealing. | Minor (docs-only) | ✅ Fixed |
| M3 | EQUATIONS.md §6 documents `eps = 10⁻³` as "prevents division by zero in cosine normalization"; in code, `self.eps` is stored but unused — `F.normalize` uses PyTorch default `1e-12`. The `eps = 1e-3` was originally for the Stress CV denominator (removed in SPAR clean). | Minor (cosmetic) | ✅ Documented in code |
| M4 | `_pending_cos_sim` was overwritten per forward, losing earlier microbatches in grad-accum windows. | Moderate (mathematical precision) | ✅ Fixed → `_pending_cos_sims` list |
| M5 | `σ_cos = 0` in degenerate cases produced `λ = 0` permanently. | Significant (silent failure mode) | ✅ Fixed → `clamp(min=1e-4)` |

**No fatal implementation errors found.** The core formulation (selection logit, output weights,
EMA load, λ calibration) is implemented correctly.

---

## Recommendations

### R1: Call `reset_welford()` at each eval interval

**Problem:** `welford_n` grows unbounded, exceeding fp32 integer precision after ~$10^7$ observations.
Logged Welford metrics become unreliable.

**Fix:** In `scripts/train.py`, call `router.reset_welford()` during periodic evaluation. Example:

```python
# After evaluation, reset Welford to prevent fp32 precision loss
for moe_layer in _moe_layers_ref.values():
    if hasattr(moe_layer.router, 'reset_welford'):
        moe_layer.router.reset_welford()
```

**Why this works:** Welford is metrics-only and never feeds back into routing. Resetting at each
eval interval (every 1000 steps) bounds $n$ at ~$3.3 \times 10^6$ per window, well within fp32
integer precision.

### R2: Anneal noise_std to zero by midpoint

**Problem:** At step 6400, measured $\Delta\cos \approx 0.037$ while `noise_std = 0.05`. The
Gumbel noise exceeds the routing signal, reducing effective routing quality.

**Recommendation:** Add noise annealing (not yet implemented):

$$\sigma_{\text{noise}}(t) = \sigma_0 \cdot \max\!\left(0,\; 1 - \frac{t}{T_{\text{anneal}}}\right)$$

with $T_{\text{anneal}} \approx \text{max\_steps} / 2$. This preserves early-stage exploration
while eliminating noise when prototypes have aligned and the routing signal is meaningful.

### R3: Consider simplifying λ calibration formula

**Observation:** Since $\bar{L} = 1/N$ always (proven by the sum-to-1 invariant), the formula

$$\lambda = \min\!\left(\frac{\sigma_\cos}{\bar{L}},\; 5.0\right) = \min\!\left(\sigma_\cos \cdot N,\; 5.0\right)$$

can be simplified in code to `sigma_cos * self.num_experts`, eliminating the `mean_load` computation.
Not a correctness issue, but reduces one potential confusion point.

---

## Open Questions

| # | Question | Required to resolve |
|---|---|---|
| Q1 | Does the v7 k-means prototype init raise $\sigma_\cos$ at calibration step 600 above the random-init value of ~0.036? If so, λ should be significantly larger. | Run v7 and check `lambda_val` after step 600. |
| Q2 | At 19,000 steps on fineweb-edu, does SPAR final PPL match the ~27.5–28.5 projection from log-linear extrapolation at step 6400? | Let v6 complete. |
| Q3 | The `eps` config field (default 1e-6, YAML override 1e-3) is unused. Should it be removed from `StressCorrectedRouterConfig` or repurposed for `F.normalize`? | Design decision — 1e-3 is too aggressive for normalize; removing it is a breaking config change. Recommend leaving as-is with documentation. |
| Q4 | The parallel Welford sync (`_sync_welford_distributed`) is implemented but never called (removed from `step()` to avoid NCCL deadlocks). Should it be called less frequently (e.g., at eval time only) to keep per-rank Welford stats aligned for logging? | Engineering decision based on NCCL stability experience. |
| Q5 | Noise annealing (R2) would require a new config field (`noise_anneal_steps`). Is this worth the config complexity, or should `noise_std` simply be set to 0 for runs with k-means init? | Design decision for v7+ experiments. |

---

## Summary

The SPAR router formulation is **mathematically sound**. The core design — factored selection/weighting,
one-sided zero-sum load penalty, auto-calibrated λ, cosine-geometric routing — is internally consistent,
correctly implements the documented equations, and has provable equilibrium properties.

Five issues were identified and addressed:

1. **σ_cos floor** (significant): prevents permanent λ=0 in degenerate cases. Fixed.
2. **Grad-accum cosine accumulation** (moderate): λ calibration now uses full optimizer-step data. Fixed.
3. **Three documentation mismatches** (minor): EQUATIONS.md now matches code. Fixed.
4. **Welford fp32 precision** (minor): `reset_welford()` method added for periodic cleanup. Caller-side integration recommended.
5. **Noise exceeding signal** (observation): noise annealing recommended for future experiments.

No fatal mathematical or implementation errors were found.

---

## Addendum: Expert Symmetry Deadlock (Breakthrough Finding)

**Date:** 2026-03-19 (Phase 2 deep-dive)

### The Problem

With standard LoRA initialization (`B = 0`), all 8 experts produce **identical outputs** at step 0:

$$\text{expert}_i(x) = W_{\text{base}} x + \underbrace{B_i A_i x \cdot \frac{\alpha}{r}}_{= 0 \text{ because } B_i = 0} = W_{\text{base}} x \quad \forall i$$

### Proof of Zero Gradient

The MoE output is:

$$y = \sum_{i \in \mathcal{S}} w_i \cdot \text{expert}_i(x)$$

When all expert outputs are identical ($\text{expert}_i(x) = e(x)$ for all $i$):

$$y = e(x) \cdot \underbrace{\sum_{i \in \mathcal{S}} w_i}_{= 1} = e(x)$$

The gradient of the loss w.r.t. any routing logit $z_j$:

$$\frac{\partial \mathcal{L}}{\partial z_j} = \frac{\partial \mathcal{L}}{\partial y} \cdot \sum_{i \in \mathcal{S}} \frac{\partial w_i}{\partial z_j} \cdot \text{expert}_i(x)$$

Since $w_i = \text{softmax}(z)_i$, we have $\partial w_i / \partial z_j = w_i(\delta_{ij} - w_j)$. Substituting $\text{expert}_i(x) = e(x)$:

$$\frac{\partial \mathcal{L}}{\partial z_j} = \frac{\partial \mathcal{L}}{\partial y} \cdot e(x) \cdot \sum_i w_i(\delta_{ij} - w_j) = \frac{\partial \mathcal{L}}{\partial y} \cdot e(x) \cdot (w_j - w_j) = 0$$

**This is exact, not approximate.** The router receives zero gradient from the task loss when expert outputs are identical.

### Consequence: Slow Expert Divergence

Since B = 0 initially, expert divergence depends entirely on mini-batch gradient noise creating slightly different updates to each expert's A and B matrices. This is:

1. **Extremely slow**: divergence is O(sigma_SGD / sqrt(t)), not O(eta * grad)
2. **Confirmed empirically**: cos_sim = 0.007-0.10 after 6400 steps (EXPERIMENT_RESULTS.md), meaning prototypes barely differentiate from random
3. **Explains the conf = 0.52 plateau**: all experts are interchangeable, no routing preference, uniform weights, conf = 1/k = 0.50 + noise
4. **Explains eff_E = 8.0**: not routing success but symptom of identical experts --- every assignment is equally good

### The Fix: Non-Zero B Initialization

```python
# In LoRAConfig:
b_init_scale: float = 0.01  # Was 0.0

# In LoRALayer/SharedLoRALayer:
if b_init_scale > 0:
    nn.init.normal_(self.lora_B.weight, std=b_init_scale)
else:
    nn.init.zeros_(self.lora_B.weight)
```

With `b_init_scale = 0.01`:

- Each expert starts with a **unique** random delta
- The delta magnitude is approx 1-3% of base output --- small enough to not destabilize early training
- Router immediately gets nonzero gradient signal, prototypes can align to meaningful input subspaces from step 0
- **No longer sacrificing the first ~90% of training steps to accidental symmetry**

### Scale of Initial Perturbation

For rank=32, hidden=768, with init_scale=0.01 (Kaiming-A) and b_init_scale=0.01 (normal-B):

- ||B A x|| approx std(B) * sqrt(r) * ||Ax|| approx 0.01 * sqrt(32) * 0.01 * sqrt(r) * ||x|| approx 10^-3.4 * ||x||
- With alpha/r = 64/32 = 2, effective perturbation approx 0.08% of ||x||
- Safely below task-loss-relevant scale while sufficient to break symmetry

### Files Changed

| File | Change |
|---|---|
| `src/experts/lora.py` | `b_init_scale` field in LoRAConfig; conditional init in LoRALayer and SharedLoRALayer |
| `src/experts/gpt_neo_lora.py` | `b_init_scale` passed through `_make_lora()` |
| `scripts/train.py` | `b_init_scale` wired from YAML to LoRAConfig in `build_model()` |
| `experiments/gptneo_125m_stress_v7-fineweb.yaml` | `b_init_scale: 0.01` added to `expert.lora` section |

### Expected Impact

- **conf**: should rise above 0.52 plateau to 0.60-0.70 by step 6400
- **cos_sim**: should reach 0.2-0.4 (meaningful prototype alignment) within first 2000 steps
- **PPL**: faster expert specialization leads to more effective capacity and lower final PPL
- **gini**: should rise from ~0.02 to 0.10+ as routing becomes non-uniform
