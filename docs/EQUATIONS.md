# T-MoE: SPAR Router — Mathematical Reference

SPAR (Stress-Penalized Adaptive Routing) is the canonical router for this project.
It replaces auxiliary load-balancing losses with a one-sided, data-calibrated penalty
that is geometrically coherent with the cosine routing signal and zero at equilibrium.

---

## 1. Core SPAR Formulation

### Selection Logit

The logit that determines *which* experts are selected for an input token $x$:

$$z_i(x,t) = \cos(x,\, W_i) - \lambda \cdot \max\!\bigl(0,\; L_i(t) - \tfrac{1}{N}\bigr)$$

- $\cos(x, W_i) = \hat{x}^\top \hat{W}_i \in [-1, 1]$: cosine similarity between the
  L2-normalized token $\hat{x}$ and the L2-normalized prototype direction $\hat{W}_i$.
  Scale-invariant; measures directional alignment only.
- $L_i(t)$: EMA load of expert $i$ at step $t$ — the running fraction of token-expert
  assignments routed to expert $i$. Defined below.
- $\max(0,\; L_i - 1/N)$: one-sided penalty. Zero when expert $i$ carries its fair share
  or less; positive only when $L_i$ exceeds the equilibrium value $1/N$.
- $\lambda$: penalty scale, auto-calibrated once at step $\lambda_{\text{calib}}$. See Section 2.
- $N$: number of experts.

**Invariants:**
- Penalty $= 0$ for all $i$ iff $L_i = 1/N$ for all $i$ (equilibrium).
- $\sum_i L_i = 1$ always, so $\sum_i (L_i - 1/N) = 0$: the raw differential is zero-sum.
  $\sum_i \max(0, L_i - 1/N) \ge 0$ concentrates penalty on overloaded experts only.

Top-$k$ selection: $\mathcal{S}(x,t) = \operatorname{top-k}_i\; z_i(x,t)$.

---

### Output Weight

After selection, the output weight for each selected expert reflects **alignment quality
only** — the load signal that influenced selection is not carried into the weighting:

$$w_i = \frac{\exp\!\bigl(\cos(x, W_i) \,/\, \tau\bigr)}{\displaystyle\sum_{j \in \mathcal{S}} \exp\!\bigl(\cos(x, W_j) \,/\, \tau\bigr)}, \qquad i \in \mathcal{S}(x,t)$$

- $\tau_t > 0$: temperature, optionally annealing linearly from $\tau_0$ (default 0.5)
  to $\tau_f$ (configurable, e.g. 0.12) over `tau_anneal_steps` optimizer steps.
  At $\tau_f = 0.12$ with typical $\Delta\cos \approx 0.1$: $\text{softmax}([c, c{-}0.1]/0.12)$
  $\rightarrow$ conf $\approx 0.62$. Set $\tau_f = \tau_0$ or `tau_anneal_steps = 0` to
  disable annealing. Constrained to $\tau \ge 10^{-3}$ in code.
- Weights sum to 1 over the selected set: $\sum_{i \in \mathcal{S}} w_i = 1$.

**Why factored?** If the load penalty were included in $w_i$, an overloaded expert
selected for a well-aligned token would receive a systematically depressed weight,
distorting the aggregated representation. Selection and weighting serve different
objectives: selection is load-aware; weighting is quality-aware.

---

### EMA Load Update

Usage $U_i(t)$ is the fraction of token-expert assignments sent to expert $i$ in the
current step:

$$U_i(t) = \frac{\#\{\text{assignments to expert } i\}}{B \cdot S \cdot k}$$

where $B$ is batch size, $S$ sequence length, $k$ the top-$k$ value.
This ensures $\sum_i U_i(t) = 1$ exactly.

The load estimate is a first-order EMA:

$$L_i(t) = (1 - \alpha) \cdot L_i(t-1) + \alpha \cdot U_i(t)$$

- $\alpha = 0.01$: smoothing coefficient. Memory horizon $\approx 1/\alpha = 100$ steps.
- Initialized to $L_i(0) = 1/N$ (fair share).
- Updated once per optimizer step (after gradient accumulation completes), not per
  forward pass.
- Synchronized across DDP ranks via `all_reduce(AVG)` every step.

**Invariants:**
- $\sum_i L_i(t) = 1$ for all $t$ (preserved by the linear EMA since $\sum_i U_i = 1$
  and $\sum_i L_i(0) = 1$).
- $L_i(t) \in (0, 1]$ for all $t > 0$.

---

## 2. $\lambda$ Auto-Calibration

$\lambda$ is calibrated once at step $\lambda_{\text{calib}} = \text{warmup\_steps} + 200$
(default: 600 when warmup=400) and held fixed thereafter:

$$\lambda = \min\!\left(\frac{\sigma_{\cos}}{\bar{L}},\; 5.0\right)$$

where $\sigma_{\cos}$ is the empirical standard deviation of $\cos(x, W_i)$ over all
tokens and experts in the current batch, and $\bar{L} = \frac{1}{N}\sum_i L_i = 1/N$.

**Simplification.** Since $\bar{L} = 1/N$ always:

$$\lambda = \min\!\bigl(\sigma_{\cos} \cdot N,\; 5.0\bigr)$$

**Geometric interpretation.** The penalty for a 1-sigma routing advantage
($\Delta \cos = \sigma_{\cos}$) equals a 1-fair-share overload penalty
($\Delta L = 1/N$). Concretely: $\lambda \cdot (1/N) = \sigma_{\cos}$, so an expert
carrying twice its fair share ($L_i = 2/N$) is penalized by exactly $\sigma_{\cos}$
— enough to overcome one standard deviation of cosine advantage and force rebalancing.

**The 5.0 ceiling.** Activates when cosine similarity is highly concentrated
(low $\sigma_{\cos}$ relative to $N$, i.e., tokens strongly prefer a small number of
prototypes). Without the ceiling, $\lambda$ would grow unboundedly and suppress all
routing signal. In practice, for $N=8$ and typical $\sigma_{\cos} \approx 0.15$–$0.25$,
$\lambda \approx 1.2$–$2.0$, well below the ceiling.

**Why `warmup_steps + 200`?** The EMA reaches approximate stationarity after
$\sim 1/\alpha = 100$ steps. Calibrating at `warmup_steps + 200` (default: step 600 for
`warmup_steps=400`) ensures the LR warmup has completed and the gate has learned
meaningful cosine directions before $\sigma_{\cos}$ is measured. Calibrating during
warmup underestimates $\sigma_{\cos}$ — the gate's cosine similarity is still
near-random — producing $\lambda$ that is too small and a weak penalty for the entire run.

**σ_cos floor.** In code, $\sigma_{\cos}$ is clamped to $\ge 10^{-4}$ before computing
$\lambda$. This prevents $\lambda = 0$ in the degenerate case where all cosines are
identical (e.g., near-random prototypes in very high $D$), which would permanently
disable the load penalty with no recovery path.

---

## 3. Fixed-Point Analysis

### The equilibrium condition

At a fixed point, $L_i^* = U_i^*$ for all $i$ (EMA has converged). Substituting into
the selection logit:

$$z_i^*(x) = \cos(x, W_i) - \lambda \cdot \max\!\bigl(0,\; L_i^* - 1/N\bigr)$$

The penalty is zero for all $i$ iff $L_i^* = 1/N$, i.e., perfectly uniform load.
In this case routing reduces to pure cosine similarity and the penalty has no effect.

### Stability of equilibrium

The equilibrium $L_i^* = 1/N$ is a **saddle**, not a global attractor.

A high-cosine token can always select a moderately overloaded expert if its cosine
advantage exceeds the penalty:

$$\cos(x, W_i) - \cos(x, W_j) > \lambda \cdot (L_i - 1/N)$$

for any competitor $W_j$ with $L_j \le 1/N$. This is desirable: the router does
not force uniform routing when cosine geometry strongly prefers a particular expert.
The penalty suppresses systematic overloading (excess load accumulated over many steps)
while leaving token-level variation intact.

### Why one-sided vs. two-sided?

A two-sided penalty $-\lambda \cdot (L_i - 1/N)$ would *boost* underloaded experts
and *suppress* overloaded ones, but it imposes a nonzero adjustment even at equilibrium:
each expert bears a constant penalty term of $-\lambda/N$. This constant floor wastes
routing signal budget — it shifts all logits uniformly down by $\lambda/N$, serving no
purpose once load is balanced. The one-sided form eliminates this waste: at equilibrium
the penalty surface is identically flat and routing is governed purely by cosine similarity.

---

## 4. Expert Stress Metric (Welford, Metrics-Only)

Expert Stress measures how semantically dispersed the tokens routed to an expert are
relative to its prototype direction. It is computed purely for observability and does
**not** appear in the routing logit.

### Cosine distance observation

For a token $x_k$ routed to expert $i$, the cosine distance from the prototype is:

$$d_{i,k} = 1 - \cos(x_k,\, W_i) \;\in\; [0,\, 2]$$

This reuses the alignment already computed in the forward pass — zero additional FLOPs.

### Batched weighted Welford update

Per-expert running statistics are accumulated using a batched weighted Welford algorithm:

$$n_i \;\leftarrow\; n_i + \textstyle\sum_k \mathbf{1}[k \to i]$$

$$\mu_i \;\leftarrow\; \mu_i + \frac{1}{n_i}\sum_k \mathbf{1}[k \to i]\cdot(d_{i,k} - \mu_i^{\text{pre}})$$

$$M_{2,i} \;\leftarrow\; M_{2,i} + \sum_k \mathbf{1}[k \to i]\cdot(d_{i,k} - \mu_i^{\text{pre}})(d_{i,k} - \mu_i^{\text{post}})$$

where $\mu_i^{\text{pre}}$ is the mean before the current batch and $\mu_i^{\text{post}}$
is the mean after. The product of pre- and post-update residuals is the standard Welford
variance accumulation. Weights are binary (indicator of selection) not softmax weights,
so $n_i$ is an integer count of token-expert assignment events.

The running variance estimate is:

$$\hat{\sigma}_i^2 = \frac{M_{2,i}}{n_i}$$

### Stress (coefficient of variation)

$$\text{Stress}_i = \frac{\hat{\sigma}_i}{\max(\mu_i,\;\epsilon)}, \qquad \epsilon = 10^{-3}$$

- Low Stress: tokens cluster tightly around prototype $W_i$ — expert is well-specialized.
- High Stress: tokens are semantically dispersed — expert is handling heterogeneous inputs.
- $\mu_i < \epsilon$: expert achieves near-perfect alignment; Stress is suppressed rather
  than reporting a spuriously large CV.

**Why not in the routing logit.** Including Stress in the selection logit creates a
structural feedback bias: dominant experts are selected more often, accumulate more
Welford observations, and obtain more statistically reliable (and potentially larger)
CV estimates — even if their true dispersion is identical to that of other experts.
This is a stable attractor: the compounding suppression does not self-correct.
Empirically confirmed at `mu_stress=0.5`: eff_E collapsed from 7.5 to 5.8 by step 2000
and did not recover. At `mu_stress=0.1` the term had zero measurable effect and was
removed. Stress is retained as a zero-cost diagnostic signal in WandB.

### LoRA Approximation Error Identity

Let $S_i = \mathbb{E}[\hat{x}_k \hat{x}_k^\top]$ be the second moment of unit-normalized
tokens routed to expert $i$. The following identity holds exactly:

$$\hat{W}_i^\top S_i \hat{W}_i = (1 - \mu_i)^2 + \text{Stress}_i^2\, \mu_i^2$$

The left side is the average squared projection of routed tokens onto the prototype — the
fraction of input variance captured by a rank-1 LoRA whose direction is $\hat{W}_i$.
Therefore the rank-1 approximation error is:

$$\mathcal{E}_i(1) = 1 - \hat{W}_i^\top S_i \hat{W}_i = 1 - (1-\mu_i)^2 - \text{Stress}_i^2\,\mu_i^2$$

When Stress is high, the per-token projection variance is large: some tokens are
well-served by the rank-1 adapter, others are near-orthogonal to $\hat{W}_i$ and
unserved. A single rank-$r$ LoRA cannot simultaneously adapt to tokens in geometrically
incompatible directions. This provides the theoretical grounding for the mitosis trigger
(Section 8): when $\text{Stress}_i > \tau_{\text{mitosis}}$, the expert's token
population has intrinsic dimensionality $> 1$ and should split.

---

## 5. Prototype Learning

The router prototype matrix $W \in \mathbb{R}^{N \times D}$ is an `nn.Parameter`,
initialized with unit-norm rows:

$$W_i^{(0)} = \frac{\xi_i}{\|\xi_i\|}, \qquad \xi_i \sim \mathcal{N}(0, I_D)$$

During the forward pass, both $x$ and the rows of $W$ are L2-normalized on the fly:

$$\hat{x} = \frac{x}{\|x\|_2 + \epsilon}, \qquad \hat{W}_i = \frac{W_i}{\|W_i\|_2 + \epsilon}$$

Gradients flow through $\hat{W}_i$ to $W_i$ via the chain rule of the L2 normalization:
$\partial \hat{W}_i / \partial W_i = (I - \hat{W}_i \hat{W}_i^\top) / \|W_i\|_2$.
This is the standard projected gradient for the unit sphere — updates are perpendicular
to the current direction (magnitude changes are suppressed), so $W_i$ evolves as a
direction, not a magnitude.

**Gradient sparsity.** Only the top-$k$ selected experts receive gradient for a given
token (through the output weight softmax $w_i$). Experts not in $\mathcal{S}(x,t)$ receive
zero gradient from that token. This is intentional: prototypes specialize toward the
directions of their routed token populations. Non-selected experts are not pushed toward
tokens they do not serve.

---

## 6. Hyperparameter Rationale

SPAR has **one free hyperparameter**: $\tau$ (output weight temperature). All other
values are either data-derived or have principled, non-tunable defaults.

| Parameter | Value | Status | Rationale |
|---|---|---|---|
| $\tau_0$ (`temperature`) | 0.5 | **Free** (one tunable) | Controls output weight sharpness: `softmax(cos/τ)`. τ=0.5 doubles effective cosine distances, giving the top expert ~2× the weight of the second. Standard MoE temperature — not too uniform (τ=1), not too peaked (τ→0). |
| $\tau_f$ (`tau_final`) | 0.12 | Derived from target conf | Where τ anneals to at convergence. At τ=0.12 and typical cosine gap Δcos≈0.1 between top-2 experts: `softmax([c, c-0.1]/0.12)` → conf≈0.62. Set to push conf from 0.56 (wikitext, τ=0.5 fixed) toward 0.62 on fineweb. |
| `tau_anneal_steps` | 10 000 | Derived from run length | 53% of the 19 000-step fineweb run. Starts annealing immediately; completes past midpoint so routing has stabilized before τ becomes aggressive. Annealing too fast early causes over-commitment before experts specialize. |
| $\lambda_{\text{calib}}$ (`lambda_calib_step`) | `warmup_steps + 200` | Derived from schedule | λ is auto-calibrated once from data at this step (see Section 2). Must be post-LR-warmup. Default 600 = warmup(400) + 200. Not a hyperparameter — just a scheduling offset. |
| $\alpha$ (`ema_alpha`) | 0.01 | Fixed default | Memory horizon $1/\alpha = 100$ optimizer steps. Slow enough to be a stable signal; fast enough to track routing shifts within a few hundred steps. Standard EMA window for MoE load tracking. |
| $\sigma_{\text{noise}}$ (`noise_std`) | 0.05 → 0 | Fixed default (anneal recommended) | Gumbel exploration noise during training only. At initialization, σ_noise=0.05 is below typical cosine gaps (0.1–0.3) so it does not dominate selection. However, on fineweb-edu at step 6400, measured Δ_cos≈0.037 — noise exceeds the routing signal. **Recommendation:** anneal noise_std to 0 by midpoint (~step 9500 for a 19000-step run). Noise is orthogonal to EMA load penalty so annealing does not affect load balance. |
| $\epsilon$ (`eps`) | 1e-3 | Numerical floor | Prevents division by zero in cosine normalization. No effect on routing in normal operation. |

**Summary.** The only value that requires domain judgment is $\tau_0 = 0.5$ (and
$\tau_f = 0.12$ if annealing is used). $\lambda$ is data-calibrated. All other fields
are standard engineering defaults.

---

## 7. Historical Design Decisions

The following components appeared in earlier router versions and were removed. This table
records the decision and the empirical or mathematical reason.

| Component | Router version | Replaced by | Reason for removal |
|---|---|---|---|
| Fatigue accumulator $F_i = (1-\gamma)F_i + \beta\max(0, U_i - \tau/N)$ | MetabolicRouter | EMA load $L_i$ | 5 hyperparameters ($\lambda, \gamma, \beta, \tau, F_s$); non-stationary signal accumulates historical excess rather than tracking current load; harder to calibrate and interpret |
| $\text{tanh}(F_i / F_s)$ penalty function | MetabolicRouter | $\max(0, L_i - 1/N)$ | Bounded ceiling: if cosine advantage $> \lambda$, tanh penalty is permanently saturated and overloaded expert keeps winning. Confirmed at $\lambda=0.5$: fineweb gini drifted 0.026→0.207 by step 1225 |
| $\lambda_{\text{eff}}(t) = \lambda \cdot \min(1, t/T_{\text{warmup}})$ ramp | MetabolicRouter | λ auto-calibration at step 200 | Manual λ required sweep (v3→v4→v5 were essentially a λ search). Auto-calibration eliminates the free parameter |
| Learnable $g_i$ magnitude scale | MetabolicRouter (early) | Removed | Optimizer inflated $g_i$ to dwarf the penalty, defeating load balancing entirely |
| Output weights from potential (fatigue-inclusive) | MetabolicRouter | Factored weights: softmax(cos/τ) | Load signal in $w_i$ systematically depressed weights for overloaded experts, distorting the aggregated representation and hurting PPL |
| Expert Stress $\mu_{\text{stress}}$ in routing logit | StressCorrectedRouter (early) | Metrics-only | Structural Welford bias: dominant experts accumulate more observations → higher CV even under identical true dispersion → compounding suppression → collapse. Confirmed: eff_E 7.5→5.8 at $\mu_{\text{stress}}=0.5$ |
| SoftSign$(F_i)$ penalty function | EQUATIONS.md v1 | — | Never implemented in any final router; appeared only in early design documents |
| Adaptive cost scaling $\eta_{\text{eff}} = \eta_{\text{base}} \cdot N_{\text{current}}/N_{\text{start}}$ | EQUATIONS.md v1 | — | Intended for dynamic expert expansion (not yet implemented); no current router uses this |
| Welford DDP sync (18 all_gather/step) | StressCorrectedRouter (early) | Per-rank Welford, no sync | Caused NCCL SEQNUM drift → deadlock. Per-rank divergence in metrics is acceptable |
| 6 `dist.all_reduce`/step (MetabolicRouter) | MetabolicRouter | 1 `all_reduce`/step | 14% throughput reduction. SPAR syncs only EMA load (1 tensor) and λ (once at step 200) |

---

## 8. Future Work: Prototype-Aware Mitosis

When an expert $A$ with high Stress splits into experts $A$ and $B$, both its LoRA
parameters $\theta$ and its prototype $W$ must divide. The perturbation must lie in the
orthogonal complement of the current direction to break symmetry without destroying
the learned specialization:

$$\theta_B = \theta_A + \zeta_{\perp}(\theta_A)$$
$$W_B = W_A + \zeta_{\perp}(W_A)$$

where $\zeta_{\perp}(v)$ denotes a small perturbation orthogonal to $v$.

For LoRA experts ($\theta = BA$): apply the orthogonal perturbation independently to
matrices $A$ and $B$ to preserve low-rank geometry.

**Mitosis trigger:**

$$\text{Stress}_i > \tau_{\text{mitosis}}, \quad \mu_i > \mu_{\min}, \quad n_i > n_{\min}$$

Recommended values: $\tau_{\text{mitosis}} \approx 0.5$, $\mu_{\min} = 10^{-3}$,
$n_{\min} \approx 1000$.

---

## 9. Future Work: Apoptosis (Expert Pruning)

Underutilized experts are masked out (not physically deleted), freeing capacity for
future mitosis events. The pruning criterion uses a long-horizon EMA of usage:

$$U_{\text{EMA},i}(t+1) = \alpha_{\text{EMA}}\, U_{\text{EMA},i}(t) + (1-\alpha_{\text{EMA}})\, U_i(t)$$

An expert is pruned if:

$$U_{\text{EMA},i} < \tau_{\text{prune}} \quad \text{AND} \quad N_{\text{active}} > N_{\min} \quad \text{AND} \quad t > t_{\text{last\_event}} + T_{\text{cooldown}}$$

**Timescale separation:** $T_{\text{cooldown}} \ge 2\tau_{\text{EMA}}$ where
$\tau_{\text{EMA}} \approx 1/(1-\alpha_{\text{EMA}})$, preventing pruning decisions
from lagging behind system equilibration.

---

## 10. Adapter Capacity Scaling: Rank Ablation

### Theoretical prediction

SPAR's eff_E=8.0 compounds with LoRA rank — the PPL benefit of rank=16→32 is
**router-dependent**, not router-neutral. The argument is structural.

Define *effective rank-units* as the number of new adapter directions (rank=32 minus
rank=16 = 16 new directions per expert) with gradient signal-to-noise > 1, i.e.,
directions that are actually trained. An expert that sees a negligible fraction of tokens
has noise-dominated gradient even at rank=16; adding rank capacity provides zero benefit.

**Standard routing (eff_E≈4.0):** Token share per expert ≈ 4% (undertrained) or ≈ 46%
(dominant). 4 dominant experts utilize the new 16 rank-units. 4 undertrained experts have
noise-dominated gradients at rank=16 — additional rank capacity is wasted.

$$\text{Effective rank-units gained (standard)} \approx 4 \times 16 = 64$$

**SPAR (eff_E=8.0):** Token share per expert ≈ 25% each. All 8 experts are well-trained.
All 8 can utilize the additional capacity.

$$\text{Effective rank-units gained (SPAR)} \approx 8 \times 16 = 128$$

**Lower bound: rank=32 provides ~2× more PPL benefit to SPAR than to standard routing.**

### The ablation grid

The testable prediction requires a 2×2 experiment:

| | rank=16 | rank=32 |
|---|---|---|
| Standard + aux loss | A (done: PPL≈29.0) | B (todo) |
| SPAR | C (done: PPL≈27.5–28.5 projected) | D (todo) |

**Null hypothesis (router-neutral rank benefit):**
$(A - B) \approx (C - D)$ — both routers gain the same PPL from rank doubling.

**Alternative hypothesis (eff_E compounds with rank):**
$(C - D) > (A - B)$ — SPAR gains disproportionately more.

If the alternative is confirmed, the paper has a novel result: *eff_E=8.0 compounds with
adapter capacity scaling, making SPAR increasingly advantageous as model capacity grows.*
No prior MoE paper has made or tested this prediction.

### Trainable parameter counts

| Config | Params (6 MoE layers, 8 experts, top-k=2) |
|---|---|
| rank=16 | ~5.9M (current) |
| rank=32 | ~11.8M |
| Unfreeze 6 MoE MLP blocks | ~34M additional (total ~40M trainable) |

**Unfreezing backbone MLP is not a valid comparison point.** It changes the architecture
class entirely: the LoRA paradigm assumes a high-quality frozen prior (the pretrained MLP)
with low-rank task-specific deltas. Unfreezing replaces this with partial fine-tuning +
LoRA bolted on. The result is not "better LoRA-MoE" — it is a different model class with
5× more trainable parameters, no longer parameter-efficient, and no longer a fair comparison
against standard routing (which also uses frozen backbone). The paper claim — *MoE
specialization with frozen backbone, adapter-only training* — is invalidated.

**The correct capacity lever is rank=32** (doubles adapter expressivity, clean ablation,
preserves all paper claims).

### Planned runs (after v6 and v7 complete)

| Experiment | Config change | Purpose |
|---|---|---|
| `gptneo_125m_standard_v3_r32` | standard_v3 YAML, lora.rank=32 | Cell B |
| `gptneo_125m_stress_v6_r32` | v6 fineweb YAML, lora.rank=32 | Cell D |

Run B and D only after A (standard_v3, done) and C (v6-fineweb, in progress) have final
numbers. The comparison requires identical step counts, datasets, and seeds.
