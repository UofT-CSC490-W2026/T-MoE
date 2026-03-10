# T-MoE Project: Mathematical Formulas & Deep Learning Techniques

This document synthesize the core technical foundations of the T-MoE (Thermodynamic Mixture-of-Experts) project, covering routing dynamics, elastic expansion, and reinforcement learning-based optimization.

---

## 1. Core Routing: Metabolic Dynamics

T-MoE replaces traditional auxiliary load-balancing losses with biological-inspired fatigue mechanics.

### Equation 1: Homeostatic Routing Potential
The potential $z_i$ of an expert $i$ to be selected for an input $x$ at time $t$:
$$z_i(x,t)= g_i \cdot \cos(x, W_i) - \lambda_{\text{eff}}(t) \cdot \mathrm{SoftSign}\!\left(F_i(t)\right) - \mu \cdot \mathrm{Dist}(i)$$

where:
$$\mathrm{SoftSign}(x) = \frac{x}{1 + |x|}$$

$$\lambda_{\text{eff}}(t) = \lambda \cdot \min\!\left(1,\; \frac{t}{T_{\text{warmup}}}\right)$$

- $g_i$: Learnable expert importance scale (magnitude from weight norm).
- $\cos(x, W_i)$: Semantic alignment (cosine similarity with expert direction $W_i$).
- $\lambda_{\text{eff}}(t)$: **Warmed-up** metabolic penalty strength. Ramps from $0 \to \lambda$
  over $T_{\text{warmup}}$ steps, allowing the gate to converge to good prototype directions
  via task loss alone before fatigue feedback begins steering it. Without this warmup the
  gate can lock into a biased routing pattern before fatigue has built up enough to correct it.
  At eval time $\lambda_{\text{eff}} = \lambda$ (no warmup applied).
- $\mu \cdot \mathrm{Dist}(i)$: Silicon Tax (distance penalty).

### Equation 2: Differential Fatigue Dynamics
Fatigue $F_i$ tracks **excess compute load** relative to fair share and recovers over time:
$$F_i(t+1) = (1-\gamma) F_i(t) + \eta_i(t) \cdot \left(U_i(t) - \frac{1}{N}\right), \qquad 0 < \gamma < 1$$
where:
$$\eta_i(t) = \eta_{eff} \cdot \min\!\left(1.0, \frac{t - \text{birth\_step}\_i}{T\_{\text{warmup}}}\right)$$
- $T_{\text{warmup}} > 0$
- $\gamma$: Recovery rate (homeostatic return).
- $\eta_{eff}$: Effective activation cost (see Equation 3).
- $U_i(t)$: **Token-count fraction** for expert $i$ — computed with uniform weight $1/k$
  per routing slot, so $U_i = (\text{slots assigned to } i) / (\text{total tokens} \cdot k) \cdot k
  = (\text{slots assigned to } i) / \text{total tokens}$.
  This ensures $\sum_i U_i = 1$ and fair share $= 1/N$ exactly.
  *v1–v2 used softmax routing probabilities for $U_i$, which is a noisy proxy for compute
  load (small gate-logit changes cause large $U_i$ swings). Token-count usage is stable
  across logit scales and correctly measures how often each expert is invoked.*
- $N$: Number of active experts.

**Zero-sum invariant**: $\sum_i (U_i - 1/N) = 0$ always, so $\sum_i F_i(t) = 0$ for all $t$
(given $F_i(0)=0$). Mean fatigue is identically zero — confirmed empirically to machine
precision ($|\bar{F}| < 10^{-8}$).

**Centering at zero**: Balanced experts converge to $F \approx 0$ (no penalty),
overloaded experts accumulate positive fatigue (SoftSign penalty), and neglected experts
accumulate negative fatigue (SoftSign bonus). This places the system in SoftSign's
maximally responsive zone (gradient $= 1$ at $x=0$, saturating to $\pm\lambda$ for large $|F|$).

### Equation 3: Adaptive Cost Scaling
To maintain stability during expert expansion, the activation cost $\eta$ scales with the number of active experts $N$:

$$\eta_{eff} = \eta_{base} \cdot \frac{N_{current}}{N_{start}}$$


---

## 2. Expert Stress: Prototype Semantic Dispersion

We define a per-expert **Stress** metric that measures how semantically
dispersed the tokens routed to each expert are, relative to the expert's own
prototype direction $W_i$. This complements the balance metrics (Gini,
effective\_E) and specialisation metrics (MI) with a forward-pass-local,
zero-overhead signal that is geometrically coherent with both the routing
mechanism (Equation 1) and the mitosis step (Equation 5).

### Equation 4: Expert Stress via Prototype Cosine Dispersion

Let $x_k$ be a token routed to expert $i$ with routing weight $w_{i,k}$
(uniform $1/k$ per slot, matching the usage convention of Equation 2).
The observation for each token is its **cosine distance from the expert
prototype**:

$$d_{i,k} = 1 - \cos(x_k,\, W_i)$$

This quantity is already computed in `compute_alignment()` as the normalised
alignment score — $d_{i,k} = 1 - \text{alignment}_{k,i}$ — and costs zero
additional FLOPs. *Requires `normalize_inputs=True` and
`normalize_weights=True` (both default in v4 config) so that the alignment is
a true cosine similarity in $[-1, 1]$ and $d_{i,k} \in [0, 2]$.*

We accumulate per-expert running statistics using a **batched weighted Welford
algorithm** (West, 1979). Within each forward pass over the batch:

$$n_i \;\leftarrow\; n_i + \textstyle\sum_k w_{i,k}$$
$$\delta_k \;=\; d_{i,k} - \mu_i \qquad \text{(residual against pre-batch mean)}$$
$$\mu_i \;\leftarrow\; \mu_i + \frac{1}{n_i}\sum_k w_{i,k}\,\delta_k$$
$$M_{2,i} \;\leftarrow\; M_{2,i} + \sum_k w_{i,k}\,\delta_k\,\bigl(d_{i,k} - \mu_i\bigr)$$

The last line uses the **post-batch** $\mu_i$ to form the second residual,
which is the standard Welford product (pre-update $\times$ post-update
residual). Applied over a batch rather than sequentially, this is an
$O(n_{\text{batch}}/n_{\text{total}})$ approximation to exact sequential
Welford — negligible error after the first few hundred steps.

The dimensionless **Stress** is the coefficient of variation of cosine
distance:

$$\boxed{\text{Stress}_i = \frac{\sqrt{M_{2,i}\,/\,n_i}}{\max(\mu_i,\;\epsilon)}}$$

where $\epsilon$ is a **physically meaningful floor**, not machine epsilon.
In $d$-dimensional hidden space, two random unit vectors have expected cosine
distance 1 with standard deviation $\approx 1/\sqrt{d}$. For $d=768$,
$1/\sqrt{d} \approx 0.036$. Setting $\epsilon = 10^{-3}$ is therefore
conservative: it only activates when $\mu_i$ falls below a distance that
is geometrically negligible in the token representation space.

*Do not set $\epsilon$ to machine epsilon ($\sim 10^{-8}$).* If an expert
achieves near-perfect alignment ($\mu_i \to 0$), a tiny residual variance
divided by a microscopic denominator will spike Stress artificially. The
correct interpretation of $\mu_i < \epsilon$ is not "infinite stress" but
"the expert is so well-aligned that the CV is undefined — suppress the
metric." Use $\epsilon = 10^{-3}$ in practice.

**Interpretation:**
- $\mu_i$: mean cosine distance of tokens from prototype $W_i$ — measures how
  far the average routed token is from the expert's specialisation direction.
- $\text{Stress}_i$ low: tokens cluster tightly around $W_i$ → expert is
  well-specialised, its prototype accurately represents its token population.
- $\text{Stress}_i$ high: tokens are semantically dispersed around $W_i$ →
  expert is handling heterogeneous token types → prototype should split
  (Equation 5).
- $\mu_i < \epsilon$: expert is near-perfectly aligned → Stress is suppressed
  (set to 0) → mitosis is not triggered regardless of variance.

---

**Connection to LoRA Approximation Error**

Let $S_i = \mathbb{E}_{w_i}[\hat{x}_k \hat{x}_k^\top]$ be the second moment
matrix of unit-normalised tokens routed to expert $i$. From the definitions of
$\mu_i$ and $\mathrm{Stress}_i$, the following identity holds exactly:

$$\hat{W}_i^\top S_i \hat{W}_i \;=\; (1-\mu_i)^2 + \mathrm{Stress}_i^2\,\mu_i^2$$

The left-hand side is the **average squared projection** of routed tokens onto
the prototype direction — equivalently, the fraction of input variance captured
by a rank-1 LoRA whose sole direction is $\hat{W}_i$. Therefore the rank-1
LoRA approximation error along $\hat{W}_i$ is:

$$\mathcal{E}_i(1) = 1 - \hat{W}_i^\top S_i \hat{W}_i \;=\; 1 - (1-\mu_i)^2 - \mathrm{Stress}_i^2\,\mu_i^2$$

This identity has a structural consequence for low-rank adaptation. Expand the
per-token squared projection as:

$$(\hat{x}_k^\top \hat{W}_i)^2 = (1 - d_{i,k})^2$$

With $d_{i,k} = \mu_i + \epsilon_k$ where $\epsilon_k$ is the zero-mean
deviation, and $\mathrm{Var}(\epsilon_k) = \mathrm{Stress}_i^2\mu_i^2$:

- When $\mathrm{Stress}_i \approx 0$: all tokens have $d_{i,k} \approx \mu_i$,
  so all per-token projections are approximately equal
  $(1-\mu_i)^2$. The LoRA serves every token in its population uniformly.
  If $\mu_i$ is also small (tokens close to $\hat{W}_i$), projection is near 1
  — the rank-1 adapter captures nearly all the input variation.

- When $\mathrm{Stress}_i$ is large: the distribution of $(1-d_{i,k})^2$ has
  high variance. Some tokens are near $\hat{W}_i$ (projection $\approx 1$,
  well-served) and others are near-orthogonal (projection $\approx 0$,
  unserved). The average capture $\hat{W}_i^\top S_i \hat{W}_i$ may remain
  moderate, but the *per-token coverage is highly uneven*. A single rank-$r$
  LoRA cannot simultaneously adapt to tokens in incompatible directions — the
  expert is trying to serve two (or more) geometrically distinct token
  populations with one low-rank adapter.

**The qualitative bound:** For a LoRA of rank $r$, efficient adaptation
requires the token distribution $\mathcal{D}_i$ to have intrinsic
dimensionality $\lesssim r$. Stress is a proxy for this intrinsic
dimensionality — concentrated routing (low Stress) implies approximately
rank-1 structure; dispersed routing (high Stress) implies the distribution
spans multiple independent directions, requiring rank $> 1$ to serve all
token subgroups. This provides the theoretical grounding for the mitosis
trigger: when $\mathrm{Stress}_i > \tau$, the single LoRA expert should
split into two experts, each inheriting a geometrically coherent subset of
$\mathcal{D}_i$.

*Note:* The identity above holds at all times. The interpretation in terms
of $\lambda_1(S_i)$ (the leading eigenvalue of $S_i$) assumes $\hat{W}_i$
aligns with the top eigenvector of $S_i$, which holds approximately at
convergence for the metabolic router since its gate is trained to maximise
cosine alignment with routed tokens.

---

**Why cosine distance, not task loss:**
Task-loss-based stress conflates model quality with routing quality — high loss
early in training inflates all experts' stress regardless of routing coherence.
Cosine-distance stress is geometrically local: it measures routing coherence
in the same space the gate operates in, independent of training stage or model
quality. It is also self-contained in the router forward pass, requiring no
information from the training loop.

**Why token-level, not batch-level observations:**
With batch=16, seq=512, top\_k=2, num\_experts=8: this formulation yields
$\approx$2048 observations per expert per batch, versus 1 observation per
expert per batch for a loss-based formulation. The Welford estimate is
statistically reliable after a single step.

**Complementarity with existing metrics:**

| Metric | Measures | Loss needed? | Obs/step/expert |
|---|---|---|---|
| Gini / effective\_E | Load balance | No | — |
| MI (GlobalSpecializationTracker) | Token-type routing consistency | No | — |
| **Stress (this)** | **Prototype semantic dispersion** | **No** | **~2048** |

Random routing achieves low Gini (balanced) but high Stress (semantically
incoherent). The metabolic router should achieve low Gini *and* low Stress —
routing that is both balanced and geometrically coherent.

**Use as mitosis trigger (future work):**
$$\text{Stress}_i > \tau_{\text{mitosis}}, \qquad \mu_i > \mu_{\min}, \qquad n_i > n_{\min}$$

The $\mu_i > \mu_{\min}$ guard prevents triggering mitosis on a near-perfectly
specialised expert whose Stress is undefined rather than genuinely high.
Recommended: $\mu_{\min} = \epsilon = 10^{-3}$, $\tau_{\text{mitosis}} \approx 0.5$,
$n_{\min} \approx 1000$ (accumulate enough observations before deciding).

### Equation 5: Prototype-Aware Mitosis (Future Work)
When an expert $A$ splits into $A$ and $B$, both its parameters $\theta$ and its router prototype $W$ must divide.

$$\theta_B = \theta_A + \zeta_{\perp}(\theta_A)$$
$$W_B = W_A + \zeta_{\perp}(W_A)$$

*Note for LoRA experts ($\theta = B A$): apply orthogonal perturbation independently to matrices $A$ and $B$ to preserve low-rank geometry.*

---

## 3. Pruning and Consolidation (Future Work)

### Equation 6: Apoptosis (Pruning)
Underutilized experts are masked out (not physically deleted), saving compute and freeing slots for future Mitosis.

$$U_{\text{EMA},i}(t+1) = \alpha_{\text{EMA}} U_{\text{EMA},i}(t) + (1-\alpha_{\text{EMA}}) U_i(t)$$

An expert is pruned if:
$$\text{Prune}(i) \iff U_{\text{EMA},i} < \tau_{\text{prune}} \quad \text{AND} \quad N_{\text{active}} > N_{\min} \quad \text{AND} \quad t > t_{\text{last\_event}} + T_{\text{cooldown}}$$

**Timescale Separation**: To prevent pruning decisions from lagging system equilibrium, we ensure $T_{\text{cooldown}} \ge 2\tau_{\mathrm{EMA}}$, where $\tau_{\mathrm{EMA}} \approx 1/(1-\alpha_{\mathrm{EMA}})$.
