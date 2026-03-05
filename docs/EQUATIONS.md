# T-MoE Project: Mathematical Formulas & Deep Learning Techniques

This document synthesize the core technical foundations of the T-MoE (Thermodynamic Mixture-of-Experts) project, covering routing dynamics, elastic expansion, and reinforcement learning-based optimization.

---

## 1. Core Routing: Metabolic Dynamics

T-MoE replaces traditional auxiliary load-balancing losses with biological-inspired fatigue mechanics.

### Equation 1: Homeostatic Routing Potential
The potential $z_i$ of an expert $i$ to be selected for an input $x$ at time $t$:
$$z_i(x,t)= g_i \cdot \cos(x, W_i) - \lambda \cdot \mathrm{SoftSign}\!\left(F_i(t)\right) - \mu \cdot \mathrm{Dist}(i)$$

where:
$$\mathrm{SoftSign}(x) = \frac{x}{1 + |x|}$$

- $g_i$: Learnable expert importance scale (magnitude from weight norm).
- $\cos(x, W_i)$: Semantic alignment (cosine similarity with expert direction $W_i$).
- $\lambda F_i(t)$: Metabolic tax (fatigue penalty).
- $\mu \cdot \mathrm{Dist}(i)$: Silicon Tax (distance penalty).

### Equation 2: Differential Fatigue Dynamics
Fatigue $F_i$ tracks **excess usage** relative to fair share and recovers over time:
$$F_i(t+1) = (1-\gamma) F_i(t) + \eta_i(t) \cdot \left(U_i(t) - \frac{1}{N}\right), \qquad 0 < \gamma < 1$$
where:
$$\eta_i(t) = \eta_{eff} \cdot \min\!\left(1.0, \frac{t - \text{birth\_step}\_i}{T\_{\text{warmup}}}\right)$$
- $T_{\text{warmup}} > 0$
- $\gamma$: Recovery rate (homeostatic return).
- $\eta_{eff}$: Effective activation cost.
- $U_i(t)$: Routing weight (softmax) assigned to expert $i$ in Top-K routing.
- $N$: Number of active experts.

**Centering at zero**: By subtracting $1/N$, balanced experts converge to $F \approx 0$ (no penalty),
overloaded experts accumulate positive fatigue (SoftSign penalty), and neglected experts
accumulate negative fatigue (SoftSign bonus). This places the system in SoftSign's
maximally responsive zone (gradient highest at $x=0$).

### Equation 3: Adaptive Cost Scaling
To maintain stability during expert expansion, the activation cost $\eta$ scales with the number of active experts $N$:

$$\eta_{eff} = \eta_{base} \cdot \frac{N_{current}}{N_{start}}$$


---

## 2. Elastic Architecture: Stress & Mitosis

To allow the router to grow dynamically, we decouple load balancing (fatigue) from structural growth (stress).

### Equation 4: Stress (Mitosis Trigger) with Weighted Welford
Stress measures the Coefficient of Variation of the loss attributed to an expert.

Let $\hat{\ell}_i(t) = w_i(t) \cdot \mathcal{L}(t)$ be the loss attributed to expert $i$, where $w_i(t)$ is the routing weight. To account for fractional attribution, we use a weighted online Welford algorithm:

$$n_i \leftarrow n_i + w_i(t)$$
$$\delta_i = \hat{\ell}_i - \mu_i$$
$$\mu_i \leftarrow \mu_i + \frac{w_i(t)}{n_i} \delta_i$$
$$M_{2,i} \leftarrow M_{2,i} + w_i(t) \cdot \delta_i(\hat{\ell}_i - \mu_i)$$
$$\mathrm{Var}_i = \frac{M_{2,i}}{n_i}$$

The dimensionless **Stress** trigger is:
$$\text{Stress}_i = \frac{\sqrt{\mathrm{Var}_i}}{\max(|\mu_i|, \epsilon)} > \tau_{\text{mitosis}}, \qquad n_i > n_{\min}$$

### Equation 5: Prototype-Aware Mitosis
When an expert $A$ splits into $A$ and $B$, both its parameters $\theta$ and its router prototype $W$ must divide.

$$\theta_B = \theta_A + \zeta_{\perp}(\theta_A)$$
$$W_B = W_A + \zeta_{\perp}(W_A)$$

*Note for LoRA experts ($\theta = B A$): apply orthogonal perturbation independently to matrices $A$ and $B$ to preserve low-rank geometry.*

---

## 3. Pruning and Consolidation

### Equation 6: Apoptosis (Pruning)
Underutilized experts are masked out (not physically deleted), saving compute and freeing slots for future Mitosis.

$$U_{\text{EMA},i}(t+1) = \alpha_{\text{EMA}} U_{\text{EMA},i}(t) + (1-\alpha_{\text{EMA}}) U_i(t)$$

An expert is pruned if:
$$\text{Prune}(i) \iff U_{\text{EMA},i} < \tau_{\text{prune}} \quad \text{AND} \quad N_{\text{active}} > N_{\min} \quad \text{AND} \quad t > t_{\text{last\_event}} + T_{\text{cooldown}}$$

**Timescale Separation**: To prevent pruning decisions from lagging system equilibrium, we ensure $T_{\text{cooldown}} \ge 2\tau_{\mathrm{EMA}}$, where $\tau_{\mathrm{EMA}} \approx 1/(1-\alpha_{\mathrm{EMA}})$.
