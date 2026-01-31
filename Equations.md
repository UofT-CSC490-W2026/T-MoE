# T-MoE Project: Mathematical Formulas & Deep Learning Techniques

This document synthesize the core technical foundations of the T-MoE (Thermodynamic Mixture-of-Experts) project, covering routing dynamics, elastic expansion, and reinforcement learning-based optimization.

---

## 1. Core Routing: Metabolic Dynamics

T-MoE replaces traditional auxiliary load-balancing losses with biological-inspired fatigue mechanics.

### Equation 1: Homeostatic Routing Potential
The potential $z_i$ of an expert $i$ to be selected for an input $x$ at time $t$:
$$z_i(x,t)= \cos(x, W_i) - \lambda \cdot \mathrm{SoftSign}\!\left(F_i(t)\right) - \mu \cdot \mathrm{Dist}(i)$$

where:
$$\mathrm{SoftSign}(x) = \frac{x}{1 + |x|}$$

- $\cos(x, W_i)$: Semantic alignment (cosine similarity with expert prototype $W_i$).
- $\lambda F_i(t)$: Metabolic tax (fatigue penalty).
- $\mu \cdot \mathrm{Dist}(i)$: Silicon Tax (distance penalty).

### Equation 2: Age-Aware Fatigue Dynamics
Fatigue $F_i$ accumulates with usage and recovers over time:
$$F_i(t+1) = (1-\gamma) F_i(t) + \eta_i(t) \cdot U_i(t), \qquad 0 < \gamma < 1$$
where:
$$\eta_i(t) = \eta_{eff} \cdot \min\!\left(1.0, \frac{t - \text{birth_step}_i}{T_{\text{warmup}}}\right),
\qquad T_{\text{warmup}} > 0$$
- $\gamma$: Recovery rate (homeostatic return).
- $\eta_{eff}$: Effective activation cost.
- $U_i(t)$: Usage indicator (1 if selected, 0 otherwise), or "weighted case" in Top-K routing it represents the routing weight (softmax) assigned to expert $i$.

### Equation 3: Adaptive Cost Scaling
To maintain stability during expert expansion, the activation cost $\eta$ scales with the number of active experts $N$:
$$\eta_{eff} = \eta_{base} \cdot \frac{N_{current}}{N_{start}}$$

---

## 2. Elastic Architecture: Living Experts

The model dynamically grows and prunes its expert pool based on training pressure.

### Equation 4: Stress (Mitosis Trigger)
Mitosis (cloning) is triggered when an expert's **Stress** (loss variance) exceeds a threshold:
$$Stress_i = Var(Loss \mid Expert_i) = E[L^2] - (E[L])^2$$
- High variance indicates the expert is struggling to specialize across its assigned samples.

### Equation 5: Expert Cloning (Mitotic Fission)
When Expert $A$ divides into $A$ and $B$:
$$\theta_B = \theta_A + \mathcal{N}(0, \sigma^2)$$
- $\theta$: LoRA adapter parameters.
- $\sigma$: Mitosis noise (prevents identical "twin" experts).

### Equation 6: Usage Inheritance
To prevent newborns from being pruned immediately, load is shared:
$$U_{A\_new} = U_{B\_new} = \frac{U_{A\_old}}{2}$$

---

## 3. Pruning & Consolidation

### Equation 7: Apoptosis (Pruning)
Experts are killed if their Exponential Moving Average (EMA) usage falls below a threshold $\tau$:
$$U_{EMA}(t+1) = \alpha U_{EMA}(t) + (1-\alpha) U_{current} < \tau$$

### Equation 8: Fusion (Entropic Merging)
Redundant experts are merged based on the cosine similarity of their **Specialty Prototypes** $P$:
$$sim(Expert_i, Expert_j) = \frac{P_i \cdot P_j}{\|P_i\| \|P_j\|} > \tau_{fusion}$$
The merged parameters follow a usage-weighted average:
$$\theta_{fused} = \frac{U_i \theta_i + U_j \theta_j}{U_i + U_j}$$

---

## 4. Phase 7: GRPO (Group Relative Policy Optimization)

The most recent phase introduces Reinforcement Learning to optimize the **number of active experts** ($N$) per token.

### Equation 9: Metabolic Reward Function
The reward $R$ balances intelligence (NLL) against metabolic tax (sparsity):
$$R = 5.0 - \log(NLL) - \beta \cdot N^2$$
- $N$: Number of active experts for the sample.
- $\beta$: Quadratic tax coefficient.

### Equation 10: Group Relative Advantage
Advantages are computed relative to a group of $G$ samples from the same prompt:
$$A_i = \frac{R_i - \text{mean}(R_{1 \dots G})}{\text{std}(R_{1 \dots G})}$$

### Equation 11: Policy Gradient Loss
Optimizes the gating threshold via:
$$\mathcal{L}_{PG} = -\mathbb{E} [ A \cdot \log \pi(N \mid x) ]$$
Where the policy $\pi$ is implemented via a **Straight-Through Estimator (STE)** on the sigmoid gate:
$$Gate = \sigma\left(\frac{potential - threshold}{temp}\right)$$

### Equation 12: Adaptive Beta (PID Control)
The tax coefficient $\beta$ is adjusted to hit a target active expert count $N_{target}$:
$$\beta_{t+1} = \beta_t + \ell \cdot (N_{avg} - N_{target})$$
- $\ell$: Learning rate for the controller.

---

## 5. Parameter Optimization: LoRA experts

T-MoE uses **LoRA (Low-Rank Adaptation)** for all experts.
- **Base Weight**: $W \in \mathbb{R}^{d \times 4d}$ (Frozen GPT-2 weights).
- **Adapter**: $\Delta W = B \times A$, where $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times 4d}$.
- **Memory Efficiency**: 64 experts require ~32MB, compared to ~6.4GB for full MLP copies.
