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

### Equation 2: Age-Aware Fatigue Dynamics
Fatigue $F_i$ accumulates with usage and recovers over time:
$$F_i(t+1) = (1-\gamma) F_i(t) + \eta_i(t) \cdot U_i(t), \qquad 0 < \gamma < 1$$
where:
$$\eta_i(t) = \eta_{eff} \cdot \min\!\left(1.0, \frac{t - \text{birth\_step}\_i}{T\_{\text{warmup}}}\right)$$
- $T_{\text{warmup}} > 0$
- $\gamma$: Recovery rate (homeostatic return).
- $\eta_{eff}$: Effective activation cost.
- $U_i(t)$: Usage indicator (1 if selected, 0 otherwise), or "weighted case" in Top-K routing it represents the routing weight (softmax) assigned to expert $i$.

### Equation 3: Adaptive Cost Scaling
To maintain stability during expert expansion, the activation cost $\eta$ scales with the number of active experts $N$:

$$\eta_{eff} = \eta_{base} \cdot \frac{N_{current}}{N_{start}}$$

---
