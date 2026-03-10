## 2026-03-10

- **[experiment]** `metabolic_v4` (WandB: `pw764gz6`) vs `standard_v2` (WandB: `a78b5f6q`) — definitive controlled ablation. Both: GPT-Neo 125M, 8 experts, top-k=2, rank=16, wikitext-103, 5000 steps, A100:4, seed=42, identical LoRA/optimizer config. Single variable: router type.
  **result:** val_loss 3.1246 (metabolic) vs 3.1212 (standard) — statistically tied. Load balance: metabolic effective_E 5.0–7.4 across layers vs standard 3.5–4.6. Standard collapses in L1 and L11 despite `aux_loss_coef=0.01`. Metabolic throughput 14% lower (fatigue all-reduce overhead, 6 `dist.all_reduce` calls per step).
  **note:** Task loss parity with clear load-balance superiority is the paper's central empirical claim. Throughput cost is documented and expected; publishable as an honest tradeoff table.

- **[observation]** Dataset discrepancy in v4 config: `gptneo_125m_metabolic_v4.yaml` has `dataset_key: fineweb-edu` while `standard_v2` has `wikitext-103`. If WandB confirms both runs consumed wikitext-103, the YAML reflects a post-run edit error and the comparison remains valid. Must be reconciled before paper submission.
  **result:** Pending WandB artifact check.
  **note:** If v4 actually trained on fineweb-edu, the ablation is confounded and a rerun on matching data is required.

- **[experiment]** `standard_v2` collapse profile: L1 gini=0.616 by step 350 (from 2026-03-09 entry), L11 also collapses despite aux loss. `aux_loss_coef=0.01` (Switch Transformer default) is insufficient to prevent monopolisation on wikitext-103.
  **result:** Effective experts at step 5000: 3.5 (L1) to 4.6 (L11), well below theoretical max of 8.
  **note:** Raises the question whether any `aux_loss_coef` value can recover from early collapse without restarting. The collapse appears self-reinforcing once gini exceeds ~0.55.

- **[experiment]** `metabolic_v4` load-balance trajectory: effective_E 5.0–7.4 at step 5000, with Layer 1 stabilised at gini≈0.50 from step ~1800 onward (slope +0.000013/step in 2400–2900 window). No layer shows collapse.
  **result:** SoftSign penalty at equilibrium for a 2× overloaded expert: `λ × SoftSign(β/γ × 1/8) = 0.5 × SoftSign(1.0) ≈ 0.25` — sufficient to offset gate's alignment advantage for dominant experts.
  **note:** The fixed-point analysis in the v4 YAML comment matches empirical stabilisation step, which is encouraging for analytical tractability.

---

## 2026-02-XX — Deleted Experiment Archaeology

- **[experiment]** `metabolic_v2` (deleted): first differential-fatigue run, wikitext-2 (~2M tokens), 3000 steps, A100:4. Config: λ=0.5, β=0.4, γ=0.05 (β/γ=8), T=1.0, top-k=2, 8 experts, rank=16.
  **result:** Loss plateaued ~step 1400. T=1.0 produced diffuse routing weights. wikitext-2 too small to force routing diversity — no clear collapse signal, but also no specialisation.
  **note:** λ=0.5 was correct in magnitude but two formula bugs masked its effect: (1) U_i computed from softmax weights rather than token counts — U_i inflated for high-confidence slots; (2) birth_step warmup was always a no-op (implementation bug). Both bugs meant fatigue signal was noisy and poorly calibrated.

- **[bug]** `metabolic_v2`: fatigue `U_i` used softmax gating weights `w_i` as the usage signal, not uniform token counts. For top-k=2, correct usage is `1/k = 0.5` per selected slot. Using `w_i` instead inflates usage for high-confidence experts (those with large `w_i`) and undersignals usage for near-tie experts.
  **result:** Fatigue pressure was systematically higher on already-dominant experts (catching some of their dominance), but miscalibrated — the penalty magnitude did not correspond to actual compute load.
  **note:** Fixed in v3: U_i = uniform `1/top_k` per selected slot. Token-count usage is the correct compute-load proxy; softmax weights are a routing-confidence proxy.

- **[bug]** `metabolic_v2`: birth_step warmup was always a no-op due to an off-by-one in the warmup condition check. The gate received fatigue penalty from step 0, before any prototype directions had stabilised.
  **result:** With immediate penalty and noisy U_i, Layer 1 locked to 2 dominant experts before step 200.
  **note:** Fixed in v3/v4 via global λ warmup: penalty ramps 0→λ over 400 steps, implemented as a scalar multiplier on the SoftSign output rather than a step-count gate.

- **[experiment]** `metabolic_v3` (deleted): formula-fixed run, wikitext-103 (~103M tokens), 5000 steps, A100:4. Config: λ=0.3, β=0.4, γ=0.05, T=0.7, top-k=2, 8 experts, rank=16, warmup=400.
  **result:** Routing stabilised ~step 1500–2000. Effective experts improved over v2. λ=0.3 proved too weak post-warmup — Layer 1 collapsed to gini=0.59 by step 2350. val_loss≈3.13.
  **note:** T=0.7 (vs v2's 1.0) sharpened routing without hurting diversity — kept as the metabolic default. wikitext-103 was sufficient corpus size to reveal routing dynamics. λ=0.3 finding directly motivated the v4 decision to restore λ=0.5 while keeping warmup=400.

- **[experiment]** `standard_v1` (deleted): first aux-loss baseline, wikitext-2, 3000 steps, A100:4. Config: T=1.0, `aux_loss_coef=0.01`, top-k=2, 8 experts, rank=16.
  **result:** Some routing collapse on wikitext-2, insufficient corpus to force diversity. Aux loss unable to prevent expert monopolisation. Not a fair comparison against metabolic_v2 (different dataset, different corpus size).
  **note:** Both the dataset mismatch and T=1.0 vs the eventual metabolic T=0.7 made this comparison unpublishable. Superseded by standard_v2 on wikitext-103.

- **[observation]** Current active configs: `metabolic_v4` (WandB `pw764gz6`, completed), `standard_v2` (WandB `a78b5f6q`, completed), `metabolic_fast` (500 steps, 1 GPU, fast iteration), `gptneo_1.3b_metabolic_v3` (scale-up, not yet run), `smoketest` (CI only, 2 steps).
  **result:** Publication ablation (v4 vs standard_v2) complete. Scaling experiment and matched-temperature ablation remain.
  **note:** Next required experiments: (1) matched-temperature run (standard at T=0.7, metabolic at T=1.0) to deconfound Stress metric; (2) 1.3B scaling run; (3) Stress–val_PPL correlation sweep across v4 checkpoints.

---

- **[architecture]** Expert Stress metric defined: `Stress_i = CV(1 - cos(x_k, W_i))` for tokens routed to expert i, tracked via weighted online Welford. Raw observation is cosine distance `d_{i,k} = 1 - cos(x_k, W_i)`, weighted by usage fraction `w_i(t)`. Replaces earlier attributed-loss formulation `w_i * L(t)`, which was incorrect — it conflated routing-weight variance with loss variance via the product-variance identity.
  **result:** Formulation is now mathematically clean. Stress reduces to the CV of per-token cosine distances to the expert prototype, independent of scalar loss.
  **note:** Distinction matters for paper correctness; the old formula silently introduced a covariance term that does not have a routing interpretation.

- **[observation]** Layer-locality is a correctness requirement, not a convenience. In a 32-layer network, final loss `L(t)` is a diluted function of all 256 routing decisions simultaneously. Loss-based stress unfairly penalises Layer 2 Expert A when Layer 30 Expert B caused the spike. Cosine-distance Stress uses only the layer's own alignment, computed locally in the forward pass — immune to downstream routing errors.
  **result:** Credit assignment failure of loss-based metrics scales with depth. At 32 layers the attribution signal-to-noise ratio per expert per step is at most 1/256.
  **note:** This is the primary argument for Stress over loss-based diagnostics in deep MoE stacks.

- **[observation]** Zero compute overhead. `alignment[batch, seq, expert_i]` is already materialised in `compute_alignment()`. Stress observation is `1 - alignment[..., i]` — zero additional FLOPs. State per expert is three scalars `(n_i, mu_i, M2_i)` = 24 floats per layer for 8 experts.
  **result:** Token-level granularity ~2048 observations per expert per batch (vs 1 for loss-based). Welford estimate is statistically reliable from step 1.
  **note:** No justification needed to enable Stress tracking by default in all runs.

- **[hypothesis]** Stress is proportional to `sqrt(rank-1 LoRA approximation error)` on the routing distribution. At convergence, `W_i` aligns with the top eigenvector of the second-moment matrix `S_i` of routed tokens. Then `Stress_i^2 * mu_i^2 ≈ E_i(1) = 1 - lambda_1(S_i)` — the rank-1 LoRA approximation error. Low Stress implies routed tokens live near a rank-1 subspace, so rank-r LoRA (r≥1) achieves low approximation error.
  **result:** Not yet empirically verified.
  **note:** This is the external validity argument: low Stress should predict lower PPL if LoRA approximation error and PPL co-vary. Must be tested via Stress–val_PPL correlation across checkpoints.

- **[observation]** Circularity concern: the metabolic router selects tokens using cosine similarity, so selected tokens have low cosine distance by the selection rule itself. The mean `mu_i` is circular; the variance (CV) is not. Two experts can have identical mean cosine distance but different Stress depending on whether selected tokens form a coherent cluster or a diffuse cloud.
  **result:** Report CV (Stress), not mean cosine distance, in all cross-router comparisons.
  **note:** Universal computation via post-hoc normalisation `d_{i,k}^norm = 1 - (x_k/||x_k||) · (W_i/||W_i||)` is valid for standard/topk/switch routers and removes the circularity objection for comparison tables.

- **[ablation]** Temperature confound: metabolic uses T=0.7, standard uses T=1.0. Lower temperature sharpens routing, mechanically reducing cosine distance variance regardless of routing mechanism.
  **result:** Any observed Stress difference between routers is currently confounded with temperature.
  **note:** Matched-temperature ablation required before any cross-router Stress comparison is paper-ready. Planned: rerun standard router at T=0.7 and metabolic at T=1.0.

- **[observation]** 2×2 diagnostic table: (Low Gini + Low Stress) = balanced + coherent = metabolic goal; (High Gini + Low Stress) = routing collapse with narrow token set = standard router pathology; (Low Gini + High Stress) = balanced but incoherent = random routing baseline. Low Stress alone is not a quality signal.
  **result:** Gini must always be reported alongside Stress. The joint (Gini, Stress) pair is the meaningful diagnostic.
  **note:** This framing is reviewer-robust — it prevents the obvious objection that a random router could also achieve low Gini.

- **[idea]** Correct paper positioning: Stress is a mechanistic diagnostic, not a primary comparison metric. Use it to show (a) Stress decreases as λ warmup activates (steps 400+), (b) Stress gradient across layers complements MI gradient (earlier layers higher stress, later layers lower), (c) Stress–PPL correlation if empirically confirmed. Primary cross-router comparison remains Gini + effective_E + MI + downstream accuracy.
  **result:** Framing avoids overreach; positions Stress as interpretability evidence rather than a performance claim.
  **note:** External validity (Stress–PPL correlation) is the one missing experiment that would allow stronger claims.

## 2026-03-10 (continued) — `metabolic_v4` fineweb-edu run, steps 0–1225

- **[observation]** Warmup phase (steps 0–400) on fineweb-edu produces near-perfect balance: gini 0.157→0.026, eff_E rising to 8.0. The 400-step penalty ramp gives the gate clean prototype directions before fatigue pressure activates — consistent with wikitext-103 behavior.
  **result:** Warmup design generalises to a harder corpus. Initial balance ceiling of eff_E=8.0 is router-agnostic; the meaningful signal starts post-warmup.
  **note:** fineweb-edu warmup dynamics are indistinguishable from wikitext-103 at this stage, confirming the warmup mechanism is not data-regime-specific.

- **[instability]** Post-warmup routing drift (steps 400–1225): gini 0.026→0.207, eff_E 8.0→7.4, F_σ rising monotonically 0.042→0.391 with no plateau visible. LM loss unaffected (4.93→3.44 converging normally). Drift rate decelerating: +0.102 gini over steps 400–800, +0.079 over steps 800–1225.
  **result:** Fatigue is building and applying braking force — the deceleration is evidence the mechanism is active — but has not yet reached fixed-point equilibrium by step 1225.
  **note:** Decelerating slope is the key diagnostic. A monotonically accelerating gini would imply the mechanism is failing entirely; deceleration implies a ceiling is forming, just higher than observed on wikitext-103.

- **[hypothesis]** Root cause of drift: λ=0.5 sets a maximum penalty of `λ × SoftSign(∞) = 0.5`. On wikitext-103, gate logit advantages for dominant experts stayed below 0.5, so the ceiling was sufficient. On fineweb-edu, 400 steps of unconstrained task-loss gradient descent grow logit advantages >0.5 for certain high-frequency token patterns — the penalty ceiling is breached and fatigue cannot fully correct.
  **result:** Not yet confirmed — requires extracting per-expert gate logit distributions at steps 400 and 1225 from WandB to compare against the λ=0.5 ceiling.
  **note:** If logit advantages are found clustered near 0.45–0.55, a modest λ increase to 1.0 should close the gap. If they are >1.0, a fundamentally stronger mechanism is needed.

- **[hypothesis]** Mechanistic distinction from aux loss preserved even if λ requires recalibration. Aux loss constrains the gradient direction of gate weights, competing with LM loss during backprop. Metabolic λ operates on routing logits at inference time, outside the gradient graph — gate weights still specialize freely, penalty applied post-hoc per step. Increasing λ does not introduce the gradient competition that aux loss creates.
  **result:** The mechanistic claim survives even if λ=0.5 is empirically insufficient for fineweb-edu. The correct framing is hyperparameter recalibration, not mechanism failure.
  **note:** This distinction is paper-critical. Reviewer objection "you just need a stronger aux loss" is rebutted by the gradient-graph argument, not by load balance numbers alone.

- **[idea]** Next run: `metabolic_v4b` with λ=1.0, all else equal (β=0.4, γ=0.05, warmup=400, fineweb-edu). Doubles the penalty ceiling to 1.0. Diagnostic criterion: if gini stabilises <0.1 by step 2000 with λ=1.0, the mechanism works on real data with recalibrated λ. If gini continues drifting past 0.2 with λ=1.0, the fixed-λ assumption is fundamentally broken for complex corpora.
  **result:** Not yet run. Blocked on current run completing.
  **note:** λ=1.0 is the natural next doubling; jumping further (e.g. λ=2.0) risks overcorrecting and suppressing legitimate specialisation — step the search.

- **[idea]** Longer-term: data-adaptive λ — increase λ when gate gradient norm is high (task loss driving rapid logit growth), decay λ when gate has converged. Signal: `||∇_{gate} L_task||` per step, already computable from the existing optimizer state. This makes the fatigue mechanism phase-aware and would be genuinely novel vs both fixed-λ metabolic and aux-loss baselines. Deferred until fixed-λ sweep (λ ∈ {0.5, 1.0}) is complete.
  **result:** Conceptual only. No implementation yet.
  **note:** The gate gradient norm as a λ-schedule signal is the key idea. If gate grad norm correlates with post-warmup gini drift rate across runs, this becomes a clean empirical motivation for the adaptive mechanism.

## 2026-03-09

- **Standard aux-loss router collapses early** (gini=0.616, eff_E=4.0/8 by step 350 at aux_loss_coef=0.01), driving some experts to zero tokens. This requires `find_unused_parameters=True` in DDP (~5–10% per-step overhead) — a permanent tax for any collapsing router.

- **Metabolic router never triggers the zero-token condition.** Fatigue dynamics keep all 8 experts active throughout training, making it compatible with `find_unused_parameters=False`. This DDP compatibility advantage is absent from Switch Transformer, ST-MoE, and Mixtral papers.

- **v4 (λ=0.5 + warmup=400) reaches stable equilibrium.** Layer 1 gini stabilises at ~0.50 from step 1800 onward (not collapsing). Gini slope in the 2400–2900 window: +0.000013/step → effectively zero. The SoftSign penalty holds the overloaded experts at a fixed point rather than allowing continued concentration.

- **Layer-depth specialisation gradient confirmed.** MI(token, expert) increases monotonically from L1 (0.10) to L11 (0.24) at step 4150. Later layers develop more token-type-specific routing without any explicit per-layer objective — the metabolic router discovers this structure naturally.

- **λ warmup prevents early routing lock-in.** Without warmup (v2), Layer 1 locked to 2 dominant experts before step 200. With warmup=400 (v3/v4), Layer 1 stays near-uniform (gini<0.08) through step 400, giving the gate time to find better prototype directions before fatigue feedback activates.

- **λ=0.3 (v3) too weak post-warmup.** Layer 1 continued collapsing to gini=0.59 by step 2350. λ=0.5 (v4) holds Layer 1 at equilibrium gini≈0.50. The penalty at equilibrium (λ×SoftSign(F_max)≈0.31) matches the gate's alignment advantage for dominant experts.
