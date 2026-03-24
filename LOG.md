## 2026-03-24 — `qwen2_1.5b_stress_v1_fineweb` (pre-fix, lambda_init=1.0 bug)

- **[bug]** `lambda_init=1.0` in SPAR config — calibrated λ for this corpus is ~0.1, so the pre-calibration penalty was 10× too strong. `z_i = cos(x,W_i) - 1.0·max(0, L_i - 1/N)` suppressed any logit advantage that exceeded fair-share, forcing near-uniform routing from step 0.
  **result:** Steps 0–999: eff_E climbed 5.4→7.6, gini fell 0.469→0.18 — the penalty was actively flattening a routing distribution that would naturally have differentiated.
  **note:** This is artificial load balance, not learned balance. The gini decline here is not a success signal; it is suppression of the routing signal itself.

- **[instability]** λ calibration fires at step 1000: `λ = min(σ_cos / mean(L), 5.0)` returns ~0.1, a 10× drop from `lambda_init=1.0`. The penalty collapses discontinuously in a single step.
  **result:** eff_E crashes 7.5→6.2, gini spikes 0.18→0.374 at the calibration boundary — direct evidence of the pent-up routing preference being released in one step.
  **note:** The spike magnitude (Δgini = +0.194, Δeff_E = −1.3) is proportional to the suppressed differentiation accumulated over 1000 steps. Calibrating at step 1000 with λ_init=1.0 is equivalent to removing a clamp: the gate distribution snaps toward its unconstrained attractor.

- **[observation]** Post-calibration recovery (steps 1000–1725): eff_E oscillates 7.0–7.7, gini settles 0.15–0.28. Both metrics trend toward the values expected from a clean run with λ_init=0.1 from step 0.
  **result:** Loss descent unaffected — 2.47→2.44 over 1725 steps, healthy slow decline consistent with frozen backbone + LoRA-only training.
  **note:** LM loss is insensitive to the routing disruption at step 1000. This reconfirms the early-training decoupling of routing quality and LM quality observed in prior GPT-Neo runs.

- **[scaling]** First Qwen2-1.5B run. 48.4M trainable params (rank=32, 6 MoE layers, 8 experts) — correct. Throughput 228k–260k tok/sec on 8×H100, consistent with MoE+DDP overhead at this scale.
  **result:** GPU utilisation oscillates 40–100% — expected grad-accum pattern (compute phase 100%, allreduce+optimizer ~40%). Two drops to ~0% at ~12.5min and ~40min: rank-0 checkpoint save + eval I/O blocking non-rank-0 ranks.
  **note:** The I/O-induced GPU idle is the same DDP checkpoint stall seen on GPT-Neo runs. Already fixed in `src/training/checkpoint.py` (`dist.barrier()` in non-rank-0 path). Confirm the fix propagated to the Qwen training config before v2.

- **[fix]** `lambda_init` corrected 1.0→0.1 in SPAR config. Pre-calibration penalty is now at the right order of magnitude; the step-1000 discontinuity will be eliminated in `qwen2_1.5b_stress_v2_fineweb`.
  **result:** Not yet run.
  **note:** The correct invariant: `lambda_init` should be set to the expected calibrated value (estimated from a short warmup probe or from a prior run on the same corpus), not to an arbitrary default. For fineweb-edu, calibrated λ≈0.1; for a new corpus, run 200 steps with `lambda_calib_step=200` and inspect the returned value before committing to a full run.
