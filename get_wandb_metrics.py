import wandb

api = wandb.Api()
try:
    runs = api.runs("aviral-bhardwaj-university-of-toronto/T-MoE")

    target_run = None
    for run in runs:
        if run.name == "gptneo_125m_metabolic_v2_20260221_224309":
            target_run = run
            break

    if target_run:
        print(f"Run Name: {target_run.name}")
        print(f"State:    {target_run.state}")
        print("\n--- Key Metrics ---")

        val_map = target_run.summary

        # ── Training metrics ─────────────────────────────────────────────
        training_keys = [
            "train/loss",
            "train/best_loss",
            "train/perplexity",
            "train/avg_perplexity",
            "train/best_perplexity",
            "train/num_steps",
            "train/tokens_per_sec",
        ]
        print("\nTraining Metrics:")
        for k in training_keys:
            if k in val_map:
                print(f"  {k}: {val_map[k]}")

        # ── Router layers ─────────────────────────────────────────────────
        router_layers = [k for k in val_map.keys() if k.startswith("router/")]
        layer_ids = sorted(set(k.split("/")[1] for k in router_layers))
        print(f"\nRouter active layers: {', '.join(layer_ids)}")

        # ── Per-layer router metrics ──────────────────────────────────────
        #
        # Collapse diagnosis thresholds (top_k=2, num_experts=8):
        #   specialization_score > 0.3  → healthy specialization
        #   specialization_score < 0.1  → near collapse (input-agnostic routing)
        #   collapse_score       > 0.9  → confirmed collapse
        #   effective_experts    < 3    → strong concentration, investigate
        #   routing_diversity_gini>0.6  → load severely imbalanced
        #   fatigue_std / fatigue_mean  → CV > 1.0 suggests extreme specialization

        BASE_METRICS = [
            ("effective_experts", "healthy: > 3.0"),
            ("expert_entropy_normalized", "healthy: > 0.5"),
            ("routing_diversity_gini", "healthy: < 0.6"),
        ]
        COLLAPSE_METRICS = [
            ("specialization_score", "spec: > 0.3 | collapse: < 0.1"),
            ("collapse_score", "collapse if > 0.9"),
            ("marginal_entropy", "H(expert) across all tokens"),
            ("conditional_entropy", "H(expert|token_type) lower=more specialized"),
        ]
        FATIGUE_METRICS = [
            ("fatigue_mean", "should ≈ β·U/γ at steady state"),
            ("fatigue_std", "high std → heterogeneous specialization"),
            ("fatigue_max", "should be < 5× fatigue_mean"),
        ]

        for layer in layer_ids:
            print(f"\n{'=' * 55}")
            print(f"  Layer: {layer}")
            print(f"{'=' * 55}")

            def get(key):
                full = f"router/{layer}/{key}"
                return val_map.get(full, None)

            def fmt(val):
                return f"{val:.4f}" if isinstance(val, float) else str(val)

            print("\n  [Diversity]")
            for k, note in BASE_METRICS:
                v = get(k)
                if v is not None:
                    print(f"    {k:<35} {fmt(v):<10}  # {note}")

            print("\n  [Collapse vs Specialization]")
            spec = get("specialization_score")
            collapse = get("collapse_score")
            eff = get("effective_experts") or 2.0

            if spec is not None:
                if eff <= 3.0 and spec < 0.1:
                    verdict = "✗ TRUE COLLAPSE (same experts for everything)"
                elif eff > 3.0 and spec < 0.2:
                    verdict = (
                        "✓ CONTEXTUAL ROUTING (routes on context, not raw token ID)"
                    )
                elif spec > 0.3:
                    verdict = "✓ TOKEN SPECIALIZATION (routes heavily on token ID)"
                else:
                    verdict = "⚠ AMBIGUOUS"

                print(f"    {'specialization_score':<35} {fmt(spec):<10}  → {verdict}")
            if collapse is not None:
                print(
                    f"    {'collapse_score':<35} {fmt(collapse):<10}  # {COLLAPSE_METRICS[1][1]}"
                )
            for k, note in COLLAPSE_METRICS[2:]:
                v = get(k)
                if v is not None:
                    print(f"    {k:<35} {fmt(v):<10}  # {note}")

            if spec is None:
                print("    (specialization_score not logged yet — waiting for tracker)")

            print("\n  [Fatigue State]")
            for k, note in FATIGUE_METRICS:
                v = get(k)
                if v is not None:
                    print(f"    {k:<35} {fmt(v):<10}  # {note}")

            # Coefficient of variation or Standard Deviation diagnostic (V2 compatibility)
            f_mean = get("fatigue_mean")
            f_std = get("fatigue_std")

            if f_mean is not None and f_std is not None:
                # If mean is near zero (Differential Fatigue v2), CV is mathematically undefined/explodes.
                if abs(f_mean) < 1e-4:
                    cv_note = (
                        "high heterogeneity"
                        if f_std > 0.5
                        else "moderate"
                        if f_std > 0.1
                        else "uniform"
                    )
                    print(
                        f"    {'fatigue_std (mean=0)':<35} {f_std:.4f}    # {cv_note}"
                    )
                elif f_mean > 0:
                    cv = f_std / f_mean
                    cv_note = (
                        "high heterogeneity"
                        if cv > 1.0
                        else "moderate"
                        if cv > 0.5
                        else "uniform"
                    )
                    print(f"    {'fatigue_cv (std/mean)':<35} {cv:.4f}    # {cv_note}")

    else:
        print("Run not found in project T-MoE")

except Exception as e:
    print(f"Error accessing WandB API: {e}")
