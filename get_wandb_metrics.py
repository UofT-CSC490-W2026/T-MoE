import wandb

api = wandb.Api()
try:
    runs = api.runs("aviral-bhardwaj-university-of-toronto/T-MoE")

    target_run = None
    for run in runs:
        if run.name == "gptneo_125m_metabolic_optimized_20260220_173320":
            target_run = run
            break

    if target_run:
        print(f"Run Name: {target_run.name}")
        print(f"State: {target_run.state}")
        print("\n--- Key Metrics ---")
        keys_to_print = [
            "train/loss",
            "train/best_loss",
            "train/perplexity",
            "train/avg_perplexity",
            "train/best_perplexity",
            "train/num_steps",
            "train/tokens_per_sec",
        ]

        # Check router metrics availability
        val_map = target_run.summary
        router_layers = [k for k in val_map.keys() if "router/" in k]
        layer_ids = sorted(list(set([k.split("/")[1] for k in router_layers])))
        print(f"\nRouter active layers: {', '.join(layer_ids)}")

        print("\nTraining Metrics:")
        for k in keys_to_print:
            if k in val_map:
                print(f"  {k}: {val_map[k]}")

        if layer_ids:
            for target_layer in layer_ids:
                print(f"\nRouter Metrics ({target_layer}):")
                router_keys = [
                    "effective_experts",
                    "expert_entropy",
                    "expert_entropy_normalized",
                    "fatigue_mean",
                    "fatigue_std",
                    "fatigue_max",
                    "routing_diversity_gini",
                ]
                for k in router_keys:
                    full_key = f"router/{target_layer}/{k}"
                    if full_key in val_map:
                        val = val_map[full_key]
                        print(
                            f"  {k}: {val:.4f}"
                            if isinstance(val, float)
                            else f"  {k}: {val}"
                        )
    else:
        print(
            "Run 'gptneo_125m_metabolic_optimized_20260220_162018' not found in project T-MoE"
        )
except Exception as e:
    print(f"Error accessing WandB API: {e}")
