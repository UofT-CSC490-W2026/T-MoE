# Experiments

## Active Configs

| Config | Model | GPUs | Steps | Purpose |
|---|---|---|---|---|
| `gptneo_125m_metabolic_fast.yaml` | 125M | 1 | 500 | Fast iteration — tune router params |
| `gptneo_125m_metabolic_v4.yaml` | 125M | 4 | 5000 | Main metabolic run (fineweb-edu) |
| `gptneo_125m_standard_v2.yaml` | 125M | 4 | 5000 | Standard router baseline |
| `gptneo_1.3b_metabolic_v3.yaml` | 1.3B | 4 | 5000 | Scale-up |
| `smoketest.yaml` | 125M | 1 | 2 | CI only |

v2/v3 wikitext-103 runs are archived in RESEARCH_LOG.md — v4 is the production config on fineweb-edu.

## Running on Modal

Edit one line in `run_modal_training.py`:
```python
CONFIG = "experiments/gptneo_125m_metabolic_v4.yaml"
```

```bash
modal run run_modal_training.py              # data + train
modal run run_modal_training.py --skip-data  # train only
modal run run_modal_training.py::stage_data  # data prep only
```

Hyperparameter sweep without editing the YAML:
```bash
modal run run_modal_training.py::stage_train \
    --overrides "router.metabolic.lambda_metabolic=0.3,router.metabolic.beta_cost=0.4"
```

## What to Monitor

- `router/*/effective_experts` — target > 6 (no collapse)
- `router/*/routing_diversity_gini` — lower = more balanced
- `router/*/fatigue_std` — should be > 0 (fatigue is active)
- `router/*/fatigue_mean` — should be ≈ 0 (zero-sum invariant)
- `val_loss` — smooth convergence
