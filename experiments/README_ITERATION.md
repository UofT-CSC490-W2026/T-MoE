# Fast Iteration Experiments for Router Development

## Overview

These experiments are optimized for **rapid router hyperparameter tuning** before running the large 1.3B experiment.

## Experiment Comparison

| Config | Model | GPUs | Steps | Runtime (A100×4) | Runtime (A100×1) | Purpose |
|---|---|---|---|---|---|---|
| `gptneo_125m_metabolic_fast.yaml` | 125M | 1-2 | 500 | ~5-8 min | ~10-15 min | **Ultra-fast iteration** — test router params in minutes |
| `gptneo_125m_metabolic_v2.yaml` | 125M | 4 | 750 | ~25-30 min | N/A | **Standard validation** — verify router works correctly |
| `gptneo_1.3b_metabolic_v3.yaml` | 1.3B | 4 | 5,000 | ~7-14 hours | N/A | **Final production run** — full-scale training |

## Usage Workflow

### Phase 1: Ultra-Fast Iteration (125M Fast)
```bash
# Test router hyperparameters quickly
modal run run_modal_training.py::stage_train --config gptneo_125m_metabolic_fast.yaml

# Tune these parameters:
# - lambda_metabolic (0.1, 0.3, 0.5, 0.7, 1.0)
# - beta_cost (0.2, 0.4, 0.6, 0.8)
# - gamma_recovery (0.01, 0.05, 0.1)
# - magnitude_max (3.0, 4.0, 5.0)
```

**What to monitor:**
- `router/*/fatigue_std` — should be > 0 (fatigue is active)
- `router/*/effective_experts` — should be close to 8 (no collapse)
- `router/*/routing_diversity_gini` — should be < 0.3 (good balance)
- `val_loss` — should converge smoothly

**Expected runtime:** 5-15 minutes per run

### Phase 2: Standard Validation (125M Full)
```bash
# Verify router works with full training run
modal run run_modal_training.py::stage_train --config gptneo_125m_metabolic_v2.yaml
```

**What to verify:**
- Router metrics are stable over full training
- No expert collapse
- Fatigue dynamics work correctly
- Convergence is smooth

**Expected runtime:** 25-30 minutes

### Phase 3: Production Run (1.3B)
```bash
# Final large-scale training with tuned hyperparameters
modal run run_modal_training.py::stage_train --config gptneo_1.3b_metabolic_v3.yaml
```

**Expected runtime:** 7-14 hours (with early stopping)

## Key Optimizations in Fast Configs

### 125M Fast:
- ✅ Single GPU (or 2 for 2× speedup)
- ✅ 500 steps (early stopping catches convergence)
- ✅ Eval every 50 steps (fast feedback)
- ✅ WikiText-2 (fast data loading)
- ✅ Aggressive early stopping (50 intervals)

### 125M Standard (v2):
- ✅ 4 GPUs (balanced speed/cost)
- ✅ 750 steps (proven convergence)
- ✅ Eval every 200 steps
- ✅ Early stopping (100 intervals)

## Cost Estimates (Modal)

| Config | GPUs | Runtime | Cost/run (A100) | Cost/run (H100) |
|---|---|---|---|---|
| 125M Fast | 1× A100 | 10-15 min | ~$0.10 | ~$0.20 |
| 125M Fast | 2× A100 | 5-8 min | ~$0.20 | ~$0.40 |
| 125M Standard | 4× A100 | 25-30 min | ~$0.50-0.60 | ~$1.00-1.20 |
| 1.3B Production | 4× A100 | 7-14 hours | ~$60-120 | ~$110-220 |
| 1.3B Production | 4× H100 | 7-14 hours | ~$110-220 | (H100 faster) |

## Recommended Iteration Strategy

1. **Start with 125M Fast** — run 5-10 experiments tuning `lambda_metabolic`, `beta_cost`, `gamma_recovery`
2. **Validate on 125M Standard** — run 1-2 full training runs with best hyperparameters from step 1
3. **Scale to 1.3B** — run final production experiment with validated hyperparameters

**Total iteration time:** ~2-3 hours of fast experiments + 1 hour of standard validation = **3-4 hours** before production run.

## Monitoring Checklist

For each experiment, verify:

- [ ] `fatigue_std > 0` — metabolic penalty is active
- [ ] `effective_experts ≈ 8` — no expert collapse
- [ ] `routing_diversity_gini < 0.3` — good load balance
- [ ] `val_loss` converges smoothly (no oscillations)
- [ ] `top1_dominance` is reasonable (not too concentrated, not too uniform)

## Quick Commands

```bash
# Fast iteration (125M, single GPU)
modal run run_modal_training.py::stage_train --config gptneo_125m_metabolic_fast.yaml

# Standard validation (125M, 4 GPUs)
modal run run_modal_training.py::stage_train --config gptneo_125m_metabolic_v2.yaml

# Production (1.3B, 4 GPUs)
modal run run_modal_training.py::stage_train --config gptneo_1.3b_metabolic_v3.yaml
```
