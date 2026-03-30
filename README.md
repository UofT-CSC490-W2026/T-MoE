# SPAR: Stress-Penalized Adaptive Routing for Balanced Sparse MoE

Aviral Bhardwaj\*, Ananya Jain\*, Ansh Agrawal\*, Mann Thakkar*\
*Equal contribution. Department of Computer Science, University of Toronto.*

---

Sparse Mixture-of-Experts (MoE) models scale capacity by activating only a small subset of experts per token, but this efficiency is undermined in the LoRA-MoE setting by an initialization bottleneck: adapters begin as nearly identical mappings, depriving the router of early task signal and leading to rapid routing collapse even with auxiliary balancing losses.

**SPAR** applies a symmetric, zero-equilibrium load correction directly to router logits, outside the gradient graph. Overloaded experts are penalized in proportion to their excess load; underloaded experts receive a proportional boost; the correction is exactly zero at equilibrium. Load control and task optimization are fully decoupled with no auxiliary loss coefficient to tune.

We evaluate SPAR on a frozen Qwen2-1.5B backbone against standard top-k routing with auxiliary loss, DeepSeek-style bias balancing, Switch Transformer (top-1), and Expert Choice routing. SPAR achieves leading perplexity on Wikitext-103 and Pile while sustaining structured expert specialization (effective experts ≈ 7.2–7.9 / 8).

```
z_i(x,t) = cos(x, W_i) − λ · (L_i(t) − k/N)   # symmetric logit correction
w_i       = softmax(cos(x, W_i) / τ_t)           # output weight, τ anneals 0.5→0.10
L_i(t)    = (1−α)·L_i(t−1) + α·U_i(t)           # EMA load, α=0.01
λ         = min(σ_cos · N, 5.0)                   # auto-calibrated once at step 1000
```

## Repository Structure

```
T-MoE/
├── src/
│   ├── routers/            # SPAR (stress_corrected.py), Standard, Switch, DeepSeek, Expert Choice
│   ├── experts/            # LoRA adapters (lora.py, qwen2_lora.py, gpt_neo_lora.py)
│   ├── layers/             # LoRAMoELayer — the MoE wrapper
│   └── models/             # Qwen2-1.5B and GPT-Neo-125M integrations
├── scripts/
│   ├── prepare_data.py     # Tokenize + pack dataset into binary shards
│   ├── train.py            # DDP training loop (AdamW, torch.compile, WandB)
│   └── eval.py             # Perplexity, LM-Harness, efficiency, routing analysis
├── evals/                  # Evaluation modules (perplexity, lm_harness, routing_analysis)
├── experiments/            # Per-experiment YAML configs (Qwen2-1.5B and GPT-Neo-125M)
├── docs/                   # Mathematical reference and experiment history
├── infra/                  # AWS Batch, S3, Terraform, Modal infrastructure
├── run_modal_training.py   # Modal orchestrator (recommended for iteration)
└── run_aws_training.py     # AWS Batch orchestrator (large-scale runs)
```

## Reproducing Paper Experiments

All paper experiments use Qwen2-1.5B on fineweb-edu. Configs are in `experiments/`.

| Experiment | Config |
|---|---|
| SPAR (ours) | `qwen2_1.5b_stress_v3-fineweb.yaml` |
| Standard + aux loss | `qwen2_1.5b_standard_v1-fineweb.yaml` |
| DeepSeek bias balancing | `qwen2_1.5b_deepseek_v1-fineweb.yaml` |
| Switch Transformer | `qwen2_1.5b_switch_v1-fineweb.yaml` |
| Expert Choice | `qwen2_1.5b_expert_choice_v1-fineweb.yaml` |

### Running on Modal (recommended)

```bash
# 1. Prepare data (CPU, saves to Modal Volume)
modal run run_modal_training.py::stage_data --config qwen2_1.5b_stress_v3-fineweb.yaml

# 2. Train (H100×8, ~19k steps)
modal run run_modal_training.py::stage_train --config qwen2_1.5b_stress_v3-fineweb.yaml

# 3. Evaluate (perplexity + LM-Harness + routing analysis)
modal run run_modal_training.py::stage_eval --config qwen2_1.5b_stress_v3-fineweb.yaml \
    --task all --checkpoint best
```

Config overrides without editing YAML:
```bash
modal run run_modal_training.py::stage_train --config qwen2_1.5b_stress_v3-fineweb.yaml \
    --overrides "training.lr=3e-4" "router.top_k=2"
```

### Running on AWS Batch

For large-scale runs backed by S3. See [infra/README.md](infra/README.md) for full setup.

```bash
python run_aws_training.py --mode batch -c qwen2_1.5b_stress_v3-fineweb
```

### Local Debugging

```bash
python -m scripts.prepare_data --config experiments/qwen2_1.5b_stress_v3-fineweb.yaml
python -m scripts.train --config experiments/qwen2_1.5b_stress_v3-fineweb.yaml
```

## Setup

```bash
# Create environment (Python 3.11)
conda create -n tmoe python=3.11 && conda activate tmoe

# Install dependencies
pip install -r requirements.txt
pip install pre-commit && pre-commit install
```

Cloud credentials (Modal, AWS, WandB, HuggingFace): see [infra/README.md](infra/README.md).

## Documentation

| Document | Contents |
|---|---|
| [docs/EQUATIONS.md](docs/EQUATIONS.md) | Full SPAR formulation, proofs, hyperparameter rationale, historical design decisions |
| [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md) | Experiment history, key findings, rejected ideas |
| [experiments/EXPERIMENT_GUIDE.md](experiments/EXPERIMENT_GUIDE.md) | How to configure and run new experiments |
| [infra/README.md](infra/README.md) | Cloud infrastructure setup (Modal, AWS, Terraform) |

## Citation

```bibtex
@article{bhardwaj2026spar,
  title   = {SPAR: Stress-Penalized Adaptive Routing for Balanced Sparse MoE},
  author  = {Bhardwaj, Aviral and Jain, Ananya and Agrawal, Ansh and Thakkar, Mann},
  year    = {2026},
  url     = {https://github.com/UofT-CSC490-W2026/T-MoE}
}
```

![Coverage](https://img.shields.io/badge/coverage-93.31%25-brightgreen)
