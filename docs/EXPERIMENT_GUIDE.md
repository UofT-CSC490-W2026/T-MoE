# T-MoE Experiment Guide

## Setup

See [README.md](../README.md) for environment setup (venv/conda, pre-commit, cloud secrets).

---

## Configuration

T-MoE uses **one YAML file per experiment**, located in `experiments/`.

```
experiments/
└── gptneo_125m_metabolic.yaml   # Model + router + training + dataset settings
```

The config is loaded with OmegaConf. Any key can be overridden at runtime using dotlist notation — no config file editing required for one-off runs.

---

## Running Experiments

### Option A: Modal (Fast Iteration)

Recommended for most experiments. Data is prepared once and cached in a persistent Modal Volume.

```bash
# Stage 1: Tokenize dataset → Modal Volume (CPU, one-time)
modal run run_modal_training.py::stage_data --config gptneo_125m_metabolic.yaml

# Stage 2: Train (GPU reads directly from Volume)
modal run run_modal_training.py::stage_train --config gptneo_125m_metabolic.yaml

# With overrides (no config edits needed)
modal run run_modal_training.py::stage_train --config gptneo_125m_metabolic.yaml \
    --overrides "training.lr=5e-4" "router.num_experts=8"
```

### Option B: AWS Batch (Large-Scale)

```bash
# Upload dataset to S3
python run_pipeline.py

# Submit training job to AWS Batch
python run_aws_training.py --mode batch -c gptneo_125m_metabolic
```

### Option C: Local (Debugging)

```bash
# Prepare shards locally
python -m scripts.prepare_data --config experiments/gptneo_125m_metabolic.yaml

# Train locally
python -m scripts.train --config experiments/gptneo_125m_metabolic.yaml
```

---

## Config Overrides

Any config value can be overridden without editing the YAML:

```bash
# Learning rate + batch size
python -m scripts.train --config experiments/gptneo_125m_metabolic.yaml \
    training.lr=5e-4 training.batch_size=16

# Dataset swap
python -m scripts.train --config experiments/gptneo_125m_metabolic.yaml \
    dataset.dataset_key=wikitext-103

# Router type
python -m scripts.train --config experiments/gptneo_125m_metabolic.yaml \
    router.type=standard router.num_experts=8
```

---

## Output Structure

All training outputs go to:

```
outputs/<experiment_name>_<YYYYMMDD_HHMMSS>/
└── ckpt.pt     # Best checkpoint (overwritten on val_loss improvement)
                # Contains: step, val_loss, lora+router state, optimizer state, config
```

On Modal, outputs are also written to the persistent Volume under `/data/outputs/`.

---

## Models

Available models are registered in `src/models/`. The `model_key` in your YAML config selects the model.

| model_key | Parameters | Hidden Dim | Layers |
|---|---|---|---|
| `gpt-neo-125m` | 125M | 768 | 12 |
| `gpt-neo-350m` | 350M | 1024 | 24 |
| `gpt-neo-1.3b` | 1.3B | 2048 | 24 |
| `gpt-neo-2.7b` | 2.7B | 2560 | 32 |

**Adding a new model (e.g. Llama):**

1. Create `src/models/llama.py` with a `VARIANTS` dict and `@ModelRegistry.register("llama")`
2. Import it in `src/models/__init__.py`
3. That's it — `train.py`, `prepare_data.py`, and `model_lookup()` all resolve automatically.

---

## Datasets

Available datasets are registered in `scripts/prepare_data.py`'s `DATASET_REGISTRY`.

| dataset_key | Description |
|---|---|
| `wikitext-2` | Small, fast — good for debugging |
| `wikitext-103` | ~50× larger than wikitext-2 |
| `openwebtext` | Large web corpus (streaming) |
| `c4` | Colossal Clean Crawled Corpus (streaming) |

**Adding a new dataset:**

Add an entry to `DATASET_REGISTRY` in `scripts/prepare_data.py` and `DATASET_CATALOG` in `src/configs/dataset.py`:

```python
"my-dataset": {
    "hf_path": "org/dataset-name",
    "hf_name": "subset",       # None if no subset
    "text_column": "text",
    "splits": {"train": "train", "val": "validation"},
},
```

---

## Routers

Configure in the `router:` section of your experiment YAML.

| `router.type` | Description |
|---|---|
| `metabolic` | Metabolic router with fatigue dynamics |
| `standard` | Top-K softmax router with optional aux loss |
| `topk` | Top-K router, no load balancing |
| `switch` | Switch (Top-1) router |
| `dynmoe` | DynMoE sigmoid-gate router |

---

## Common Sweeps

### LoRA Rank Sweep

```bash
for rank in 8 16 32 64; do
    python -m scripts.train \
        --config experiments/gptneo_125m_metabolic.yaml \
        experiment_name=lora_rank_${rank} \
        expert.lora.rank=${rank} \
        expert.lora.alpha=${rank}
done
```

### Dataset Sweep

```bash
for dataset in wikitext-2 wikitext-103 openwebtext; do
    python -m scripts.train \
        --config experiments/gptneo_125m_metabolic.yaml \
        experiment_name=dataset_${dataset} \
        dataset.dataset_key=${dataset}
done
```

### Router Comparison

```bash
for router in metabolic standard topk; do
    python -m scripts.train \
        --config experiments/gptneo_125m_metabolic.yaml \
        experiment_name=router_${router} \
        router.type=${router}
done
```

---

## Key Files

| File | Purpose |
|---|---|
| `experiments/*.yaml` | Per-experiment config |
| `config.yaml` | Global defaults |
| `scripts/prepare_data.py` | Tokenize + pack dataset to shards |
| `scripts/train.py` | Main training loop |
| `run_modal_training.py` | Modal orchestrator |
| `run_aws_training.py` | AWS Batch orchestrator |
| `src/models/gpt_neo.py` | GPT-Neo backbone + VARIANTS |
| `src/routers/` | Router implementations |
| `src/configs/model.py` | `model_lookup()` — resolves model keys via registry |
| `src/configs/dataset.py` | `DATASET_CATALOG` |

---

## FAQ

**Q: How do I change the dataset?**
Set `dataset.dataset_key` in your YAML or pass it as a CLI override:
```bash
python -m scripts.train --config experiments/gptneo_125m_metabolic.yaml dataset.dataset_key=c4
```

**Q: Where does the HuggingFace token need to be?**
Set `HF_TOKEN` as an environment variable locally. On Modal, store it in the `tmoe-secrets` Workspace Secret.

**Q: How do I resume training?**
```bash
python -m scripts.train --config experiments/gptneo_125m_metabolic.yaml --resume outputs/my_run/ckpt.pt
```

**Q: Do I need to re-run stage_data every run?**
No. `stage_data` is idempotent — it skips if shards already exist on the Modal Volume.

**Q: How do I disable WandB?**
Set `logging.enabled: false` in your YAML, or unset `WANDB_API_KEY`.
