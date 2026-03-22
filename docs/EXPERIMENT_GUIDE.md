# T-MoE Experiment Guide

## Running on Modal

Set the active config in `run_modal_training.py`:
```python
CONFIG = "experiments/gptneo_125m_metabolic_v4.yaml"
```
GPU type and count are read automatically from `compute.modal.gpu` in that YAML.

```bash
modal run run_modal_training.py              # full pipeline (data + train)
modal run run_modal_training.py --skip-data  # train only
modal run run_modal_training.py::stage_data  # data prep only
modal run run_modal_training.py::stage_train \
    --overrides "training.lr=3e-4,training.steps=3000"
```

## Running Locally

```bash
python -m scripts.prepare_data --config experiments/gptneo_125m_metabolic_v4.yaml
python -m scripts.train --config experiments/gptneo_125m_metabolic_v4.yaml
torchrun --standalone --nproc_per_node=4 \
    -m scripts.train --config experiments/gptneo_125m_metabolic_v4.yaml
```

## Datasets

| dataset_key | Tokens | Use |
|---|---|---|
| `wikitext-2` | ~2M | Smoke tests only |
| `wikitext-103` | ~103M | Quick ablations |
| `fineweb-edu` | ~10B | Production (default) |
| `c4` | ~350B | Large-scale |

To add a dataset: add one entry to `DATASET_REGISTRY` in `src/configs/dataset.py`.

## Models

| model_key | Parameters |
|---|---|
| `gpt-neo-125m` | 125M |
| `gpt-neo-1.3b` | 1.3B |
| `gpt-neo-2.7b` | 2.7B |

To add a model: create `src/models/<name>.py` with a `VARIANTS` dict and `@ModelRegistry.register("<name>")`, import in `src/models/__init__.py`.

## Routers

| `router.type` | Description |
|---|---|
| `metabolic` | Fatigue-based load balancing, no aux loss |
| `standard` | Top-K with aux loss (baseline) |
| `topk` | Top-K, no load balancing |
| `switch` | Top-1 |

## Output Structure

```
outputs/<experiment_name>_<YYYYMMDD_HHMMSS>/checkpoints/
    checkpoint_step_N.pt   # periodic
    best_model.pt          # best val_loss
```

On Modal: `/vol/outputs/<experiment_name>/`

## FAQ

**Do I need to re-run stage_data when switching experiments?**
Only if the dataset or model (tokenizer) changes. Shards are stored at
`/vol/data/<dataset>/vocab<N>/` — same tokenizer = shards are reused.

**How do I resume?**
```bash
python -m scripts.train --config ... --resume outputs/run/checkpoints/checkpoint_step_1000.pt
```

**How do I disable WandB?**
Set `logging.enabled: false` in the YAML or unset `WANDB_API_KEY`.
