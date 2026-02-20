# T-MoE Experiment System - Setup Guide

## 📋 Quick Setup

### 1. Install Dependencies

> [!IMPORTANT]
> Use **Python 3.11**. **Python 3.14 is currently incompatible** with Hydra’s argument parser.

### 2. Configure WandB
```bash
wandb login
export WANDB_ENTITY="your_team"
export WANDB_PROJECT="your_project_name"
```

---

## 🔧 Configuration

T-MoE uses a two-tier configuration system:
1. **Base Config (`config.yaml`)**: Contains global defaults (resource paths, partitions, etc.). **Do not modify this file for specific experiments.**
2. **Experiment Config (`experiments/*.yaml`)**: Overwrites base settings for specific runs.

### Model Catalog
All backbone models are defined in `catalog/model_catalog.py`. Use the `model_key` in your experiment config:

| Model Key | Parameters | Hidden Dim | Layers | Description |
| :--- | :--- | :--- | :--- | :--- |
| `gpt-neo-125m` | 125M | 768 | 12 | GPT-Neo small |
| `gpt-neo-350m` | 350M | 1024 | 24 | GPT-Neo medium |
| `gpt-neo-1.3b` | 1.3B | 2048 | 24 | GPT-Neo large |
| `gpt2` | 117M | 768 | 12 | GPT-2 small |
| `gpt2-medium` | 345M | 1024 | 24 | GPT-2 medium |
| `llama-7b` | 7B | 4096 | 32 | Llama 2 (Placeholder) |

### Single Configuration File: `config.yaml`

All experiment settings are in one file with these sections:

1. **Experiment Metadata** - name, seed, execution environment
2. **Model** - backbone type, MoE injection layers
3. **Router** - routing strategy and parameters
4. **Expert** - expert type and configuration (LoRA)
5. **Training** - batch size, learning rate, checkpointing
6. **Dataset** - uses `catalog/dataset_catalog.py` for dataset selection
7. **Logging** - WandB configuration
- `configs/dataset.py` - DatasetConfig (uses datacatalog)
- `catalog/dataset_catalog.py` - Dataset catalog (wikitext-2, c4, etc.)

---

## 🚀 Usage

### Local Execution (Interactive)

```bash
# Basic run (using shorthand -c)
python train.py -c gptneo_125m_lora

# Or using long flag --config
python train.py --config gptneo_125m_lora
```

### AWS Execution

1. **Edit config.yaml** (or your experiment config) for AWS settings:
   ```yaml
   execution_env: aws
   compute:
     aws:
       data_root: s3://your-bucket/datasets
       output_root: s3://your-bucket/outputs
   ```

2. **Run on EC2:**
   ```bash
   python train.py --config gptneo_125m_lora
   ```

---

## 📂 Output Structure

All outputs follow this structure (both local and AWS):

```
{output_root}/experiments/<experiment_name>_<timestamp>/
├── checkpoints/
│   ├── checkpoint_step_500.pt
│   ├── checkpoint_step_1000.pt
│   ├── best_model.pt
│   └── *.json (metadata)
├── logs/
│   └── training.log
├── wandb/
│   └── (wandb files)
└── config.yaml (saved copy)
```

**Local:** `./outputs/experiments/...`
**AWS:** `s3://your-bucket/outputs/experiments/...`

---

## 🗂️ Dataset Catalog

The system uses `catalog/dataset_catalog.py` for dataset management:

```python
# Available datasets:
- wikitext-2        # Small (2M tokens)
- wikitext-103      # Large (103M tokens)
- c4                # Colossal Clean Crawled Corpus
- openwebtext       # Reddit outlinks
- the_pile          # Diverse corpus
- code              # GitHub code
```

**Usage in config.yaml:**
```yaml
dataset:
  dataset_key: wikitext-2  # Uses catalog
```

**Custom dataset:**
```yaml
dataset:
  custom_dataset_name: my-org/my-dataset
  custom_dataset_config: subset
  text_column: content
```

---

## 🎯 Pipeline Overview

### Local Pipeline

1. **Config** → Read `config.yaml`
2. **Datacatalog** → Resolve dataset from `catalog/dataset_catalog.py`
3. **Model** → Build with LoRA experts + router (using `configs/router.py`)
4. **Train** → Run with AMP, checkpointing, WandB
5. **Output** → Save to `./outputs/experiments/<name>_<timestamp>/`

### AWS Pipeline

1. **Config** → Read `config.yaml` with `execution_env=aws`
2. **Datacatalog** → Resolve dataset (downloads to local cache)
3. **Model** → Build (same as local)
4. **Train** → Run on EC2 GPU
5. **Output** → Save to `s3://bucket/outputs/experiments/<name>_<timestamp>/`

---

## 📊 Monitoring

**WandB Dashboard:**
```
https://wandb.ai/<your_username>/<your_project_name>
```

**Logs:**
- Local: `./outputs/experiments/<name>_*/logs/`
- AWS: `s3://bucket/outputs/experiments/<name>_*/logs/`

**Checkpoints:**
- Best model: `checkpoints/best_model.pt`
- Latest: `checkpoints/checkpoint_step_<N>.pt`

---

## 🔧 Example Workflows

### 1. Router Comparison
```yaml
# config.yaml - TopK Router
router:
  type: TopKRouter
  num_experts: 4
  top_k: 1
```

```yaml
# config.yaml - Metabolic Router
router:
  type: MetabolicRouter
  num_experts: 8
  top_k: 2
  metabolic:
    lambda_metabolic: 0.1
    gamma_recovery: 0.01
```

### 2. LoRA Rank Sweep
```bash
for rank in 8 16 32 64; do
    python train.py \
        --config gptneo_125m_lora \
        experiment_name=lora_rank_${rank} \
        expert.lora.rank=${rank} \
        expert.lora.alpha=${rank}
done
```

### 3. Dataset Sweep
```bash
for dataset in wikitext-2 c4 openwebtext; do
    python train.py \
        --config gptneo_125m_lora \
        experiment_name=dataset_${dataset} \
        dataset.dataset_key=${dataset}
done
```

---

## 📝 Key Files

**Main:**
- `config.yaml` - Single comprehensive config
- `train.py` - Main training script

**Existing Configs (Used):**
- `configs/base.py` - BaseConfig
- `configs/router.py` - Router configs
- `configs/dataset.py` - Dataset config
- `catalog/dataset_catalog.py` - Dataset catalog

**Source:**
- `src/models/` - Model backbones
- `src/experts/lora.py` - LoRA experts
- `src/routers/` - Router implementations
- `src/training/` - Trainer, checkpointing
- `src/utils/experiment.py` - Builders (uses datacatalog)

**Documentation:**
- `docs/EXPERIMENT_GUIDE.md` - This file

---

## ❓ FAQ

**Q: How do I add a new dataset?**
A: Add to `catalog/dataset_catalog.py`:
```python
DATASET_CATALOG["my-dataset"] = {
    "name": "org/dataset-name",
    "config": "subset",
    "text_column": "text",
    "streaming": False,
    "description": "My dataset",
}
```

**Q: How do I change data paths?**
A: Edit `config.yaml`:
```yaml
compute:
  local:
    data_root: ./data
    output_root: ./outputs
```

**Q: How do I disable AWS?**
A: Set `execution_env: local` in `config.yaml`.
