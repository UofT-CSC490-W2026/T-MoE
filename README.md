# T-MoE

Mixture-of-Experts fine-tuning of GPT-Neo using LoRA adapters and a Metabolic Router.
The system supports fast iteration via Modal and large-scale training via AWS Batch.

## Architecture

```
T-MoE/
├── scripts/
│   ├── prepare_data.py   # Stage 1: Tokenize + pack dataset into binary shards
│   ├── train.py          # Stage 2: Train model (reads shards, no Hydra, DDP-ready)
│   ├── eval.py           # Stage 3: Evaluation (placeholder)
│   └── generate.py       # Stage 4: Text generation (placeholder)
├── experiments/          # Per-experiment YAML configs
├── src/                  # Model, layers, routers, training logic
├── infra/                # AWS Batch, S3, Terraform infrastructure
├── run_modal_training.py # Modal orchestrator (fast iteration)
└── run_aws_training.py   # AWS Batch orchestrator (heavy training)
```

## Running Experiments

### Option A: Modal (Fast Iteration)

Recommended for development and experiments up to medium scale.

```bash
# Stage 1: Prepare data (runs on cheap CPU, saves to Modal Volume)
modal run run_modal_training.py::stage_data --config gptneo_125m_metabolic.yaml

# Stage 2: Train (reads from Modal Volume, no S3 transfers)
modal run run_modal_training.py::stage_train --config gptneo_125m_metabolic.yaml

# With config overrides
modal run run_modal_training.py::stage_train --config gptneo_125m_metabolic.yaml \
    --overrides "training.lr=1e-4" "router.num_experts=4"
```

### Option B: AWS Batch (Heavy Training)

For large-scale runs backed by S3 storage.

```bash
# Stage 1: Upload dataset to S3
python run_pipeline.py

# Stage 2: Submit training job to AWS Batch
python run_aws_training.py --mode batch -c gptneo_125m_metabolic
```

### Local Debugging

```bash
# Prepare data locally
python -m scripts.prepare_data --config experiments/gptneo_125m_metabolic.yaml

# Train locally (verify nothing crashes before cloud run)
python -m scripts.train --config experiments/gptneo_125m_metabolic.yaml
```

## Config Overrides

All commands support OmegaConf dotlist overrides without editing the YAML:

```bash
python -m scripts.train --config experiments/gptneo_125m_metabolic.yaml \
    training.lr=5e-4 training.batch_size=32 router.num_experts=8
```

## Setup

1. **Local Environment**
### Option A: Virtual Environment (venv)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install pre-commit
pre-commit install
```

### Option B: Conda
```bash
# Create conda environment
conda create -n tmoe python=3.11
conda activate tmoe

# Install dependencies
pip install -r requirements.txt
pip install pre-commit
pre-commit install
```

2. **Cloud Infrastructure**
See [infra/README.md](infra/README.md) for environment setup instructions for Modal, AWS, WandB, and HuggingFace.
