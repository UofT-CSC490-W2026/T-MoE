# T-MoE: Thermodynamic Mixture-of-Experts

[![Tests](https://github.com/UofT-CSC490-W2026/T-MoE/actions/workflows/test.yml/badge.svg)](https://github.com/UofT-CSC490-W2026/T-MoE/actions/workflows/test.yml)
[![Lint](https://github.com/UofT-CSC490-W2026/T-MoE/actions/workflows/lint.yml/badge.svg)](https://github.com/UofT-CSC490-W2026/T-MoE/actions/workflows/lint.yml)

T-MoE is a research project implementing biological-inspired **Metabolic Routing** for Mixture-of-Experts (MoE) models. Instead of traditional auxiliary load-balancing losses, it uses fatigue mechanics and homeostatic dynamics to manage expert usage and architectural evolution.

## 🌟 Key Features

- **Metabolic Routing**: Biological-inspired routing with fatigue dynamics and homeostatic recovery
- **Multiple Router Architectures**: 5 different router implementations (Metabolic, Standard, Top-K, Switch, DynMoE)
- **Advanced Metrics Tracking**: Comprehensive monitoring of routing entropy, load balancing (Gini coefficient), fatigue statistics
- **Hardware-Aware**: Distance penalties for expert placement and silicon tax calculations
- **Age-Aware Dynamics**: Newborn expert warmup and adaptive cost scaling
- **Elastic Architecture**: Support for dynamic expert pools with living expert mechanics
- **Weights & Biases Integration**: Built-in logging for experiment tracking

## 📚 Core Concepts

For a deep dive into the mathematical foundations, routing potentials, and fatigue dynamics, see:
- **[Equations.md](./Equations.md)** - Comprehensive technical documentation with all mathematical formulas
- **Topics covered**: Homeostatic routing potential, age-aware fatigue dynamics, elastic architecture, GRPO optimization, and more

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.9+ (tested with 3.10)
- [PyTorch](https://pytorch.org/) 2.10+
- [Weights & Biases](https://wandb.ai/) (for logging)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/UofT-CSC490-W2026/T-MoE.git
   cd T-MoE
   ```

2. **Create a virtual environment** (choose conda or venv):

   **Using conda:**
   ```bash
   conda create -n tmoe python=3.10
   conda activate tmoe
   ```

   **Using venv:**
   ```bash
   python3 -m venv tmoe-env
   source tmoe-env/bin/activate        # On macOS/Linux
   tmoe-env\Scripts\activate           # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## 🚀 Quick Start

### Basic Usage

```python
import torch
from configs.router import MetabolicRouterConfig
from src.routers.metabolic import MetabolicRouter

# Create router configuration
config = MetabolicRouterConfig(
    hidden_dim=256,
    num_experts=8,
    top_k=2,
    lambda_metabolic=0.5,  # Fatigue pressure
    mu_silicon=0.1,        # Hardware distance penalty
    gamma_recovery=0.95,   # Recovery rate
)

# Initialize router
router = MetabolicRouter(config).to('cuda')

# Forward pass with input tensor [batch, seq_len, hidden_dim]
x = torch.randn(4, 32, 256, device='cuda')
weights, indices, metrics = router(x, return_metrics=True)

# weights: [batch, seq_len, top_k] - routing weights
# indices: [batch, seq_len, top_k] - selected expert indices
# metrics: dict with routing statistics
```

### Computing Metrics

```python
# Get comprehensive routing metrics
all_metrics = router.metrics_tracker.compute_all_metrics(indices, weights)

# Log to Weights & Biases
router.metrics_tracker.log_to_wandb(all_metrics, step=100)

# Available metrics:
# - expert_entropy, normalized_entropy
# - expert_counts, expert_probs
# - gini_coefficient (load balancing)
# - effective_experts
# - fatigue_mean, fatigue_std (for MetabolicRouter)
```

## 📦 Available Router Types

| Router | Description | Key Parameters | Use Case |
|--------|-------------|----------------|----------|
| **MetabolicRouter** | Biological-inspired with fatigue dynamics | `lambda_metabolic`, `mu_silicon`, `gamma_recovery` | Research on adaptive routing |
| **StandardRouter** | Classic Top-K with softmax | `top_k`, `aux_loss_alpha` | Baseline comparisons |
| **TopKRouter** | Standard softmax + top-k selection | `top_k`, `temperature` | Simple top-k routing |
| **SwitchRouter** | Top-1 specialist routing | `noise_std` | Single expert per token |
| **DynMoERouter** | Sigmoid threshold-based | `threshold`, `temperature` | Dynamic expert count |

All routers inherit from `BaseRouter` and support:
- Device placement (CPU/CUDA/MPS)
- Deterministic behavior in eval mode
- Optional metrics tracking
- Configurable normalization

## ⚙️ Configuration

The project uses dataclass-based configurations located in `configs/`:

### Router Configuration

```python
from configs.router import MetabolicRouterConfig

config = MetabolicRouterConfig(
    # Base parameters
    hidden_dim=256,
    num_experts=8,
    top_k=2,
    temperature=1.0,
    noise_std=0.0,
    
    # Metabolic-specific
    lambda_metabolic=0.5,      # Fatigue penalty weight
    mu_silicon=0.1,            # Distance penalty weight
    gamma_recovery=0.95,       # Fatigue recovery rate (0-1)
    beta_cost=1.0,             # Base activation cost
    warmup_steps=1000,         # Newborn expert warmup
    normalize_inputs=True,     # L2 normalize inputs
    normalize_weights=True,    # L2 normalize prototypes
)
```

### Dataset Configuration

```python
from configs.dataset import DatasetConfig

dataset_config = DatasetConfig(
    dataset_key="wikitext-2",  # From catalog
    sequence_length=512,
    streaming=False,
)
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Suite
```bash
pytest tests/routers/test_metabolic_router.py -v
```

### Run with Coverage
```bash
pytest --cov=src tests/
```

### Test Structure
- `tests/routers/` - Router implementation tests (40+ tests)
  - Shape validation
  - Weight normalization
  - Fatigue dynamics
  - Metrics computation
  - Device compatibility (CPU/CUDA/MPS)
- `tests/conftest.py` - Shared fixtures and test utilities

## 📁 Project Structure

```
T-MoE/
├── src/
│   ├── routers/          # Router implementations
│   │   ├── base.py       # BaseRouter abstract class
│   │   ├── metabolic.py  # MetabolicRouter with fatigue
│   │   ├── standard.py   # StandardRouter, SwitchRouter
│   │   └── dynmoe.py     # DynMoERouter with thresholds
│   ├── metrics/          # RouterMetricsTracker
│   │   └── router_metrics.py
│   ├── core/             # Router registry system
│   │   └── registry.py
│   └── utils/            # Utilities (checkpointing, etc.)
├── configs/              # Dataclass configurations
│   ├── router.py         # Router configs
│   ├── dataset.py        # Dataset configs
│   └── base.py           # Base config classes
├── tests/                # Comprehensive test suite
│   ├── routers/          # Router-specific tests
│   └── conftest.py       # Shared fixtures
├── catalog/              # Dataset catalog
│   └── dataset_catalog.py
├── Equations.md          # Mathematical documentation
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 📊 Metrics & Monitoring

The `RouterMetricsTracker` computes:

- **Routing Entropy**: Shannon entropy and normalized entropy (diversity measure)
- **Load Balancing**: Gini coefficient (0=perfect, 1=imbalanced)
- **Effective Experts**: exp(entropy) - actual number of experts used
- **Expert Usage**: Per-expert counts and probability distributions
- **Fatigue Statistics**: Mean, std, min, max, per-expert (MetabolicRouter only)

All metrics support Weights & Biases logging with histograms and scalar tracking.

## 🔧 Development

### Code Quality Tools

The project uses:
- **black** - Code formatting
- **ruff** - Fast linting
- **pre-commit** - Git hooks for automatic checks
- **pytest** - Testing framework

### Running Linters

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Run pre-commit hooks manually
pre-commit run --all-files
```

### CI/CD

- **GitHub Actions** workflows for automated testing and linting
- Tests run on every push and pull request
- Python 3.10 on Ubuntu latest

## 📖 Documentation

- **[Equations.md](./Equations.md)** - Complete mathematical foundations including:
  - Homeostatic routing potential (Eq. 1)
  - Age-aware fatigue dynamics (Eq. 2)
  - Adaptive cost scaling (Eq. 3)
  - Elastic architecture (mitosis, apoptosis, fusion)
  - GRPO reinforcement learning
  - LoRA parameter optimization

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linters (`pytest tests/` and `ruff check .`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is part of the UofT CSC490 W2026 course. Please refer to the course guidelines for usage and distribution.

## 🙏 Acknowledgments

- Inspired by biological metabolic systems and homeostatic dynamics
- Built on PyTorch for efficient tensor operations
- Uses Weights & Biases for experiment tracking

---

For questions or issues, please open an issue on GitHub or contact the maintainers.
