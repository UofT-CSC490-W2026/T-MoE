# T-MoE: Thermodynamic Mixture-of-Experts

T-MoE is a research project implementing biological-inspired **Metabolic Routing** for Mixture-of-Experts (MoE) models. Instead of traditional auxiliary load-balancing losses, it uses fatigue mechanics and homeostatic dynamics to manage expert usage and architectural evolution.

## Core Concepts

For a deep dive into the mathematical foundations, routing potentials, and fatigue dynamics, see: **[Equations](file:///Users/aviralbhardwaj/Documents/Github/T-MoE/Equations.md)**

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.9+
- [PyTorch](https://pytorch.org/) (2.6+)
- [Weights & Biases](https://wandb.ai/) (for logging)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aviralbhardwaj/T-MoE.git
   cd T-MoE
   ```

2. Create a virtual environment (use either conda or venv):

   **Using conda:**
   ```bash
   conda create -n tmoe python=3.14
   conda activate tmoe
   ```

   **Using venv:**
   ```bash
   python3.14 -m venv tmoe-env
   source tmoe-env/bin/activate  # On macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

To verify the installation and core router components:
```bash
pytest tests/
```

## Project Structure

- `src/routers/`: Implementation of Metabolic and Standard routers.
- `configs/`: Hydra-style dataclass configurations.
- `src/metrics/`: Specialized MoE performance and fatigue tracking.
- `Equations.md`: Comprehensive technical documentation.
