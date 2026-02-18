import sys
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

from src.types import EXPERIMENTS_DIR


def load_experiment_config(config_name: str, overrides: List[str]) -> DictConfig:
    """
    Load and merge base config with experiment-specific config.

    This function:
    1. Initializes Hydra with the project root config directory
    2. Loads the base configuration (config.yaml)
    3. Loads the experiment-specific configuration from experiments/
    4. Merges them with experiment config taking precedence
    5. Applies any CLI overrides
    """
    # Initialize Hydra with project root as config path
    # config_path is relative to this file (src/utils/config_loader.py)
    # so we need to go up two directories to reach project root
    with hydra.initialize(version_base=None, config_path="../.."):
        # Load base configuration
        base_cfg = hydra.compose(config_name="config", overrides=overrides)

        # Load experiment configuration
        exp_path = EXPERIMENTS_DIR / f"{config_name}.yaml"
        if not exp_path.exists():
            print(f"\n{'!' * 80}")
            print(f"❌ Error: Experiment config '{exp_path}' not found.")
            print("!" * 80)
            print("\nAvailable experiments:")
            if EXPERIMENTS_DIR.exists():
                for f in sorted(EXPERIMENTS_DIR.glob("*.yaml")):
                    print(f"  - {f.stem}")
            print("!" * 80 + "\n")
            sys.exit(1)

        # Load and merge experiment config
        exp_cfg = OmegaConf.load(exp_path)
        config = OmegaConf.merge(base_cfg, exp_cfg)

        # Ensure experiment_name is set from config name if not specified
        if "experiment_name" not in exp_cfg:
            config.experiment_name = config_name

    return config


def list_available_experiments() -> None:
    """
    List all available experiment configurations.
    """
    print("\nAvailable experiments:")
    if EXPERIMENTS_DIR.exists():
        for f in sorted(EXPERIMENTS_DIR.glob("*.yaml")):
            print(f"  - {f.stem}")
    else:
        print("  (No experiments directory found)")
    print()
