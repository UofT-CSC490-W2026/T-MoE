import sys
from pathlib import Path
from typing import List

from omegaconf import DictConfig, OmegaConf

from src.project_types import EXPERIMENTS_DIR


def load_experiment_config(
    config_path_or_name: str, overrides: List[str] = None
) -> DictConfig:
    """
    Load experiment config from a YAML file and apply CLI overrides.

    Works like Hydra's dotlist overrides but is completely DDP-safe
    (no global Hydra initialization state).

    Args:
        config_path_or_name: Either a full path to a .yaml file OR a bare
            experiment name (e.g. "gptneo_125m_metabolic") resolved against
            the experiments/ directory.
        overrides: List of OmegaConf dotlist overrides, e.g.
            ["training.lr=1e-4", "training.batch_size=8"]

    Returns:
        Merged DictConfig ready for use.
    """
    # Resolve config path
    path = Path(config_path_or_name)
    if not path.suffix:
        # Bare name: look in experiments/ directory
        path = EXPERIMENTS_DIR / f"{config_path_or_name}.yaml"

    if not path.exists():
        print(f"\nError: Experiment config not found at '{path}'")
        print("\nAvailable experiments:")
        if EXPERIMENTS_DIR.exists():
            for f in sorted(EXPERIMENTS_DIR.glob("*.yaml")):
                print(f"  - {f.stem}")
        sys.exit(1)

    cfg = OmegaConf.load(path)

    # Apply CLI overrides
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    # Ensure experiment_name is always set
    if "experiment_name" not in cfg:
        cfg.experiment_name = path.stem

    return cfg
