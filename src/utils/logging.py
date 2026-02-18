import os
from datetime import datetime

from omegaconf import DictConfig, OmegaConf

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def initialize_wandb(config: DictConfig, output_dir: str) -> None:
    """
    Initialize Weights & Biases logging.
    """
    if not WANDB_AVAILABLE:
        print("⚠️  WandB not installed - logging disabled")
        return

    if not config.logging.enabled:
        print("ℹ️  WandB logging disabled via config")
        return

    # Resolve entity from config or environment
    entity = config.logging.entity or os.getenv("WANDB_ENTITY", None)

    # Generate run name with timestamp
    run_name = f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Resolve project from environment or config, with fallback
    project = os.getenv("WANDB_PROJECT") or config.logging.project or "T-MoE"

    # Initialize WandB
    wandb.init(
        project=project,
        entity=entity,
        group=config.logging.group,
        name=run_name,
        tags=config.logging.tags,
        config=OmegaConf.to_container(config, resolve=True),
        mode=config.logging.mode,
        notes=config.logging.notes,
        dir=output_dir,
    )

    print(f"✅ WandB initialized: {wandb.run.name}")
    print(f"   View at: {wandb.run.url}")


def finalize_wandb() -> None:
    """
    Finalize WandB run if it exists.
    """
    if WANDB_AVAILABLE and wandb.run:
        wandb.finish()
        print("✅ WandB run finished")
