import sys
import os
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.utils.config_loader import (
    load_experiment_config,
    list_available_experiments,
)
from src.utils.slurm import generate_sbatch_script, submit_sbatch_script
from src.utils.logging import initialize_wandb, finalize_wandb
from src.utils.experiment import (
    setup_experiment,
    build_model,
    build_dataloaders,
    build_optimizer,
)
from src.training.trainer import Trainer


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_main(args, overrides) -> None:
    """Main training entry point."""

    # Validate config argument
    if not args.config:
        print("\n" + "!" * 80)
        print("❌ Error: Missing required argument -c/--config <experiment_name>")
        print("!" * 80)
        print(
            "\nPlease provide an experiment configuration from the experiments/ directory."
        )
        print("Example: python train.py -c gptneo_125m_lora")
        list_available_experiments()
        print("!" * 80 + "\n")
        sys.exit(1)

    # Load configuration
    config = load_experiment_config(args.config, overrides)

    print("=" * 80)
    print("T-MoE Training Configuration")
    print("=" * 80)
    print(OmegaConf.to_yaml(config))
    print("=" * 80)

    # Generate SBATCH script if applicable (but only if not already running in SLURM)
    # Check for SLURM_JOB_ID to detect if we're already inside a submitted job
    running_in_slurm = os.getenv("SLURM_JOB_ID") is not None

    if not running_in_slurm:
        sbatch_script = generate_sbatch_script(config, config_name=args.config)
        if sbatch_script and config.compute.local.slurm.enabled:
            # Auto-submit to SLURM
            if submit_sbatch_script(sbatch_script):
                print("✅ Job submitted to SLURM. Exiting local process.")
                return  # Job submitted, exit

    # Set random seed
    set_seed(config.seed)
    print(f"✅ Random seed set to: {config.seed}")

    # Setup experiment directory
    output_dir = setup_experiment(config)
    print(f"✅ Output directory: {output_dir}")
    config_path = Path(output_dir) / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(config, f)
    print(f"✅ Configuration saved to: {config_path}")

    # Initialize WandB
    initialize_wandb(config, output_dir)

    # Build model
    print("\n" + "=" * 80)
    print("Building Model")
    print("=" * 80)
    model = build_model(config)
    print(f"✅ Model: {config.model.get('model_key', 'custom')}")
    print(f"   Total parameters: {model.get_total_params():,}")
    print(f"   Trainable parameters: {model.get_trainable_params():,}")

    # Build dataloaders
    print("\n" + "=" * 80)
    print("Loading Datasets")
    print("=" * 80)
    train_dataloader, val_dataloader = build_dataloaders(config)
    print(f"✅ Training batches: {len(train_dataloader)}")
    if val_dataloader:
        print(f"   Validation batches: {len(val_dataloader)}")

    # Build optimizer
    print("\n" + "=" * 80)
    print("Setting up Optimizer")
    print("=" * 80)
    optimizer = build_optimizer(model, config)
    print(f"✅ Optimizer: {config.training.optimizer}")
    print(f"   Learning rate: {config.training.lr}")

    # Resolve device
    device = config.device.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n✅ Training device: {device}")

    # Create trainer
    print("\n" + "=" * 80)
    print("Initializing Trainer")
    print("=" * 80)
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        config=config,
        output_dir=output_dir,
        device=device,
    )

    # Start training
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80 + "\n")

    try:
        final_metrics = trainer.train()

        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"✅ Final loss: {final_metrics['loss']:.4f}")
        print(f"   Best loss: {final_metrics['best_loss']:.4f}")
        print(f"   Best step: {final_metrics['best_step']}")
        print(f"   Output directory: {output_dir}")

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

    finally:
        finalize_wandb()
