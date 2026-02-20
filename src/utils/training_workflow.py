"""
Shared training execution utilities for T-MoE.

This module provides common training workflow functions that can be reused
across different execution backends (AWS Batch, Modal, local).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import TYPE_CHECKING, Tuple, Dict, Any

import numpy as np
import torch
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig


def execute_training_workflow(
    experiment_config: DictConfig,
    cache_dir: str,
) -> Tuple[str, Dict[str, Any]]:
    """
    Execute the complete training workflow.

    This is the shared training logic used by all backends (AWS Batch, Modal, local).
    It handles:
    - Random seed setting
    - Experiment directory setup
    - Model/dataloader/optimizer building
    - Training execution
    - WandB integration

    Args:
        experiment_config: Experiment configuration (from load_experiment_config)
        cache_dir: Directory containing the dataset cache

    Returns:
        Tuple of (output_dir, final_metrics)
    """
    from src.utils.experiment import (
        setup_experiment,
        build_model,
        build_dataloaders,
        build_optimizer,
    )
    from src.utils.logging import initialize_wandb, finalize_wandb
    from src.training.trainer import Trainer

    # Update cache directory in config
    OmegaConf.update(experiment_config, "compute.aws.cache_dir", cache_dir, merge=True)

    # Set random seed for reproducibility
    seed = experiment_config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"✅ Random seed set to: {seed}")

    # Setup experiment directory
    output_dir = setup_experiment(experiment_config)
    print(f"✅ Output directory: {output_dir}")

    # Save configuration
    config_path = Path(output_dir) / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(experiment_config, f)
    print(f"✅ Configuration saved to: {config_path}")

    # Initialize WandB
    initialize_wandb(experiment_config, output_dir)

    try:
        # Build model
        print("\n" + "=" * 70)
        print("Building Model")
        print("=" * 70)
        model = build_model(experiment_config)
        print(f"✅ Model: {experiment_config.model.get('model_key', 'custom')}")
        print(f"   Total parameters: {model.get_total_params():,}")
        print(f"   Trainable parameters: {model.get_trainable_params():,}")

        # Build dataloaders
        print("\n" + "=" * 70)
        print("Loading Datasets")
        print("=" * 70)
        train_dataloader, val_dataloader = build_dataloaders(experiment_config)
        print(f"✅ Training batches: {len(train_dataloader)}")
        if val_dataloader:
            print(f"   Validation batches: {len(val_dataloader)}")

        # Build optimizer
        print("\n" + "=" * 70)
        print("Setting up Optimizer")
        print("=" * 70)
        optimizer = build_optimizer(model, experiment_config)
        print(f"✅ Optimizer: {experiment_config.training.optimizer}")
        print(f"   Learning rate: {experiment_config.training.lr}")

        # Determine device
        device = experiment_config.device.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n✅ Training device: {device}")

        # Create trainer
        print("\n" + "=" * 70)
        print("Initializing Trainer")
        print("=" * 70)
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            config=experiment_config,
            output_dir=output_dir,
            device=device,
        )

        # Start training
        print("\n" + "=" * 70)
        print("Starting Training")
        print("=" * 70 + "\n")

        final_metrics = trainer.train()

        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"✅ Final loss: {final_metrics['loss']:.4f}")
        print(f"   Best loss: {final_metrics['best_loss']:.4f}")
        print(f"   Best step: {final_metrics['best_step']}")
        print(f"   Output directory: {output_dir}")

        return output_dir, final_metrics

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        raise

    finally:
        finalize_wandb()
