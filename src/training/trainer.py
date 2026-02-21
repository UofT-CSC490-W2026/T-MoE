import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch import nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.training.checkpoint import CheckpointManager
from src.metrics.training_metrics import TrainingMetricsTracker
from src.metrics.router_metrics import RouterMetricsTracker


class Trainer:
    """
    Production-grade trainer for T-MoE models.

    Features:
    - Gradient accumulation
    - Mixed precision training (AMP)
    - Checkpoint management (best/periodic)
    - Early stopping
    - Comprehensive logging (WandB)
    - Router metrics tracking
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        config: Any,
        output_dir: str,
        device: str = "cuda",
    ):
        """
        Initialize trainer.

        Args:
            model: T-MoE model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            optimizer: Optimizer
            config: Experiment configuration
            output_dir: Output directory for checkpoints/logs
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.config = config
        self.device = device

        # Training config (MUST be set before _build_scheduler)
        self.max_steps = config.training.steps
        self.grad_accum_steps = getattr(
            config.training, "gradient_accumulation_steps", 1
        )
        self.log_interval = getattr(config.training, "log_interval", 10)
        self.eval_interval = getattr(config.training, "eval_interval", 100)
        self.save_interval = getattr(config.training, "save_interval", 500)
        self.clip_grad_norm = getattr(config.training, "clip_grad_norm", 1.0)

        # Learning rate scheduler (depends on self.max_steps)
        self.scheduler = self._build_scheduler()

        # Mixed precision
        self.use_amp = getattr(config.training, "use_amp", True)

        # Determine device-aware mixed precision settings
        if self.use_amp:
            if self.device == "cuda":
                self.scaler = GradScaler("cuda", enabled=True)
                self.autocast_device = "cuda"
            elif "cpu" in self.device:
                # CPU mixed precision (BFloat16) doesn't use a scaler
                self.scaler = GradScaler("cpu", enabled=False)
                self.autocast_device = "cpu"
                print(
                    "ℹ️  CPU Mixed Precision enabled (using BFloat16, no scaling required)"
                )
            else:
                self.scaler = GradScaler(enabled=False)
                self.autocast_device = self.device
        else:
            self.scaler = GradScaler(enabled=False)
            self.autocast_device = self.device

        # Early stopping
        self.early_stopping_patience = getattr(
            config.training, "early_stopping_patience", None
        )
        self.early_stopping_counter = 0
        self.best_val_loss = float("inf")

        # Output directory
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"

        # Managers and trackers
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            keep_last_n=getattr(config.training, "keep_last_n_checkpoints", 3),
            save_best=True,
            execution_env=config.execution_env,
        )

        self.train_metrics = TrainingMetricsTracker(window_size=100)

        # Router metrics (if model has router)
        self.router_metrics = None
        if hasattr(model, "moe_layers") and model.moe_layers:
            # Get first MoE layer's router
            first_moe = list(model.moe_layers.values())[0]
            if hasattr(first_moe, "router"):
                self.router_metrics = RouterMetricsTracker(first_moe.router)

        # Training state
        self.global_step = 0
        self.epoch = 0

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler from config."""
        if not hasattr(self.config.training, "lr_scheduler"):
            return None

        scheduler_type = self.config.training.lr_scheduler

        if scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            kwargs = getattr(self.config.training, "lr_scheduler_kwargs", {})
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_steps,
                eta_min=kwargs.get("eta_min", 1e-5),
            )
        elif scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR

            kwargs = getattr(self.config.training, "lr_scheduler_kwargs", {})
            return LinearLR(
                self.optimizer,
                start_factor=kwargs.get("start_factor", 1.0),
                end_factor=kwargs.get("end_factor", 0.0),
                total_iters=self.max_steps,
            )
        elif scheduler_type == "step":
            from torch.optim.lr_scheduler import StepLR

            kwargs = getattr(self.config.training, "lr_scheduler_kwargs", {})
            return StepLR(
                self.optimizer,
                step_size=kwargs.get("step_size", 1000),
                gamma=kwargs.get("gamma", 0.1),
            )
        else:
            print(
                f"⚠️  Unknown scheduler type: {scheduler_type}. Continuing without scheduler."
            )
            return None

    def train(self) -> Dict[str, Any]:
        """
        Run training loop.

        Returns:
            Final training metrics
        """
        self.model.train()
        train_iterator = iter(self.train_dataloader)

        print(f"Starting training for {self.max_steps} steps...")
        print(f"Gradient accumulation steps: {self.grad_accum_steps}")
        print(
            f"Effective batch size: {self.config.training.batch_size * self.grad_accum_steps}"
        )
        print(f"Mixed precision: {self.use_amp}")

        while self.global_step < self.max_steps:
            step_start_time = time.time()

            # Accumulate gradients
            accum_loss = 0.0
            for accum_step in range(self.grad_accum_steps):
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    # Reset iterator for new epoch
                    self.epoch += 1
                    train_iterator = iter(self.train_dataloader)
                    batch = next(train_iterator)

                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass with AMP
                with autocast(device_type=self.autocast_device, enabled=self.use_amp):
                    logits, loss, moe_metrics = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch[
                            "labels"
                        ],  # Now properly includes -100 for padding
                        return_metrics=True,
                    )

                    # Scale loss for gradient accumulation
                    loss = loss / self.grad_accum_steps

                # Backward pass
                self.scaler.scale(loss).backward()
                accum_loss += loss.item()

            # Optimizer step
            if self.clip_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.clip_grad_norm,
                )

            # Step optimizer with gradient scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # Update MoE router fatigue after optimizer step
            # This applies the accumulated usage from gradient accumulation
            if hasattr(self.model, "moe_layers") and self.model.moe_layers:
                for moe_layer in self.model.moe_layers.values():
                    if hasattr(moe_layer, "step"):
                        moe_layer.step()
                    if hasattr(moe_layer, "router") and hasattr(
                        moe_layer.router, "clear_aux_state"
                    ):
                        moe_layer.router.clear_aux_state()

            if self.scheduler is not None:
                self.scheduler.step()

            # Update metrics
            step_time = time.time() - step_start_time
            batch_size = batch["input_ids"].shape[0]
            seq_len = batch["input_ids"].shape[1]

            train_metrics = self.train_metrics.update(
                loss=accum_loss,
                batch_size=batch_size * self.grad_accum_steps,
                seq_len=seq_len,
                step_time=step_time,
            )

            self.global_step += 1

            # Logging
            if self.global_step % self.log_interval == 0:
                self._log_metrics(train_metrics, moe_metrics)

            # Validation
            if self.val_dataloader and self.global_step % self.eval_interval == 0:
                val_metrics = self.evaluate()

                # Check for improvement
                if self.early_stopping_patience:
                    if val_metrics["loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["loss"]
                        self.early_stopping_counter = 0
                    else:
                        self.early_stopping_counter += 1

                    if self.early_stopping_counter >= self.early_stopping_patience:
                        print(f"Early stopping triggered at step {self.global_step}")
                        break

            # Checkpointing
            if self.global_step % self.save_interval == 0:
                is_best = train_metrics["loss"] < self.train_metrics.best_loss
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=self.global_step,
                    metrics=train_metrics,
                    is_best=is_best,
                    metadata={"epoch": self.epoch},
                )

        # Save final checkpoint
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            step=self.global_step,
            metrics=train_metrics,
            is_best=False,
            metadata={"epoch": self.epoch, "final": True},
        )

        print(f"Training complete! Best loss: {self.train_metrics.best_loss:.4f}")
        return train_metrics

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """
        Run evaluation on validation set.

        Returns:
            Validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        num_batches = 0

        for batch in self.val_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with autocast(device_type=self.autocast_device, enabled=self.use_amp):
                _, loss, _ = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],  # Now properly includes -100 for padding
                    return_metrics=False,
                )

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        metrics = {
            "loss": avg_loss,
            "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
        }

        # Log validation metrics
        if WANDB_AVAILABLE and wandb.run:
            wandb.log(
                {f"val/{k}": v for k, v in metrics.items()}, step=self.global_step
            )

        self.model.train()
        return metrics

    def _log_metrics(
        self,
        train_metrics: Dict[str, Any],
        moe_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log metrics to console and WandB."""
        # Console logging
        lr = (
            self.scheduler.get_last_lr()[0]
            if self.scheduler
            else self.config.training.lr
        )
        log_str = f"Step {self.global_step}/{self.max_steps} | "
        log_str += f"Loss: {train_metrics['loss']:.4f} | "
        log_str += f"PPL: {train_metrics['perplexity']:.2f} | "
        log_str += f"LR: {lr:.2e} | "
        log_str += f"Tokens/s: {train_metrics.get('tokens_per_sec', 0):.0f}"
        print(log_str)

        # Log learning rate to WandB
        if WANDB_AVAILABLE and wandb.run:
            wandb.log({"train/lr": lr}, step=self.global_step)

        # WandB logging
        self.train_metrics.log_to_wandb(train_metrics, step=self.global_step)

        # Router metrics
        if moe_metrics and self.router_metrics:
            for layer_name, layer_metrics in moe_metrics.items():
                if "indices" in layer_metrics and "weights" in layer_metrics:
                    router_metrics = self.router_metrics.compute_all_metrics(
                        indices=layer_metrics["indices"],
                        weights=layer_metrics["weights"],
                    )
                    self.router_metrics.log_to_wandb(
                        router_metrics,
                        step=self.global_step,
                        prefix=f"router/{layer_name}",
                    )

    def resume_from_checkpoint(self, checkpoint_path: Optional[str] = None) -> None:
        """Resume training from checkpoint."""
        metadata = self.checkpoint_manager.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            checkpoint_path=Path(checkpoint_path) if checkpoint_path else None,
        )

        self.global_step = metadata["step"]
        self.epoch = metadata.get("metadata", {}).get("epoch", 0)

        print(f"✅ Resumed from checkpoint at step {self.global_step}")
