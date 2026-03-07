import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.training.checkpoint import CheckpointManager
from src.metrics.training_metrics import TrainingMetricsTracker
from src.metrics.router_metrics import RouterMetricsTracker, GlobalSpecializationTracker


class Trainer:
    """Production trainer for T-MoE with FSDP support, gradient accumulation, and WandB logging."""

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
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        self._is_fsdp = isinstance(model, FSDP)
        self.model = model if self._is_fsdp else model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.config = config
        self.device = device

        self.max_steps = config.training.steps
        self.grad_accum_steps = getattr(
            config.training, "gradient_accumulation_steps", 1
        )
        self.log_interval = getattr(config.training, "log_interval", 10)
        self.eval_interval = getattr(config.training, "eval_interval", 100)
        self.save_interval = getattr(config.training, "save_interval", 500)
        self.clip_grad_norm = getattr(config.training, "clip_grad_norm", 1.0)

        self.scheduler = self._build_scheduler()

        # Precision via COMPUTE_DTYPE — GradScaler only for float16
        from src.training.precision import COMPUTE_DTYPE, needs_grad_scaler

        self.compute_dtype = COMPUTE_DTYPE
        if needs_grad_scaler() and "cuda" in str(self.device):
            from torch.amp import GradScaler

            self.scaler = GradScaler("cuda", enabled=True)
        else:
            self.scaler = None

        # Early stopping
        self.early_stopping_patience = getattr(
            config.training, "early_stopping_patience", None
        )
        self.early_stopping_counter = 0
        self.best_val_loss = float("inf")

        # Output
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.checkpoint_dir),
            keep_last_n=getattr(config.training, "keep_last_n_checkpoints", 3),
            save_best=True,
            execution_env=config.execution_env,
        )
        self.train_metrics = TrainingMetricsTracker(window_size=100)

        # Cache moe_layers ref before FSDP __getattr__ can interfere
        _unwrapped = model._fsdp_wrapped_module if self._is_fsdp else model
        self._moe_layers_ref = getattr(_unwrapped, "moe_layers", {})

        self.router_metrics = None
        self.global_specialization_trackers = {}
        if hasattr(_unwrapped, "moe_layers") and _unwrapped.moe_layers:
            first_moe = list(_unwrapped.moe_layers.values())[0]
            if hasattr(first_moe, "router"):
                self.router_metrics = RouterMetricsTracker(first_moe.router)

            vocab_size = getattr(
                getattr(_unwrapped, "config", None), "vocab_size", None
            )
            if vocab_size is None:
                backbone_cfg = getattr(
                    getattr(_unwrapped, "backbone", None), "config", None
                )
                vocab_size = (
                    getattr(backbone_cfg, "vocab_size", 50257)
                    if backbone_cfg
                    else 50257
                )

            for layer_name, layer in _unwrapped.moe_layers.items():
                if hasattr(layer, "router"):
                    tracker_key = (
                        f"layer_{layer_name}"
                        if not str(layer_name).startswith("layer_")
                        else str(layer_name)
                    )
                    self.global_specialization_trackers[tracker_key] = (
                        GlobalSpecializationTracker(
                            vocab_size=vocab_size,
                            num_experts=layer.router.num_experts,
                            device="cpu",
                        )
                    )

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
        print(
            f"Gradient accumulation: {self.grad_accum_steps} | "
            f"Effective batch: {self.config.training.batch_size * self.grad_accum_steps} | "
            f"dtype: {self.compute_dtype}"
        )

        while self.global_step < self.max_steps:
            step_start_time = time.time()
            accum_loss = 0.0

            for accum_step in range(self.grad_accum_steps):
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    self.epoch += 1
                    train_iterator = iter(self.train_dataloader)
                    batch = next(train_iterator)

                batch = {k: v.to(self.device) for k, v in batch.items()}

                logits, loss, moe_metrics = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch["labels"],
                    return_metrics=True,
                )

                if moe_metrics:
                    for layer_name, layer_metrics in moe_metrics.items():
                        if (
                            layer_name in self.global_specialization_trackers
                            and "indices" in layer_metrics
                        ):
                            self.global_specialization_trackers[layer_name].update(
                                token_ids=batch["input_ids"],
                                expert_indices=layer_metrics["indices"],
                            )

                loss = loss / self.grad_accum_steps

                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                accum_loss += loss.item()

            # Gradient clipping
            if self.clip_grad_norm > 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                if self._is_fsdp:
                    self.model.clip_grad_norm_(self.clip_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip_grad_norm
                    )

            # Optimizer step
            if self.scaler is not None:
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_stepped = old_scale <= self.scaler.get_scale()
            else:
                self.optimizer.step()
                optimizer_stepped = True
            self.optimizer.zero_grad()

            # Update MoE fatigue (cached ref, not through FSDP __getattr__)
            if self._moe_layers_ref:
                for moe_layer in self._moe_layers_ref.values():
                    if hasattr(moe_layer, "step"):
                        moe_layer.step()
                    if hasattr(moe_layer, "router") and hasattr(
                        moe_layer.router, "clear_aux_state"
                    ):
                        moe_layer.router.clear_aux_state()

            if self.scheduler is not None and optimizer_stepped:
                self.scheduler.step()

            # Metrics
            step_time = time.time() - step_start_time
            train_metrics = self.train_metrics.update(
                loss=accum_loss,
                batch_size=batch["input_ids"].shape[0] * self.grad_accum_steps,
                seq_len=batch["input_ids"].shape[1],
                step_time=step_time,
            )
            self.global_step += 1

            if self.global_step % self.log_interval == 0:
                self._log_metrics(train_metrics, moe_metrics)

            if self.val_dataloader and self.global_step % self.eval_interval == 0:
                val_metrics = self.evaluate()
                if self.early_stopping_patience:
                    if val_metrics["loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["loss"]
                        self.early_stopping_counter = 0
                    else:
                        self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        print(f"Early stopping at step {self.global_step}")
                        break

            if self.global_step % self.save_interval == 0:
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    step=self.global_step,
                    metrics=train_metrics,
                    is_best=train_metrics["loss"] < self.train_metrics.best_loss,
                    metadata={"epoch": self.epoch},
                )

        # Final checkpoint
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
        total_loss, num_batches = 0.0, 0

        for batch in self.val_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            _, loss, _ = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"],
                return_metrics=False,
            )
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        metrics = {
            "loss": avg_loss,
            "perplexity": torch.exp(torch.tensor(avg_loss)).item(),
        }

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
                    # Inject global specialization metrics
                    if layer_name in self.global_specialization_trackers:
                        global_stats = self.global_specialization_trackers[
                            layer_name
                        ].compute_metrics()
                        layer_metrics.update(global_stats)

                    # layer_metrics already contains layer-specific metrics computed by
                    # its own router (including its specific fatigue state).
                    self.router_metrics.log_to_wandb(
                        layer_metrics,
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
