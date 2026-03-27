"""Test that a final checkpoint is saved even when val_loader is None."""

import torch
from unittest.mock import patch

from src.training.checkpoint import CheckpointManager


def test_final_checkpoint_saved_when_val_loader_is_none(tmp_path):
    """
    Regression test: when training ends without a validation loader,
    the final checkpoint must still be saved with train_loss metrics.
    This covers the `else` branch of the "Final save" block in train.py.
    """
    ckpt_dir = tmp_path / "checkpoints"
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: 1.0)

    ckpt_manager = CheckpointManager(
        checkpoint_dir=str(ckpt_dir),
        keep_last_n=3,
        save_best=True,
        trainable_only=False,
    )

    max_steps = 5
    accum_loss = 2.345

    # Simulate the "Final save" else branch from train.py:
    #   val_loader is None → save with train_loss only, is_best=False
    with (
        patch("src.training.checkpoint.is_main_process", return_value=True),
        patch("torch.distributed.is_initialized", return_value=False),
    ):
        saved_path = ckpt_manager.save_checkpoint(
            model,
            optimizer,
            scheduler,
            step=max_steps,
            metrics={"train_loss": accum_loss},
            is_best=False,
        )

    # Checkpoint file was created
    assert saved_path.exists()
    assert saved_path.name == f"checkpoint_step_{max_steps}.pt"

    # Load and verify contents
    ckpt = torch.load(saved_path, weights_only=False)
    assert ckpt["step"] == max_steps
    assert ckpt["metrics"]["train_loss"] == accum_loss
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert "scheduler_state_dict" in ckpt

    # No best_model.pt should exist since is_best=False
    assert not (ckpt_dir / "best_model.pt").exists()

    # JSON sidecar was written
    assert (ckpt_dir / f"checkpoint_step_{max_steps}.json").exists()


def test_final_checkpoint_not_marked_best_without_validation(tmp_path):
    """
    When val_loader is None, the final save uses is_best=False,
    so best_model.pt must NOT be created.
    """
    ckpt_dir = tmp_path / "checkpoints"
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    ckpt_manager = CheckpointManager(
        checkpoint_dir=str(ckpt_dir),
        keep_last_n=3,
        save_best=True,
        trainable_only=False,
    )

    with (
        patch("src.training.checkpoint.is_main_process", return_value=True),
        patch("torch.distributed.is_initialized", return_value=False),
    ):
        ckpt_manager.save_checkpoint(
            model,
            optimizer,
            step=10,
            metrics={"train_loss": 1.0},
            is_best=False,
        )

    assert not (ckpt_dir / "best_model.pt").exists()
    assert (ckpt_dir / "checkpoint_step_10.pt").exists()
