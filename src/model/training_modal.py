"""
Simplified Modal training script for GPT-Neo 125M on WikiText-2.

This script follows the structural patterns of the visight training pipeline
but removes all complexity related to S3, dataset staging, YOLO, ONNX export,
profiling, and custom configurations.

Run with:
  - Dev (personal workspace): modal run src/model/train.py
  - Prod (t-moe workspace): MODAL_ENV=prod modal run src/model/train.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import modal

# Add current directory to path for config import
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

# ----------------------------
# Configuration
# ----------------------------
from config import ENV, get_config, get_app_name, MODEL_NAME, DATASET_NAME

CONFIG = get_config()

# ----------------------------
# Modal Volume for persistence
# ----------------------------
volume = modal.Volume.from_name("tmoe-training-artifacts", create_if_missing=True)
VOLUME_PATH = Path("/root/data")
OUTPUT_DIR = VOLUME_PATH / "models"

# ----------------------------
# Modal app & image
# ----------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
    )
    .add_local_file(
        str(_here / "config.py"),
        remote_path="/root/config.py"
    )
)

app = modal.App(get_app_name(), image=image)

# ----------------------------
# Training function
# ----------------------------
@app.function(
    gpu=CONFIG["gpu"],
    cpu=CONFIG["cpu"],
    timeout=CONFIG["timeout"],
    volumes={VOLUME_PATH: volume},
)
def train_gpt_neo(
    epochs: int = 3,
    batch_size: int = 8,
    max_length: int = 128,
    max_train_samples: Optional[int] = None,
):
    """
    Fine-tune GPT-Neo 125M on WikiText-2 dataset.
    
    Args:
        epochs: Number of training epochs
        batch_size: Per-device training batch size
        max_length: Maximum sequence length for tokenization
        max_train_samples: Maximum number of training samples to use (None = use all)
    """
    import datasets
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        Trainer,
        TrainingArguments,
    )

    print(f"=" * 60)
    print(f"T-MoE Training - Environment: {ENV}")
    print(f"=" * 60)
    print(f"Training configuration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max length: {max_length}")
    if max_train_samples:
        print(f"  Max train samples: {max_train_samples}")
    print(f"Infrastructure:")
    print(f"  GPU: {CONFIG['gpu']}")
    print(f"  CPU: {CONFIG['cpu']}")
    print(f"  Timeout: {CONFIG['timeout']}s")
    print(f"=" * 60)

    # Load tokenizer and model
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set pad token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Load dataset
    print("\nLoading dataset...")
    dataset = datasets.load_dataset(DATASET_NAME)

    # Tokenization function
    def tokenize_function(examples):
        # Tokenize the text
        result = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        # For causal language modeling, labels are the same as input_ids
        result["labels"] = result["input_ids"].copy()
        return result

    # Tokenize dataset
    print("\nTokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    
    # Limit training samples if specified
    if max_train_samples is not None:
        print(f"\nLimiting training data to {max_train_samples} samples...")
        tokenized_dataset["train"] = tokenized_dataset["train"].select(range(min(max_train_samples, len(tokenized_dataset["train"]))))
        print(f"Training samples: {len(tokenized_dataset['train'])}")

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=100,
        save_steps=500,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=500,
        logging_dir=str(OUTPUT_DIR / "logs"),
    )

    # Initialize trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = OUTPUT_DIR / f"gpt-neo-{timestamp}"
    
    print(f"\nSaving final model to {model_dir}...")
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    
    # Commit volume to persist changes
    volume.commit()

    print("\nTraining complete!")
    print(f"Model saved to volume: {model_dir}")
    print(f"Volume name: tmoe-training-artifacts")
    print(f"Access with: modal volume ls tmoe-training-artifacts")


# ----------------------------
# Local entrypoint
# ----------------------------
@app.local_entrypoint()
def main(
    epochs: int = 3,
    batch_size: int = 8,
    max_length: int = 128,
    max_train_samples: Optional[int] = None,
):
    """
    Kick off a training job on Modal.
    
    Args:
        epochs: Number of training epochs (default: 3)
        batch_size: Per-device training batch size (default: 8)
        max_length: Maximum sequence length for tokenization (default: 128)
    """
    train_gpt_neo.remote(
        epochs=epochs,
        batch_size=batch_size,
        max_length=max_length,
        max_train_samples=max_train_samples,
    )
