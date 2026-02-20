import os
from pathlib import Path
from datetime import datetime
from typing import Any, Tuple, Optional

from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

from configs.model import ModelConfig
from configs.dataset import DatasetConfig

from src.core import ModelRegistry
from src.layers.lora_moe import LoRAMoELayer
from src.experts.lora import LoRAConfig
from src.project_types import RouterType, ExecutionEnv, ExpertType

# Side-effect import: triggers @ModelRegistry.register decorator
from src.models import gpt_neo  # noqa: F401


def _load_from_local_files(
    cache_dir: str,
    train_split: str = "train",
    eval_split: Optional[str] = "validation",
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Attempt to load datasets from pre-downloaded files in cache_dir.

    Looks for files named ``{split}.jsonl`` or ``{split}.parquet`` inside
    *cache_dir* (these are the files produced by the fallback ingestion
    pipeline and downloaded from S3 by the orchestrator).

    Args:
        cache_dir: Local directory containing data files.
        train_split: Name of the training split file (without extension).
        eval_split: Name of the eval split file (without extension), or None.

    Returns:
        (train_dataset, val_dataset) — both may be None if no files found.
    """
    cache_path = Path(cache_dir)

    def _find_and_load(split_name: str):
        """Find a data file for the given split and load it."""
        for ext, loader in [(".jsonl", "json"), (".parquet", "parquet")]:
            candidate = cache_path / f"{split_name}{ext}"
            if candidate.is_file():
                print(f"  Loading {split_name} from local file: {candidate}")
                ds = load_dataset(loader, data_files=str(candidate), split="train")
                return ds
        return None

    train_ds = _find_and_load(train_split)
    if train_ds is None:
        print(f"  No local data files found for split '{train_split}' in {cache_dir}")
        return None, None

    val_ds = None
    if eval_split:
        val_ds = _find_and_load(eval_split)
        if val_ds is None:
            print(f"  Warning: No local data file for eval split '{eval_split}'")

    return train_ds, val_ds


def setup_experiment(config: DictConfig) -> str:
    """
    Setup experiment directory structure based on execution environment.
    """
    # Get output root based on execution environment
    if config.execution_env == ExecutionEnv.AWS:
        output_root = config.compute.aws.output_root
    else:
        output_root = config.compute.local.output_root

    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = (
        Path(output_root) / "experiments" / f"{config.experiment_name}_{timestamp}"
    )

    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "logs").mkdir(parents=True, exist_ok=True)
    (exp_dir / "wandb").mkdir(parents=True, exist_ok=True)

    return str(exp_dir)


def _get_model_config(config: DictConfig) -> "ModelConfig":
    """
    Create ModelConfig from experiment configuration.
    """
    return ModelConfig(
        model_type=config.model.get("model_type", "gpt_neo"),
        variant=config.model.get("variant", "125m"),
        freeze_backbone=config.model.freeze_backbone,
        moe_layer_indices=config.model.moe_layer_indices,
        device=config.device.device,
    )


def build_model(config: DictConfig) -> torch.nn.Module:
    """
    Build model with MoE layers injected using registries.

    This function is fully extensible - to add new model types (Llama, Mixtral),
    simply register them in the ModelRegistry and ExpertRegistry.
    """
    # Use helper to get ModelConfig
    model_config = _get_model_config(config)

    # Get model info from registry
    model_info = model_config.get_model_info()

    print(f"Model: {model_config.get_description()}")
    print(f"  HF Name: {model_info['hf_name']}")
    print(f"  Type: {model_info['model_type']}")
    if model_info.get("hidden_dim"):
        print(f"  Hidden Dim: {model_info['hidden_dim']}")

    # Build backbone using registry (extensible to any registered model)
    model_cls = ModelRegistry.get(model_info["model_type"])
    model = model_cls(
        variant=model_info["variant"],
        freeze_backbone=model_config.freeze_backbone,
        moe_layer_indices=model_config.moe_layer_indices,
        device=model_config.device,
    )

    # Resolve expert type: validate against ExpertType enum for early, clear errors
    config_expert_type_str = config.expert.get("type", ExpertType.GPTNEO_LORA.value)
    try:
        expert_type = ExpertType(config_expert_type_str)
    except ValueError:
        available = [e.value for e in ExpertType]
        raise ValueError(
            f"Unknown expert type: '{config_expert_type_str}'. Available: {available}"
        )

    # Build MoE layers
    moe_layers = {}
    for layer_idx in config.model.moe_layer_indices:
        # Convert negative indices to positive
        actual_layer_idx = layer_idx
        if layer_idx < 0:
            actual_layer_idx = model.num_layers + layer_idx

        # Get the original MLP module to load frozen weights from
        original_mlp = model.backbone.transformer.h[actual_layer_idx].mlp

        # Build router
        router = _create_router_from_hydra_config(config, model.hidden_dim)

        # Build LoRA config
        lora_config = LoRAConfig(
            hidden_dim=model.hidden_dim,
            rank=config.expert.lora.rank,
            alpha=config.expert.lora.alpha,
            dropout=config.expert.lora.dropout,
            init_scale=config.expert.lora.init_scale,
        )

        # Build LoRA MoE Layer atomically (loads base weights in one step)
        moe_layer = LoRAMoELayer.from_pretrained_mlp(
            mlp=original_mlp,
            router=router,
            lora_config=lora_config,
            num_experts=config.expert.count,
            expert_type=expert_type,
        )

        moe_layers[actual_layer_idx] = moe_layer

    # Inject MoE layers
    model.inject_moe_layers(moe_layers)

    return model


# Mapping from YAML config names to factory router types
_ROUTER_TYPE_MAPPING = {
    RouterType.METABOLIC.value: RouterType.METABOLIC.value,
    RouterType.STANDARD.value: RouterType.STANDARD.value,
    RouterType.TOPK_ROUTER.value: RouterType.STANDARD.value,
}


def _create_router_from_hydra_config(config: DictConfig, hidden_dim: int):
    """
    Create router from Hydra config using the factory.

    Handles the translation from YAML config names (e.g., "MetabolicRouter")
    to factory router types (e.g., "metabolic").
    """
    from src.routers import create_router

    # Map config router type to factory type
    config_router_type = config.router.type
    if config_router_type not in _ROUTER_TYPE_MAPPING:
        available = list(_ROUTER_TYPE_MAPPING.keys())
        raise ValueError(
            f"Unknown router type: '{config_router_type}'. Available: {available}"
        )

    router_type = _ROUTER_TYPE_MAPPING[config_router_type]

    # Build kwargs based on router type
    router_kwargs = {
        "noise_std": config.router.noise_std,
        "temperature": config.router.temperature,
    }

    # Add metabolic-specific parameters if applicable
    if router_type == RouterType.METABOLIC and hasattr(config.router, "metabolic"):
        router_kwargs.update(
            {
                "lambda_metabolic": config.router.metabolic.lambda_metabolic,
                "mu_silicon": config.router.metabolic.mu_silicon,
                "gamma_recovery": config.router.metabolic.gamma_recovery,
                "beta_cost": config.router.metabolic.beta_cost,
                "warmup_steps": config.router.metabolic.warmup_steps,
                "normalize_inputs": config.router.metabolic.normalize_inputs,
                "normalize_weights": config.router.metabolic.normalize_weights,
            }
        )

    return create_router(
        router_type=router_type,
        hidden_dim=hidden_dim,
        num_experts=config.router.num_experts,
        top_k=config.router.top_k,
        **router_kwargs,
    )


def build_dataloaders(config: DictConfig) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Build training and validation dataloaders using existing datacatalog.
    """
    # Use existing DatasetConfig to leverage datacatalog
    dataset_config = DatasetConfig(
        dataset_key=config.dataset.dataset_key,
        custom_dataset_name=config.dataset.custom_dataset_name,
        custom_dataset_config=config.dataset.custom_dataset_config,
        text_column=config.dataset.text_column,
        max_seq_len=config.dataset.max_seq_len,
        num_samples=config.dataset.num_samples,
        streaming=config.dataset.streaming,
        train_split=config.dataset.train_split,
        eval_split=config.dataset.eval_split,
    )

    # Get dataset info from catalog
    dataset_info = dataset_config.get_dataset_info()

    print(f"Dataset: {dataset_config.get_description()}")
    print(f"  Name: {dataset_info['name']}")
    print(f"  Config: {dataset_info.get('config')}")
    print(f"  Text column: {dataset_info['text_column']}")

    # Load tokenizer using model catalog
    model_config = _get_model_config(config)
    model_info = model_config.get_model_info()
    tokenizer = AutoTokenizer.from_pretrained(model_info["hf_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine cache directory based on execution environment
    if config.execution_env == ExecutionEnv.AWS:
        cache_dir = config.compute.aws.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
    else:
        cache_dir = Path(config.compute.local.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------
    # Load datasets — S3-backed (AWS) or HuggingFace Hub (local/SLURM)
    # -------------------------------------------------------------------
    train_dataset = None
    val_dataset = None

    if config.execution_env == "aws":
        # Try to load from pre-downloaded files in cache_dir
        train_dataset, val_dataset = _load_from_local_files(
            cache_dir=str(cache_dir),
            train_split=config.dataset.train_split,
            eval_split=config.dataset.eval_split,
        )

    if train_dataset is None:
        # Fallback: load directly from HuggingFace Hub
        train_dataset = load_dataset(
            dataset_info["name"],
            dataset_info.get("config"),
            split=config.dataset.train_split,
            streaming=dataset_info["streaming"],
            cache_dir=str(cache_dir),
        )

        if config.dataset.eval_split:
            try:
                val_dataset = load_dataset(
                    dataset_info["name"],
                    dataset_info.get("config"),
                    split=config.dataset.eval_split,
                    streaming=dataset_info["streaming"],
                    cache_dir=str(cache_dir),
                )
            except Exception as e:
                print(f"Warning: Could not load validation split: {e}")

    # Tokenize datasets and create labels
    def tokenize_function(examples):
        # Tokenize the text
        tokenized = tokenizer(
            examples[dataset_info["text_column"]],
            padding="max_length",
            truncation=True,
            max_length=config.dataset.max_seq_len,
            return_tensors=None,
        )

        # Create labels for causal language modeling
        # CRITICAL: Set padding tokens to -100 so they're ignored in loss
        # Without this, the model gets "free" predictions on padding tokens
        labels = []
        for input_ids in tokenized["input_ids"]:
            # Create labels by masking padding tokens with -100
            label_ids = [
                -100 if token_id == tokenizer.pad_token_id else token_id
                for token_id in input_ids
            ]
            labels.append(label_ids)

        tokenized["labels"] = labels
        return tokenized

    # Handle streaming vs non-streaming
    if dataset_info["streaming"]:
        train_dataset = train_dataset.map(tokenize_function, batched=True)
        if val_dataset:
            val_dataset = val_dataset.map(tokenize_function, batched=True)
    else:
        train_dataset = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=train_dataset.column_names,
        )
        train_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )

        if val_dataset:
            val_dataset = val_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=val_dataset.column_names,
            )
            val_dataset.set_format(
                type="torch", columns=["input_ids", "attention_mask", "labels"]
            )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=config.dataset.shuffle and not dataset_info["streaming"],
        num_workers=config.dataset.num_workers,
        drop_last=config.dataset.drop_last,
        prefetch_factor=config.dataset.prefetch_factor
        if config.dataset.num_workers > 0
        else None,
    )

    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            drop_last=False,
        )

    return train_dataloader, val_dataloader


def build_optimizer(
    model: torch.nn.Module, config: DictConfig
) -> torch.optim.Optimizer:
    """
    Build optimizer from config.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if config.training.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.training.lr,
            betas=config.training.betas,
            eps=config.training.eps,
            weight_decay=config.training.weight_decay,
        )
    elif config.training.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=config.training.lr,
            betas=config.training.betas,
            eps=config.training.eps,
        )
    elif config.training.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=config.training.lr,
            momentum=config.training.get("momentum", 0.9),
            weight_decay=config.training.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")

    return optimizer
