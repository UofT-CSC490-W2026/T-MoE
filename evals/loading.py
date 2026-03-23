from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:  # pragma: no cover - exercised implicitly in lightweight envs
    DictConfig = Any  # type: ignore[misc,assignment]

    class _OmegaConfShim:
        @staticmethod
        def is_config(value: Any) -> bool:
            return False

        @staticmethod
        def select(config: Any, key: str, default: Any = None) -> Any:
            current = config
            for part in key.split("."):
                if not isinstance(current, dict) or part not in current:
                    return default
                current = current[part]
            return current

        @staticmethod
        def to_container(config: Any, resolve: bool = True) -> Any:
            return config

    OmegaConf = _OmegaConfShim()

from src.configs.model import model_lookup
from src.core import ModelRegistry
from src.experts.lora import LoRAConfig
from src.layers.lora_moe import LoRAMoELayer
from src.project_types import ExpertType
from src.routers.factory import ROUTER_CONFIG_CLASSES, create_router
from src.training import CheckpointManager


def _cfg_select(config: Any, key: str, default: Any = None) -> Any:
    if OmegaConf.is_config(config):
        return OmegaConf.select(config, key, default=default)

    current = config
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _require(config: Any, key: str) -> Any:
    value = _cfg_select(config, key)
    if value is None:
        raise ValueError(f"Missing required config value: '{key}'")
    return value


def _as_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, tuple):
        return [int(v) for v in value]
    if OmegaConf.is_config(value):
        return [int(v) for v in OmegaConf.to_container(value, resolve=True)]
    return [int(value)]


def _router_kwargs(config: Any) -> Tuple[str, int, int, Dict[str, Any]]:
    raw_router = _require(config, "router")
    if OmegaConf.is_config(raw_router):
        router_cfg = OmegaConf.to_container(raw_router, resolve=True)
    else:
        router_cfg = dict(raw_router)

    router_type = router_cfg.pop("type")
    num_experts = int(router_cfg.pop("num_experts"))
    top_k = int(router_cfg.pop("top_k"))

    nested_cfg = router_cfg.pop(router_type, None)
    if isinstance(nested_cfg, dict):
        router_cfg.update(nested_cfg)

    router_cfg.pop("hidden_dim", None)

    config_cls = ROUTER_CONFIG_CLASSES[router_type]
    valid_field_names = {
        field.name
        for field in fields(config_cls)
        if field.name not in {"hidden_dim", "num_experts", "top_k", "router_type"}
    }
    router_cfg = {
        key: value for key, value in router_cfg.items() if key in valid_field_names
    }

    return router_type, num_experts, top_k, router_cfg


def _build_moe_layers(model, config: Any) -> Dict[int, LoRAMoELayer]:
    moe_layer_indices = _as_list(_require(config, "model.moe_layer_indices"))
    expert_type = ExpertType(_cfg_select(config, "expert.type", ExpertType.GPTNEO_LORA))
    expert_count = int(_require(config, "expert.count"))

    model_info = model_lookup(_require(config, "model.model_key"))
    lora_config = LoRAConfig(
        hidden_dim=int(model_info["hidden_dim"]),
        rank=int(_cfg_select(config, "expert.lora.rank", 16)),
        alpha=int(_cfg_select(config, "expert.lora.alpha", 16)),
        dropout=float(_cfg_select(config, "expert.lora.dropout", 0.0)),
        init_scale=float(_cfg_select(config, "expert.lora.init_scale", 0.01)),
    )

    router_type, num_experts, top_k, router_kwargs = _router_kwargs(config)
    if expert_count != num_experts:
        raise ValueError(
            "Config mismatch: expert.count must match router.num_experts "
            f"(got expert.count={expert_count}, router.num_experts={num_experts})"
        )

    blocks: Iterable[Any] = model.backbone.transformer.h
    num_layers = len(blocks)
    moe_layers: Dict[int, LoRAMoELayer] = {}
    for layer_idx in moe_layer_indices:
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(
                f"Invalid model.moe_layer_indices entry {layer_idx}; "
                f"valid range is [0, {num_layers - 1}]"
            )

        router = create_router(
            router_type=router_type,
            hidden_dim=int(model_info["hidden_dim"]),
            num_experts=num_experts,
            top_k=top_k,
            **router_kwargs,
        )
        base_mlp = model.backbone.transformer.h[layer_idx].mlp
        moe_layers[layer_idx] = LoRAMoELayer.from_pretrained_mlp(
            mlp=base_mlp,
            router=router,
            lora_config=lora_config,
            num_experts=expert_count,
            expert_type=expert_type,
        )

    return moe_layers


def build_model_from_config(config: Any, device: str = "cpu"):
    """
    Rebuild a T-MoE model from an experiment config without loading checkpoint weights.

    This reconstructs the pretrained backbone plus injected MoE layers so a saved
    checkpoint can be loaded cleanly for offline evaluation.
    """
    model_key = _require(config, "model.model_key")
    model_info = model_lookup(model_key)
    model_cls = ModelRegistry.get(model_info["model_type"])

    model = model_cls(
        variant=model_info["variant"],
        freeze_backbone=bool(_cfg_select(config, "model.freeze_backbone", True)),
        moe_layer_indices=_as_list(_cfg_select(config, "model.moe_layer_indices", [])),
        device=str(device),
    )

    moe_layers = _build_moe_layers(model, config)
    model.inject_moe_layers(moe_layers)
    model.to(device)
    return model


def load_model_for_eval(
    config: Any,
    checkpoint_path: str | Path,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
):
    """
    Rebuild a T-MoE model from config and load a saved checkpoint for offline eval.

    Returns:
        (model, checkpoint_info)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'")

    model = build_model_from_config(config, device=device)
    checkpoint_manager = CheckpointManager(checkpoint_dir=str(checkpoint_path.parent))
    checkpoint_info = checkpoint_manager.load_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
    )
    if dtype is None:
        model.to(device)
    else:
        model.to(device=device, dtype=dtype)
    model.eval()
    return model, checkpoint_info
