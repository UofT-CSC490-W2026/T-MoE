import dataclasses
from typing import Any

from src.configs.router import (
    MetabolicRouterConfig,
    StandardRouterConfig,
    TopKRouterConfig,
    SwitchRouterConfig,
    DeepSeekRouterConfig,
    ExpertChoiceRouterConfig,
    StressCorrectedRouterConfig,
)
from src.core import RouterRegistry
from src.routers.base import BaseRouter
from src.project_types import RouterType


# Mapping of router type strings → config classes.
# Keys must match the string literals used in @RouterRegistry.register(...) decorators.
ROUTER_CONFIG_CLASSES = {
    "metabolic": MetabolicRouterConfig,
    "standard": StandardRouterConfig,
    "topk": TopKRouterConfig,
    "switch": SwitchRouterConfig,
    "deepseek": DeepSeekRouterConfig,
    "expert_choice": ExpertChoiceRouterConfig,
    "stress_corrected": StressCorrectedRouterConfig,
}


def create_router(
    router_type: str, hidden_dim: int, num_experts: int, top_k: int, **kwargs
) -> BaseRouter:
    """
    Create a router instance with the specified configuration.

    This is the single source of truth for router creation, consolidating
    logic that was previously duplicated across multiple modules.

    Args:
        router_type: Type of router ("stress_corrected", "metabolic", "standard", "topk", "switch", "deepseek", "expert_choice")
        hidden_dim: Dimension of input embeddings
        num_experts: Number of experts to route to
        top_k: Number of top experts per token
        **kwargs: Additional router-specific configuration

    Returns:
        Configured router instance

    Raises:
        ValueError: If router_type is not recognized

    Example:
        >>> router = create_router(
        ...     router_type="metabolic",
        ...     hidden_dim=768,
        ...     num_experts=8,
        ...     top_k=2,
        ...     lambda_metabolic=0.1,
        ... )
    """
    if router_type not in ROUTER_CONFIG_CLASSES:
        available = sorted(ROUTER_CONFIG_CLASSES.keys())
        raise ValueError(
            f"Unknown router type: '{router_type}'. Available: {available}"
        )

    # Create config with provided parameters.
    # Filter kwargs to only fields declared by this config class so that shared
    # top-level params (e.g. noise_std, temperature) don't crash dataclasses that
    # don't declare them (e.g. MetabolicRouterConfig after v6 cleanup).
    config_cls = ROUTER_CONFIG_CLASSES[router_type]
    valid_fields = {f.name for f in dataclasses.fields(config_cls)}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
    config = config_cls(
        hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k, **filtered_kwargs
    )

    # Get router class from registry and instantiate
    router_cls = RouterRegistry.get(router_type)
    return router_cls(config)


def create_router_from_config(config: Any) -> BaseRouter:
    """
    Create a router from a config object.

    Args:
        config: Router configuration (MetabolicRouterConfig or StandardRouterConfig)

    Returns:
        Configured router instance
    """
    router_type = getattr(config, "router_type", RouterType.METABOLIC)
    # Registry keys are strings (.value); config stores RouterType enum — convert.
    router_key = (
        router_type.value if isinstance(router_type, RouterType) else router_type
    )
    router_cls = RouterRegistry.get(router_key)
    return router_cls(config)
