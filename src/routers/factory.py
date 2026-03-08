from typing import Any

from src.configs.router import (
    MetabolicRouterConfig,
    StandardRouterConfig,
    TopKRouterConfig,
    SwitchRouterConfig,
    DynMoERouterConfig,
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
    "dynmoe": DynMoERouterConfig,
}


def create_router(
    router_type: str, hidden_dim: int, num_experts: int, top_k: int, **kwargs
) -> BaseRouter:
    """
    Create a router instance with the specified configuration.

    This is the single source of truth for router creation, consolidating
    logic that was previously duplicated across multiple modules.

    Args:
        router_type: Type of router ("metabolic" or "standard")
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

    # Create config with provided parameters
    config_cls = ROUTER_CONFIG_CLASSES[router_type]
    config = config_cls(
        hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k, **kwargs
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
    router_cls = RouterRegistry.get(router_type)
    return router_cls(config)


def get_available_router_types() -> list:
    """Return list of available router types."""
    return list(ROUTER_CONFIG_CLASSES.keys())
