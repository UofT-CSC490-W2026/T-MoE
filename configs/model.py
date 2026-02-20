from dataclasses import dataclass
from typing import Dict, Any, Optional, List
from src.core import ModelRegistry

from configs import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for model backbone.

    Uses registry pattern for model instantiation.

    Example:
        # Using GPT-Neo 125M
        config = ModelConfig(model_type="gpt_neo", variant="125m")

        # Using GPT-Neo 2.7B
        config = ModelConfig(model_type="gpt_neo", variant="2.7b")
    """

    # Model selection (registry-based)
    model_type: str = "gpt_neo"  # Registry key (e.g., "gpt_neo", "gpt2", "llama")
    variant: str = "125m"  # Model variant (e.g., "125m", "350m", "1.3b")

    # Model configuration
    freeze_backbone: bool = True
    moe_layer_indices: List[int] = None

    # Device
    device: str = "cuda"

    # Model dimensions (autopopulated from model class)
    hidden_dim: Optional[int] = None
    num_layers: Optional[int] = None
    vocab_size: Optional[int] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.moe_layer_indices is None:
            self.moe_layer_indices = [-1]

    def get_model_info(self) -> Dict[str, Any]:
        """Get resolved model configuration from registry.

        Returns:
            Dictionary with model configuration including:
            - model_type: Registry key
            - variant: Model variant
            - hidden_dim: Hidden dimension size
            - num_layers: Number of transformer layers
        """
        # Get model class from registry
        model_cls = ModelRegistry.get(self.model_type)

        # Get variant info
        variant_info = model_cls.get_variant_info(self.variant)

        return {
            "model_type": self.model_type,
            "variant": self.variant,
            "hf_name": variant_info["hf_name"],
            "hidden_dim": variant_info["hidden_dim"],
            "num_layers": variant_info["num_layers"],
            "description": variant_info.get("description", ""),
        }

    def get_description(self) -> str:
        """Get model description."""
        info = self.get_model_info()
        return info.get("description", f"{self.model_type}/{self.variant}")
