from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


def model_lookup(model_key: str) -> Dict[str, Any]:
    """
    Resolve a model_key (e.g. "gpt-neo-125m") to its full metadata dict.

    Resolution strategy: import all registered model classes from src.models,
    then scan each class's VARIANTS dict for a matching key.

    This means adding a new model requires only:
      1. Create src/models/<name>.py with a VARIANTS dict
      2. Register the class with @ModelRegistry.register("<name>")

    Nothing else needs updating — train.py, prepare_data.py, and this function
    all resolve automatically.

    Returns dict with keys: model_type, variant, hf_name, hidden_dim, num_layers, ...

    Raises:
        ValueError: If model_key is not found in any registered model's VARIANTS.
    """
    # Trigger registration decorators
    import src.models  # noqa: F401 — side-effect: populates ModelRegistry

    from src.core import ModelRegistry

    all_keys = []
    for model_type, model_cls in ModelRegistry.items():
        variants = getattr(model_cls, "VARIANTS", {})
        for variant_key, variant_info in variants.items():
            # Build the canonical model_key from hf_name or by convention
            # Convention: "gpt-neo-125m" → model_type="gpt_neo", variant="125m"
            canonical_key = (
                variant_info.get("model_key")
                or f"{model_type.replace('_', '-')}-{variant_key}"
            )
            all_keys.append(canonical_key)
            if canonical_key == model_key or variant_key == model_key:
                return {
                    "model_type": model_type,
                    "variant": variant_key,
                    **variant_info,
                }

    raise ValueError(
        f"Unknown model_key: '{model_key}'.\n"
        f"Known keys: {sorted(all_keys)}\n"
        f"To add a new model: create src/models/<name>.py with a VARIANTS dict "
        f"and @ModelRegistry.register('<name>')."
    )


@dataclass
class ModelConfig:
    """Model configuration dataclass used by experiment.py."""

    model_key: str = "gpt-neo-125m"
    freeze_backbone: bool = True
    moe_layer_indices: List[int] = field(default_factory=lambda: [-1])
    device: str = "cuda"

    # These are resolved lazily via model_lookup — not hardcoded
    model_type: Optional[str] = None
    variant: Optional[str] = None

    def get_model_info(self) -> Dict[str, Any]:
        """Look up model metadata from the registry."""
        return model_lookup(self.model_key)

    def get_description(self) -> str:
        info = self.get_model_info()
        return f"{info['hf_name']} ({info.get('hidden_dim', '?')}d, {info.get('num_layers', '?')}L)"
