from typing import Dict, Any

# Model catalog - maps short keys to HuggingFace model configurations
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    # GPT-Neo models (EleutherAI)
    "gpt-neo-125m": {
        "name": "EleutherAI/gpt-neo-125M",
        "type": "GPTNeo",
        "hidden_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "description": "GPT-Neo 125M parameters",
    },
    "gpt-neo-350m": {
        "name": "EleutherAI/gpt-neo-350M",
        "type": "GPTNeo",
        "hidden_dim": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "description": "GPT-Neo 350M parameters",
    },
    "gpt-neo-1.3b": {
        "name": "EleutherAI/gpt-neo-1.3B",
        "type": "GPTNeo",
        "hidden_dim": 2048,
        "num_layers": 24,
        "num_heads": 16,
        "description": "GPT-Neo 1.3B parameters",
    },
    "gpt-neo-2.7b": {
        "name": "EleutherAI/gpt-neo-2.7B",
        "type": "GPTNeo",
        "hidden_dim": 2560,
        "num_layers": 32,
        "num_heads": 20,
        "description": "GPT-Neo 2.7B parameters",
    },
    # GPT-2 models (OpenAI)
    "gpt2": {
        "name": "gpt2",
        "type": "GPT2",
        "hidden_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "description": "GPT-2 small (117M parameters)",
    },
    "gpt2-medium": {
        "name": "gpt2-medium",
        "type": "GPT2",
        "hidden_dim": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "description": "GPT-2 medium (345M parameters)",
    },
    "gpt2-large": {
        "name": "gpt2-large",
        "type": "GPT2",
        "hidden_dim": 1280,
        "num_layers": 36,
        "num_heads": 20,
        "description": "GPT-2 large (774M parameters)",
    },
    "gpt2-xl": {
        "name": "gpt2-xl",
        "type": "GPT2",
        "hidden_dim": 1600,
        "num_layers": 48,
        "num_heads": 25,
        "description": "GPT-2 XL (1.5B parameters)",
    },
    # Llama models (Meta) - Placeholder for future implementation
    "llama-7b": {
        "name": "meta-llama/Llama-2-7b-hf",
        "type": "Llama",
        "hidden_dim": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "description": "Llama 2 7B parameters",
    },
    "llama-13b": {
        "name": "meta-llama/Llama-2-13b-hf",
        "type": "Llama",
        "hidden_dim": 5120,
        "num_layers": 40,
        "num_heads": 40,
        "description": "Llama 2 13B parameters",
    },
}


def get_available_models() -> list[str]:
    """Return list of available model keys."""
    return list(MODEL_CATALOG.keys())


def get_model_info(model_key: str) -> Dict[str, Any]:
    """Get model configuration by key."""
    if model_key not in MODEL_CATALOG:
        available = ", ".join(get_available_models())
        raise KeyError(
            f"Model '{model_key}' not found in catalog. Available models: {available}"
        )
    return MODEL_CATALOG[model_key]


def validate_model_key(model_key: str) -> bool:
    """Check if a model key exists in the catalog."""
    return model_key in MODEL_CATALOG


def get_models_by_type(model_type: str) -> list[str]:
    """Get all model keys of a specific type."""
    return [key for key, info in MODEL_CATALOG.items() if info["type"] == model_type]
