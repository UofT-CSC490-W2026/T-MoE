"""
Configuration for T-MoE training infrastructure.
"""

import os

# Environment: "dev" (personal workspace) or "prod" (t-moe workspace)
ENV = os.getenv("MODAL_ENV", "dev")

# Infrastructure configuration per environment
INFRASTRUCTURE_CONFIG = {
    "prod": {
        "gpu": "A10G:1",
        "cpu": 4,
        "timeout": 60 * 60 * 4,  # 4 hours for prod
        "profile": "t-moe",  # T-MoE workspace
    }
}

# App naming
APP_NAME = "tmoe-gpt-neo-training"

# Model and dataset constants
MODEL_NAME = "EleutherAI/gpt-neo-125m"
DATASET_NAME = "KrisMinchev/wikitext-2-raw-v1"

def get_config():
    """Get the current environment configuration."""
    return INFRASTRUCTURE_CONFIG[ENV]

def get_app_name():
    """Get the app name with environment suffix."""
    return f"{APP_NAME}-{ENV}"
