from enum import Enum
from pathlib import Path

# Path
CONFIG_DIR = Path(".")
EXPERIMENTS_DIR = Path("experiments")
SCRIPTS_DIR = Path("scripts")


# Enum class
class ExecutionEnv(str, Enum):
    """Execution environment types."""

    LOCAL = "local"
    AWS = "aws"


class RouterType(str, Enum):
    """Router types for MoE layers."""

    METABOLIC = "metabolic"
    STANDARD = "standard"
    TOPK_ROUTER = "topk"
    SWITCH = "switch"
    STRESS_CORRECTED = "stress_corrected"
    DEEPSEEK = "deepseek"
    EXPERT_CHOICE = "expert_choice"


class ExpertType(str, Enum):
    """Expert types for MoE layers."""

    GPTNEO_LORA = "gpt_neo_lora"
    # Future expert types can be added here
    # LLAMA_LORA = "llama_lora"


class ModelType(str, Enum):
    """Supported model architectures."""

    GPTNEO = "gpt_neo"
    # LLAMA = "llama"
