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

    TOPK = "TopKRouter"
    STANDARD = "StandardRouter"
    METABOLIC = "MetabolicRouter"


class ExpertType(str, Enum):
    """Expert types for MoE layers."""

    LORA = "LoRA"


class ModelType(str, Enum):
    """Supported model architectures."""

    GPTNEO = "GPTNeo"
    GPT2 = "GPT2"
    LLAMA = "Llama"
