from dataclasses import dataclass, field
from .base import BaseConfig, DeviceConfig, LoggingConfig, TrainingConfig
from .router import RouterConfig
from .dataset import DatasetConfig
from .model import ModelConfig


@dataclass
class ExpertConfig(BaseConfig):
    """Configuration for experts."""

    type: str = "LoRA"
    count: int = 4
    # LoRA specific
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    init_scale: float = 0.01


@dataclass
class ComputeConfig(BaseConfig):
    """Configuration for execution environment (local vs AWS).

    For device-specific settings (device, dtype, compile), use DeviceConfig separately.
    """

    execution_env: str = "local"  # local, aws


@dataclass
class TMoEExperimentConfig(BaseConfig):
    """Main experiment configuration composing all sub-configs."""

    experiment_name: str = "gptneo_lora_moe"
    seed: int = 42
    execution_env: str = "local"

    model: ModelConfig = field(default_factory=ModelConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
