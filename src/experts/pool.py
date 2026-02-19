import os

import torch
from torch import nn

from src.core.registry import ExpertRegistry
from src.experts.lora import LoRAConfig, LoRAMLPExpert
from src.project_types import ExpertType


class ExpertPool(nn.Module):
    """Manages a fixed collection of LoRA MLP experts indexed 0 … N-1."""

    def __init__(
        self,
        config: LoRAConfig,
        num_experts: int,
        expert_type: ExpertType = ExpertType.GPTNEO_LORA,
    ):
        super().__init__()
        self.config = config
        self.expert_type = expert_type
        self.expert_class = ExpertRegistry.get(expert_type.value)
        self.experts = nn.ModuleList(
            [self.expert_class(config) for _ in range(num_experts)]
        )

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def __getitem__(self, idx: int) -> LoRAMLPExpert:
        return self.experts[idx]

    def load_from_mlp(self, mlp: nn.Module) -> None:
        for expert in self.experts:
            expert.load_from_mlp(mlp)

    def freeze_base_weights(self) -> None:
        for expert in self.experts:
            expert.freeze_base_weights()

    def save_expert(self, idx: int, path: str) -> None:
        torch.save(self.experts[idx].state_dict(), path)

    def load_expert(self, idx: int, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        self.experts[idx].load_state_dict(state, strict=False)

    def save_all(self, dir_path: str) -> None:
        """Save every expert to ``<dir_path>/expert_<i>.pt``."""
        os.makedirs(dir_path, exist_ok=True)
        for i, expert in enumerate(self.experts):
            torch.save(expert.state_dict(), os.path.join(dir_path, f"expert_{i}.pt"))

    def load_all(self, dir_path: str) -> None:
        """Load every expert from ``<dir_path>/expert_<i>.pt``."""
        for i in range(len(self.experts)):
            path = os.path.join(dir_path, f"expert_{i}.pt")
            if os.path.exists(path):
                self.load_expert(i, path)
