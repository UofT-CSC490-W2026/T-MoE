"""
Expert Pool — manages a fixed-size collection of LoRA experts.

Uses ``nn.ModuleList`` indexed by integer, matching the router's output indices.
"""

from typing import Dict, Any, Optional

import torch
from torch import nn

from src.core.registry import ExpertRegistry
from src.experts.lora import LoRAConfig, LoRAMLPExpert


class ExpertPool(nn.Module):
    """
    A pool of ``num_experts`` LoRA MLP experts, indexed 0 … N-1.

    The pool is created once at init and the experts are accessed by integer
    index — the same integers the Router returns.
    """

    def __init__(self, config: LoRAConfig, num_experts: int, expert_type: str = "gpt_neo_lora"):
        super().__init__()
        self.config = config
        self.expert_class = ExpertRegistry.get(expert_type)
        self.experts = nn.ModuleList(
            [self.expert_class(config) for _ in range(num_experts)]
        )

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def __getitem__(self, idx: int) -> LoRAMLPExpert:
        return self.experts[idx]

    # ── checkpoint helpers ──

    def save_expert(self, idx: int, path: str) -> None:
        """Save a single expert's state dict."""
        torch.save(self.experts[idx].state_dict(), path)

    def load_expert(self, idx: int, path: str) -> None:
        """Load a single expert's state dict."""
        state = torch.load(path, map_location="cpu")
        self.experts[idx].load_state_dict(state, strict=False)

    def save_all(self, dir_path: str) -> None:
        """Save every expert to ``<dir_path>/expert_<i>.pt``."""
        import os
        os.makedirs(dir_path, exist_ok=True)
        for i, expert in enumerate(self.experts):
            torch.save(expert.state_dict(), os.path.join(dir_path, f"expert_{i}.pt"))

    def load_all(self, dir_path: str) -> None:
        """Load every expert from ``<dir_path>/expert_<i>.pt``."""
        import os
        for i in range(len(self.experts)):
            path = os.path.join(dir_path, f"expert_{i}.pt")
            if os.path.exists(path):
                self.load_expert(i, path)
