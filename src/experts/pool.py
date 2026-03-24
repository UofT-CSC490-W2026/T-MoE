import os

import torch
from torch import nn

from src.core.registry import ExpertRegistry
from src.experts.lora import LoRAConfig, LoRAMLPExpert
from src.project_types import ExpertType


class ExpertPool(nn.Module):
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
        # Populated by make_base_trainable() when config.trainable_base=True.
        # Registered as nn.Parameters so optimizer sees them; biases stay as frozen buffers.
        self.shared_fc_weight: nn.Parameter | None = None
        self.shared_proj_weight: nn.Parameter | None = None

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def __getitem__(self, idx: int) -> LoRAMLPExpert:
        return self.experts[idx]

    def load_from_mlp(self, mlp: nn.Module) -> None:
        for expert in self.experts:
            expert.load_from_mlp(mlp)

    def consolidate_shared_weights(self) -> None:
        if self.num_experts < 2:
            return

        e0 = self.experts[0]

        if hasattr(e0, "get_lora_layer_names"):
            layer_names = e0.get_lora_layer_names()
        elif hasattr(e0, "c_fc") and e0.c_fc is not None:
            layer_names = ["c_fc", "c_proj"]
        else:
            return

        ref = {}
        for name in layer_names:
            layer = getattr(e0, name, None)
            if layer is not None:
                ref[name] = {k: v for k, v in layer._buffers.items()}

        for expert in self.experts[1:]:
            for name, buffers in ref.items():
                layer = getattr(expert, name, None)
                if layer is not None:
                    for buf_name, buf_val in buffers.items():
                        layer._buffers[buf_name] = buf_val

    def make_base_trainable(self) -> None:
        """
        Promote shared base weights to trainable nn.Parameters at pool level.

        Must be called AFTER consolidate_shared_weights() (so we're on-device
        and buffers are shared) and BEFORE DDP/FSDP wrapping.

        After this call, ExpertPool.shared_fc_weight and shared_proj_weight are
        trainable parameters. LoRAMoELayer.forward() passes them explicitly to
        each expert, so F.linear() sees the actual parameter (not .data) and
        gradients flow correctly.
        """
        if self.num_experts == 0:
            return
        e0 = self.experts[0]
        if not (hasattr(e0, "c_fc") and e0.c_fc is not None):
            return
        self.shared_fc_weight = nn.Parameter(
            e0.c_fc._buffers["shared_weight"].clone().float()
        )
        self.shared_proj_weight = nn.Parameter(
            e0.c_proj._buffers["shared_weight"].clone().float()
        )

    def freeze_base_weights(self) -> None:
        for expert in self.experts:
            expert.freeze_base_weights()

    def save_expert(self, idx: int, path: str) -> None:
        torch.save(self.experts[idx].state_dict(), path)

    def load_expert(self, idx: int, path: str) -> None:
        state = torch.load(path, map_location="cpu")
        self.experts[idx].load_state_dict(state, strict=False)

    def save_all(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        for i, expert in enumerate(self.experts):
            torch.save(expert.state_dict(), os.path.join(dir_path, f"expert_{i}.pt"))

    def load_all(self, dir_path: str) -> None:
        for i in range(len(self.experts)):
            path = os.path.join(dir_path, f"expert_{i}.pt")
            if os.path.exists(path):
                self.load_expert(i, path)
