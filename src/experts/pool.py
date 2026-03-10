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

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def __getitem__(self, idx: int) -> LoRAMLPExpert:
        return self.experts[idx]

    def load_from_mlp(self, mlp: nn.Module) -> None:
        for expert in self.experts:
            expert.load_from_mlp(mlp)

    def consolidate_shared_weights(self) -> None:
        """
        Collapse N independent GPU copies of frozen MLP weights → 1 shared copy.

        Problem: after ExpertPool.load_from_mlp(), each GPTNeoLoRAMLP independently
        calls SharedLoRALayer(shared_weight=w.detach()), creating N CPU tensors that
        point to the same storage (data_ptr equality). But model.to("cuda") calls
        _apply() on each buffer independently, breaking storage sharing and creating
        N separate GPU allocations (one per expert × 2 linears × 6 MoE layers).

        At 125M: 8 experts × 6 layers × 2 linears × 4.7M params × 2 bytes ≈ 900 MB waste.
        At 1.3B: 8 × 12 × 2 × ~33M params × 2 bytes ≈ 12 GB waste — a showstopper.

        Fix: after model.to(device), make experts 1..N-1 reference expert 0's buffer
        tensors directly. The weights are frozen (never written after load_from_mlp),
        so aliasing is safe. Call this method once, after model.to(device) and before
        DDP/FSDP wrapping.

        Memory saved: (N-1)/N of shared-weight GPU footprint per MoE layer.
        """
        if self.num_experts < 2:
            return

        e0 = self.experts[0]
        if not (hasattr(e0, "c_fc") and e0.c_fc is not None):
            return  # not a GPTNeoLoRAMLP pool; skip silently

        ref = {
            "c_fc_w": e0.c_fc._buffers["shared_weight"],
            "c_fc_b": e0.c_fc._buffers.get("shared_bias"),
            "c_proj_w": e0.c_proj._buffers["shared_weight"],
            "c_proj_b": e0.c_proj._buffers.get("shared_bias"),
        }

        for expert in self.experts[1:]:
            if hasattr(expert, "c_fc") and expert.c_fc is not None:
                expert.c_fc._buffers["shared_weight"] = ref["c_fc_w"]
                expert.c_fc._buffers["shared_bias"] = ref["c_fc_b"]
            if hasattr(expert, "c_proj") and expert.c_proj is not None:
                expert.c_proj._buffers["shared_weight"] = ref["c_proj_w"]
                expert.c_proj._buffers["shared_bias"] = ref["c_proj_b"]

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
