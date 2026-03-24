import torch
from torch import nn

from src.core.registry import ExpertRegistry
from src.experts.lora import LoRAConfig, SharedLoRALayer, LoRAMLPExpert


@ExpertRegistry.register("qwen2_lora")
class Qwen2LoRAMLP(LoRAMLPExpert):
    """LoRA expert for Qwen2 SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))."""

    def __init__(self, config: LoRAConfig):
        super().__init__(config)
        self.config = config
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None
        self.act_fn = nn.SiLU()

    def _make_lora(self, weight: torch.Tensor) -> SharedLoRALayer:
        return SharedLoRALayer(
            shared_weight=weight,
            shared_bias=None,
            rank=self.config.rank,
            alpha=self.config.alpha,
            dropout=self.config.dropout,
            init_scale=self.config.init_scale,
            b_init_scale=self.config.b_init_scale,
        )

    def load_from_mlp(self, mlp: nn.Module) -> None:
        for attr in ("gate_proj", "up_proj", "down_proj"):
            if not hasattr(mlp, attr):
                found = [k for k in vars(mlp) if not k.startswith("_")]
                raise ValueError(f"Qwen2 MLP missing '{attr}'. Got: {found}")

        self.gate_proj = self._make_lora(mlp.gate_proj.weight.detach())
        self.up_proj = self._make_lora(mlp.up_proj.weight.detach())
        self.down_proj = self._make_lora(mlp.down_proj.weight.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_proj is None:
            raise RuntimeError("Call load_from_mlp() before using Qwen2LoRAMLP.")
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def get_lora_layer_names(self) -> list[str]:
        return ["gate_proj", "up_proj", "down_proj"]
