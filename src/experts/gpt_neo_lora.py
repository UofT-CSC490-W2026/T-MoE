from typing import Optional

import torch
from torch import nn

from src.core.registry import ExpertRegistry
from src.experts.lora import LoRAConfig, SharedLoRALayer, LoRAMLPExpert


def _extract_linear_weight(
    layer: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return (weight [out, in], bias). Transposes Conv1D weights from [in, out]."""
    w = layer.weight
    bias = getattr(layer, "bias", None)
    if w.dim() == 2 and hasattr(layer, "nf"):  # HuggingFace Conv1D
        w = w.t()
    return w, bias


def _get_activation() -> nn.Module:
    try:
        from transformers.activations import NewGELUActivation

        return NewGELUActivation()
    except ImportError:
        return nn.GELU(approximate="tanh")


@ExpertRegistry.register("gpt_neo_lora")
class GPTNeoLoRAMLP(LoRAMLPExpert):
    """LoRA expert for GPT-Neo MLP: hidden → c_fc → GELU → c_proj → dropout → hidden."""

    def __init__(self, config: LoRAConfig):
        super().__init__(config)
        self.config = config
        self.c_fc = None  # Instantiated in load_from_mlp
        self.c_proj = None  # Instantiated in load_from_mlp
        self.act = _get_activation()
        # NOTE: GPT-Neo's original MLP does not include output dropout.
        # This is an intentional regularization addition for LoRA fine-tuning.
        # With dropout=0.0 (default) this is a no-op.
        self.dropout = nn.Dropout(config.dropout)

    def _make_lora(self, weight: torch.Tensor, bias) -> SharedLoRALayer:
        return SharedLoRALayer(
            shared_weight=weight,
            shared_bias=bias,
            rank=self.config.rank,
            alpha=self.config.alpha,
            dropout=self.config.dropout,
            init_scale=self.config.init_scale,
            b_init_scale=self.config.b_init_scale,
        )

    def forward(
        self,
        x: torch.Tensor,
        fc_weight: Optional[torch.Tensor] = None,
        proj_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.c_fc is None:
            raise RuntimeError("Call load_from_mlp() before using GPTNeoLoRAMLP.")
        return self.dropout(
            self.c_proj(
                self.act(self.c_fc(x, base_weight=fc_weight)), base_weight=proj_weight
            )
        )

    def load_from_mlp(self, mlp: nn.Module) -> None:
        fc_layer = getattr(mlp, "c_fc", None) or getattr(mlp, "fc_in", None)
        proj_layer = getattr(mlp, "c_proj", None) or getattr(mlp, "fc_out", None)

        if fc_layer is None or proj_layer is None:
            raise ValueError(
                f"GPT-Neo MLP missing c_fc/c_proj. "
                f"Got: {[k for k in vars(mlp) if not k.startswith('_')]}"
            )

        fc_w, fc_b = _extract_linear_weight(fc_layer)
        proj_w, proj_b = _extract_linear_weight(proj_layer)
        self.c_fc = self._make_lora(fc_w, fc_b)
        self.c_proj = self._make_lora(proj_w, proj_b)
