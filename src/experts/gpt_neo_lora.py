import torch
from torch import nn

from src.core.registry import ExpertRegistry
from src.experts.lora import LoRAConfig, LoRALayer, LoRAMLPExpert


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
        self.c_fc = LoRALayer(
            config.hidden_dim,
            config.intermediate_dim,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            init_scale=config.init_scale,
        )
        self.c_proj = LoRALayer(
            config.intermediate_dim,
            config.hidden_dim,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            init_scale=config.init_scale,
        )
        self.act = _get_activation()
        # NOTE: GPT-Neo's original MLP does not include output dropout.
        # This is an intentional regularization addition for LoRA fine-tuning.
        # With dropout=0.0 (default) this is a no-op.
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))

    def load_from_mlp(self, mlp: nn.Module) -> None:
        fc_layer = getattr(mlp, "c_fc", None) or getattr(mlp, "fc_in", None)
        proj_layer = getattr(mlp, "c_proj", None) or getattr(mlp, "fc_out", None)

        if fc_layer is None or proj_layer is None:
            raise ValueError(
                f"GPT-Neo MLP missing c_fc/c_proj. "
                f"Got: {[k for k in vars(mlp) if not k.startswith('_')]}"
            )

        fc_w, fc_b = _extract_linear_weight(fc_layer)
        actual_out, actual_in = fc_w.shape
        if (actual_in, actual_out) != (self.c_fc.in_features, self.c_fc.out_features):
            raise ValueError(
                f"Dim mismatch: LoRAConfig expects ({self.c_fc.in_features}, "
                f"{self.c_fc.out_features}) but MLP has ({actual_in}, {actual_out})."
            )

        proj_w, proj_b = _extract_linear_weight(proj_layer)
        self.c_fc.load_base_weight(fc_w, fc_b)
        self.c_proj.load_base_weight(proj_w, proj_b)
        self.freeze_base_weights()
