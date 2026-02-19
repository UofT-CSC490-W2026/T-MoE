import torch
from torch import nn

from src.core.registry import ExpertRegistry
from src.experts.lora import LoRAConfig, LoRALayer, LoRAMLPExpert


def _extract_linear_weight(
    layer: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Extract weight (in Linear convention [out, in]) and optional bias from a
    pretrained layer.  Handles both ``nn.Linear`` and HuggingFace ``Conv1D``
    (whose weight is stored transposed as [in, out]).
    """
    w = layer.weight
    bias = getattr(layer, "bias", None)

    # Conv1D stores weight as [in, out] — transpose to [out, in]
    if w.dim() == 2 and hasattr(layer, "nf"):
        w = w.t()

    return w, bias


def _get_activation() -> nn.Module:
    """Return GPT-Neo's NewGELU if available, else fallback to approx tanh GELU."""
    try:
        from transformers.activations import NewGELUActivation

        return NewGELUActivation()
    except ImportError:
        return nn.GELU(approximate="tanh")


@ExpertRegistry.register("gpt_neo_lora")
class GPTNeoLoRAMLP(LoRAMLPExpert):
    """
    LoRA expert matching the GPT-Neo MLP layout::

        hidden → c_fc (expand) → GELU → c_proj (contract) → dropout → hidden
    """

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
        """
        Load frozen base weights from a GPT-Neo MLP.

        Looks for ``c_fc`` / ``c_proj`` attributes (standard naming)
        and falls back to ``fc_in`` / ``fc_out`` (some variants).
        """
        fc_layer = getattr(mlp, "c_fc", None) or getattr(mlp, "fc_in", None)
        proj_layer = getattr(mlp, "c_proj", None) or getattr(mlp, "fc_out", None)

        if fc_layer is None or proj_layer is None:
            raise ValueError(
                f"GPT-Neo MLP missing c_fc/c_proj (or fc_in/fc_out). "
                f"Got attributes: {[k for k in vars(mlp) if not k.startswith('_')]}"
            )

        w, b = _extract_linear_weight(fc_layer)
        self.c_fc.load_base_weight(w, b)

        w, b = _extract_linear_weight(proj_layer)
        self.c_proj.load_base_weight(w, b)

        self.freeze_base_weights()
