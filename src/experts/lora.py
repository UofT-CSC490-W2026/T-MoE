from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from src.experts.base import BaseExpert


@dataclass
class LoRAConfig:
    """Configuration for LoRA layers and experts."""

    hidden_dim: int
    intermediate_dim: Optional[int] = None  # defaults to 4 × hidden_dim
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    init_scale: float = 0.01
    b_init_scale: float = (
        0.0  # Non-zero breaks expert symmetry for MoE routing (see LOG entry)
    )
    trainable_base: bool = False  # if True, shared base weights are trainable nn.Parameters (held at ExpertPool level)
    shared_base_rank: int = (
        0  # 0 = disabled; >0 enables shared base LoRA on c_proj for all tokens
    )
    shared_base_alpha: float = 0.0  # 0 = auto (set to shared_base_rank → scaling=1.0); explicit value overrides

    def __post_init__(self):
        if self.intermediate_dim is None:
            self.intermediate_dim = 4 * self.hidden_dim
        # Auto-set shared_base_alpha = shared_base_rank (scaling=1.0) when not explicitly provided.
        # Matches LoRA Without Regret finding: alpha=rank is optimal.
        if self.shared_base_alpha == 0.0 and self.shared_base_rank > 0:
            self.shared_base_alpha = float(self.shared_base_rank)

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


class LoRALayer(nn.Module):
    """
    Frozen linear projection + low-rank trainable adapter.

    output = base(x) + (B @ A @ x) · scaling
    A is Kaiming-initialised, B is zero-initialised — net delta is 0 at init.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: int,
        dropout: float = 0.0,
        init_scale: float = 0.01,
        b_init_scale: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=init_scale)
        if b_init_scale > 0:
            nn.init.normal_(self.lora_B.weight, std=b_init_scale)
        else:
            nn.init.zeros_(self.lora_B.weight)

        # Non-persistent buffers: move with .to(device), excluded from state_dict + optimizer.
        self.register_buffer("base_weight", None, persistent=False)
        self.register_buffer("base_bias", None, persistent=False)

    def load_base_weight(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        """Store a frozen copy of the pretrained weight. Expects Linear convention [out, in]."""
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Weight shape {weight.shape} does not match "
                f"expected ({self.out_features}, {self.in_features})"
            )
        self.register_buffer("base_weight", weight.detach().clone(), persistent=False)
        if bias is not None:
            self.register_buffer("base_bias", bias.detach().clone(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.base_weight is not None:
            w = (
                self.base_weight.to(x.dtype)
                if self.base_weight.dtype != x.dtype
                else self.base_weight
            )
            b = (
                self.base_bias.to(x.dtype)
                if (self.base_bias is not None and self.base_bias.dtype != x.dtype)
                else self.base_bias
            )
            base_out = nn.functional.linear(x, w, b)
        else:
            base_out = torch.zeros(
                *x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype
            )
        lora_out = self.lora_B(
            self.lora_dropout(self.lora_A(x.to(self.lora_A.weight.dtype)))
        ).to(x.dtype)
        return base_out + lora_out * self.scaling


class SharedLoRALayer(nn.Module):
    """
    LoRA layer with shared frozen weights across experts — avoids per-expert cloning.
    Prefer over LoRALayer for large models (Llama 3B+) to avoid OOM.

    `shared_weight` is a non-persistent buffer: moves with `.to(device)`, excluded from
    optimizer AND `state_dict`. Reconstructed via `load_from_mlp` on model load.
    """

    def __init__(
        self,
        shared_weight: torch.Tensor,  # [out, in] — Linear convention
        shared_bias: Optional[torch.Tensor],
        rank: int,
        alpha: int,
        dropout: float = 0.0,
        init_scale: float = 0.01,
        b_init_scale: float = 0.0,
    ):
        super().__init__()
        out_features, in_features = shared_weight.shape
        self.scaling = alpha / rank

        self.register_buffer("shared_weight", shared_weight.detach(), persistent=False)
        self.register_buffer(
            "shared_bias",
            shared_bias.detach() if shared_bias is not None else None,
            persistent=False,
        )

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=init_scale)
        if b_init_scale > 0:
            nn.init.normal_(self.lora_B.weight, std=b_init_scale)
        else:
            nn.init.zeros_(self.lora_B.weight)

    def forward(
        self,
        x: torch.Tensor,
        base_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        w = base_weight if base_weight is not None else self.shared_weight
        if w.dtype != x.dtype:
            w = w.to(x.dtype)
        b = self.shared_bias
        if b is not None and b.dtype != x.dtype:
            b = b.to(x.dtype)
        base_out = nn.functional.linear(x, w, b)
        lora_out = self.lora_B(
            self.lora_dropout(self.lora_A(x.to(self.lora_A.weight.dtype)))
        ).to(x.dtype)
        return base_out + lora_out * self.scaling


class SharedBaseLoRA(nn.Module):
    """
    Trainable LoRA delta applied to ALL tokens on the c_proj stage.

    Does NOT include the frozen base — only the low-rank correction:
        delta(h) = (h @ A.T) @ B.T * scaling
    where h = act(W_fc · x) is the frozen hidden state passed in from outside.

    B is zero-initialized → delta is zero at init (standard LoRA guarantee).
    """

    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=0.01)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        param_dtype = self.lora_A.weight.dtype
        out = self.lora_B(self.lora_A(h.to(param_dtype))).to(h.dtype)
        return out * self.scaling


class LoRAMLPExpert(BaseExpert):
    """Abstract base for architecture-specific LoRA MLP experts."""

    def __init__(self, config: LoRAConfig):
        super().__init__(config)

    @abstractmethod
    def load_from_mlp(self, mlp: nn.Module) -> None:
        """Copy frozen base weights from a pretrained MLP block."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the expert MLP."""

    def freeze_base_weights(self) -> None:
        for name, param in self.named_parameters():
            if "base_weight" in name or "base_bias" in name:
                param.requires_grad = False
