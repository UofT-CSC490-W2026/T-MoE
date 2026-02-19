from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from src.experts.base import BaseExpert


# ── Config ──────────────────────────────────────────────────────────────────


@dataclass
class LoRAConfig:
    """Configuration shared by all LoRA layers and experts."""

    hidden_dim: int
    intermediate_dim: Optional[int] = None  # defaults to 4 × hidden_dim
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    init_scale: float = 0.01

    def __post_init__(self):
        if self.intermediate_dim is None:
            self.intermediate_dim = 4 * self.hidden_dim

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank


# ── Single LoRA Layer ───────────────────────────────────────────────────────


class LoRALayer(nn.Module):
    """
    Low-rank adapter wrapped around a (frozen) linear projection.

    Math
    ----
    output = base(x) + (B @ A @ x) · scaling

    * ``base`` comes from `load_base_weight` and is frozen.
    * ``A`` is Kaiming-initialised, ``B`` is zero-initialised →
      the adapter is the identity at init.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: int,
        dropout: float = 0.0,
        init_scale: float = 0.01,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # Trainable adapters
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialisation: A ← Kaiming, B ← 0  (net effect = 0 at start)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=init_scale)
        nn.init.zeros_(self.lora_B.weight)

        # Frozen base weight (loaded later via `load_base_weight`)
        self.base_weight: Optional[nn.Parameter] = None
        self.base_bias: Optional[nn.Parameter] = None

    # ── loading ──

    def load_base_weight(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Store a frozen copy of the pretrained weight (and optional bias).

        The weight must already be in **Linear convention**: ``[out, in]``.
        Callers are responsible for transposing Conv1D weights beforehand.
        """
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Weight shape {weight.shape} does not match "
                f"expected ({self.out_features}, {self.in_features})"
            )
        self.base_weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
        if bias is not None:
            self.base_bias = nn.Parameter(bias.detach().clone(), requires_grad=False)

    # ── forward ──

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``base(x) + lora_delta(x)`` — works with any leading dims."""
        # Base path (frozen)
        if self.base_weight is not None:
            base_out = nn.functional.linear(x, self.base_weight, self.base_bias)
        else:
            base_out = torch.zeros(
                *x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype
            )

        # LoRA path (trainable)
        lora_out = self.lora_B(self.lora_dropout(self.lora_A(x)))
        return base_out + lora_out * self.scaling


# ── Shared LoRA Layer ───────────────────────────────────────────────────────


class SharedLoRALayer(nn.Module):
    """
    LoRA layer referencing shared frozen weights — no per-expert cloning.
    Use for large models (Llama 3B+) to avoid OOM.

    Math: output = base(x) + (B @ A @ x) * scaling
    shared_weight is a buffer: moves with .to(device), excluded from optimizer.
    """

    def __init__(
        self,
        shared_weight: torch.Tensor,  # [out, in] — Linear convention
        shared_bias: Optional[torch.Tensor],
        rank: int,
        alpha: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        out_features, in_features = shared_weight.shape
        self.scaling = alpha / rank

        self.register_buffer("shared_weight", shared_weight.detach())
        self.register_buffer(
            "shared_bias",
            shared_bias.detach() if shared_bias is not None else None,
        )

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=0.01)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = nn.functional.linear(x, self.shared_weight, self.shared_bias)
        lora_out = self.lora_B(self.lora_dropout(self.lora_A(x)))
        return base_out + lora_out * self.scaling


# ── Abstract MLP Expert ─────────────────────────────────────────────────────


class LoRAMLPExpert(BaseExpert):
    """
    Abstract base for architecture-specific LoRA MLP experts.

    Subclasses implement ``forward`` and ``load_from_mlp`` for a specific
    transformer family (GPT-2, LLaMA, Mistral, …).
    """

    def __init__(self, config: LoRAConfig):
        super().__init__(config)

    @abstractmethod
    def load_from_mlp(self, mlp: nn.Module) -> None:
        """Copy frozen base weights from a pretrained MLP block."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the expert MLP. Returns tensor of same shape as input."""

    def freeze_base_weights(self) -> None:
        """Ensure every ``base_weight`` / ``base_bias`` has ``requires_grad=False``."""
        for name, param in self.named_parameters():
            if "base_weight" in name or "base_bias" in name:
                param.requires_grad = False
