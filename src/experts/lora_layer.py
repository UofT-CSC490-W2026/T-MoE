from typing import Optional

import torch
from torch import nn


class LoRALayer(nn.Module):
    """
    Single LoRA-augmented linear layer.

    Implements: output = base_weight @ x + (lora_B @ lora_A @ x) * scaling
    where base_weight is frozen and lora_A, lora_B are trainable.

    This is the primitive building block for all LoRA-based experts.
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
        """
        Initialize LoRA layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: LoRA rank (bottleneck dimension)
            alpha: LoRA scaling factor numerator
            dropout: Dropout probability for LoRA path
            init_scale: Scale for Kaiming initialization
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank

        # LoRA adapters
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Initialize: A with Kaiming, B with zeros (identity at init)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=init_scale)
        nn.init.zeros_(self.lora_B.weight)

        # Base layer (frozen, loaded separately)
        self.base_weight: Optional[nn.Parameter] = None
        self.base_bias: Optional[nn.Parameter] = None

    def load_from_linear(self, linear: nn.Linear) -> None:
        """
        Load frozen base weights from a pretrained linear layer.

        Args:
            linear: Pretrained linear layer to wrap

        Raises:
            ValueError: If dimensions don't match
        """
        if linear.in_features != self.in_features:
            raise ValueError(
                f"Input dim mismatch: {linear.in_features} != {self.in_features}"
            )
        if linear.out_features != self.out_features:
            raise ValueError(
                f"Output dim mismatch: {linear.out_features} != {self.out_features}"
            )

        # Clone and freeze
        self.base_weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        if linear.bias is not None:
            self.base_bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: base(x) + lora_delta(x).

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        # Base transformation
        if self.base_weight is not None:
            base_out = nn.functional.linear(x, self.base_weight, self.base_bias)
        else:
            base_out = torch.zeros(
                *x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype
            )

        # LoRA delta
        lora_out = self.lora_A(x)
        if self.dropout is not None:
            lora_out = self.dropout(lora_out)
        lora_out = self.lora_B(lora_out)

        return base_out + lora_out * self.scaling
