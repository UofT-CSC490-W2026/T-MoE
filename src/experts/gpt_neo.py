import torch
from torch import nn

from src.core import ExpertRegistry
from src.experts.lora_mlp import LoRAMLPExpert, LoRAConfig
from src.experts.lora_layer import LoRALayer
from src.types import ExpertType


@ExpertRegistry.register(ExpertType.GPTNEO_LORA.value)
class GPTNeoLoRAExpert(LoRAMLPExpert):
    """
    GPT-Neo MLP expert with LoRA.

    Architecture: fc1 (768→3072) -> GELU -> fc2 (3072→768)
    Both fc1 and fc2 use LoRA (base frozen + adapters trainable).

    Registered as: "gpt_neo_lora"
    """

    def __init__(self, config: LoRAConfig):
        super().__init__(config)

        # Two LoRA layers matching GPT-Neo MLP structure
        self.fc1 = LoRALayer(
            in_features=config.hidden_dim,
            out_features=config.intermediate_dim,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            init_scale=config.init_scale,
        )

        self.fc2 = LoRALayer(
            in_features=config.intermediate_dim,
            out_features=config.hidden_dim,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            init_scale=config.init_scale,
        )

        # GPT-Neo uses NewGELU activation
        self.activation = self._get_activation()

    def _get_activation(self) -> nn.Module:
        """Get NewGELU activation (GPT-Neo specific)."""
        try:
            from transformers.activations import NewGELUActivation

            return NewGELUActivation()
        except ImportError:
            # Fallback to standard GELU
            return nn.GELU()

    def load_from_mlp(self, mlp: nn.Module, share_weights: bool = True) -> None:
        """
        Load frozen weights from GPT-Neo MLP.

        Expected structure:
        - mlp.c_fc: first linear (768 → 3072)
        - mlp.c_proj: second linear (3072 → 768)

        Args:
            mlp: GPT-Neo MLP module
            share_weights: Whether to share base weights to save memory.

        Raises:
            ValueError: If MLP structure doesn't match GPT-Neo
        """
        if not hasattr(mlp, "c_fc") or not hasattr(mlp, "c_proj"):
            raise ValueError(
                "MLP must have 'c_fc' and 'c_proj' attributes (GPT-Neo structure)"
            )

        self.fc1.load_from_linear(mlp.c_fc, share_weights=share_weights)
        self.fc2.load_from_linear(mlp.c_proj, share_weights=share_weights)
        self.freeze_base_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward: fc1 -> activation -> fc2

        Args:
            x: Input tensor [..., hidden_dim]

        Returns:
            Output tensor [..., hidden_dim]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def get_param_count(self) -> int:
        """
        Get trainable parameter count (LoRA adapters only).

        Returns:
            Number of trainable parameters
        """
        # fc1: (hidden * rank) + (rank * intermediate)
        # fc2: (intermediate * rank) + (rank * hidden)
        fc1_params = (
            self.config.hidden_dim + self.config.intermediate_dim
        ) * self.config.rank
        fc2_params = (
            self.config.intermediate_dim + self.config.hidden_dim
        ) * self.config.rank
        return fc1_params + fc2_params

    def get_flops(self, x: torch.Tensor) -> int:
        """
        Estimate FLOPs for forward pass.

        Args:
            x: Input tensor

        Returns:
            Estimated FLOP count
        """
        num_tokens = x.shape[0] if x.dim() == 2 else x.shape[0] * x.shape[1]

        # fc1 LoRA: 2 * (hidden * rank + rank * intermediate)
        # fc2 LoRA: 2 * (intermediate * rank + rank * hidden)
        fc1_flops = (
            2
            * num_tokens
            * (
                self.config.hidden_dim * self.config.rank
                + self.config.rank * self.config.intermediate_dim
            )
        )
        fc2_flops = (
            2
            * num_tokens
            * (
                self.config.intermediate_dim * self.config.rank
                + self.config.rank * self.config.hidden_dim
            )
        )

        return fc1_flops + fc2_flops
