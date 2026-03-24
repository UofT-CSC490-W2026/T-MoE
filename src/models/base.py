from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple

import torch
from torch import nn


class BaseModelBackbone(nn.Module, ABC):
    """
    Abstract base class for pre-trained model backbones.

    Model backbones process input through frozen/partially-frozen layers,
    with MoE layers injected at specified positions to produce expert-augmented outputs.
    """

    def __init__(
        self,
        model_name: str,
        hidden_dim: int,
        freeze_backbone: bool = True,
        moe_layer_indices: Optional[list[int]] = None,
    ):
        """
        Initialize model backbone.

        Args:
            model_name: HuggingFace model identifier (e.g., "EleutherAI/gpt-neo-125M")
            hidden_dim: Hidden dimension of the model
            freeze_backbone: Whether to freeze backbone parameters
            moe_layer_indices: Layer indices where MoE layers are injected (None = last layer)
        """
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.freeze_backbone = freeze_backbone
        self.moe_layer_indices = moe_layer_indices or []

        # To be set by subclasses
        self.backbone = None
        self.moe_layers: dict = {}

    @abstractmethod
    def load_pretrained(self) -> None:
        """Load pre-trained weights from HuggingFace."""
        pass

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Forward pass with MoE augmentation.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_metrics: Whether to return routing metrics
            **kwargs: Additional model-specific arguments

        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            metrics: Optional routing metrics from MoE layers
        """
        pass

    def get_mlp_at(self, idx: int) -> nn.Module:
        raise NotImplementedError

    def inject_moe_layers(self, moe_layers: Dict[int, nn.Module]) -> None:
        """
        Inject MoE layers at specified positions.

        Args:
            moe_layers: Dict mapping layer index to MoE module
        """
        self.moe_layers = {str(k): v for k, v in moe_layers.items()}

    def freeze_parameters(self) -> None:
        """Freeze all backbone parameters (except MoE layers)."""
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def unfreeze_parameters(self) -> None:
        """Unfreeze all parameters."""
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = True

    def get_trainable_params(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
