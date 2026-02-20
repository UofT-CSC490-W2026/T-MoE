from typing import Optional, Dict, Any, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig

from src.core import ModelRegistry
from src.models.base import BaseModelBackbone
from src.project_types import ModelType


@ModelRegistry.register(ModelType.GPTNEO.value)
class GPTNeoBackbone(BaseModelBackbone):
    """
    GPT-Neo model backbone with MoE layer injection.

    Loads pre-trained GPT-Neo models and injects MoE layers at specified positions.
    Supports multiple variants (125M, 350M, 1.3B, 2.7B).

    Registered as: "gpt_neo"
    """

    # Model variant configurations
    VARIANTS = {
        "125m": {
            "hf_name": "EleutherAI/gpt-neo-125M",
            "hidden_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "description": "GPT-Neo 125M parameters",
        },
        "350m": {
            "hf_name": "EleutherAI/gpt-neo-350M",
            "hidden_dim": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "description": "GPT-Neo 350M parameters",
        },
        "1.3b": {
            "hf_name": "EleutherAI/gpt-neo-1.3B",
            "hidden_dim": 2048,
            "num_layers": 24,
            "num_heads": 16,
            "description": "GPT-Neo 1.3B parameters",
        },
        "2.7b": {
            "hf_name": "EleutherAI/gpt-neo-2.7B",
            "hidden_dim": 2560,
            "num_layers": 32,
            "num_heads": 20,
            "description": "GPT-Neo 2.7B parameters",
        },
    }

    def __init__(
        self,
        variant: str = "125m",
        freeze_backbone: bool = True,
        moe_layer_indices: Optional[list[int]] = None,
        device: str = "cuda",
    ):
        """
        Initialize GPT-Neo backbone.

        Args:
            variant: Model variant (125m, 350m, 1.3b, 2.7b)
            freeze_backbone: Whether to freeze backbone parameters
            moe_layer_indices: Layer indices for MoE injection
            device: Device to load model on

        Raises:
            ValueError: If variant not found
        """
        if variant not in self.VARIANTS:
            available = ", ".join(self.VARIANTS.keys())
            raise ValueError(f"Invalid variant '{variant}'. Available: {available}")

        # Get variant configuration
        variant_config = self.VARIANTS[variant]
        model_name = variant_config["hf_name"]
        hidden_dim = variant_config["hidden_dim"]

        super().__init__(
            model_name=model_name,
            hidden_dim=hidden_dim,
            freeze_backbone=freeze_backbone,
            moe_layer_indices=moe_layer_indices,
        )

        self.variant = variant
        self.device = device
        self.vocab_size = None  # Set after loading
        self.num_layers = variant_config["num_layers"]

        # Load model
        self.load_pretrained()

        if freeze_backbone:
            self.freeze_parameters()

    @classmethod
    def get_variant_info(cls, variant: str) -> Dict[str, Any]:
        """
        Get configuration for a specific variant.

        Args:
            variant: Variant name

        Returns:
            Variant configuration dictionary
        """
        if variant not in cls.VARIANTS:
            available = ", ".join(cls.VARIANTS.keys())
            raise ValueError(f"Invalid variant '{variant}'. Available: {available}")
        return cls.VARIANTS[variant].copy()

    @classmethod
    def list_variants(cls) -> list[str]:
        """List all available variants."""
        return list(cls.VARIANTS.keys())

    def load_pretrained(self) -> None:
        """Load pre-trained GPT-Neo from HuggingFace."""
        self.backbone = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        ).to(self.device)

        # Load vocab size from config
        config = AutoConfig.from_pretrained(self.model_name)
        self.vocab_size = config.vocab_size

    def inject_moe_layers(self, moe_layers: Dict[int, nn.Module]) -> None:
        """
        Inject pre-built MoE layers by replacing MLP modules in transformer blocks.

        This performs true in-network injection where subsequent layers see MoE outputs.

        Args:
            moe_layers: Dictionary mapping layer indices to MoE layer instances
        """
        if not moe_layers:
            print("Warning: No MoE layers provided. Skipping injection.")
            return

        # Validate indices
        num_layers = len(self.backbone.transformer.h)
        indices_to_inject = list(moe_layers.keys())

        print(f"Injecting MoE layers at positions: {indices_to_inject}")

        for idx, tmoe_layer in moe_layers.items():
            if idx < 0 or idx >= num_layers:
                raise ValueError(f"Invalid layer index {idx} (num_layers={num_layers})")

            # Get the transformer block
            block = self.backbone.transformer.h[idx]

            # Replace the MLP module
            block.mlp = tmoe_layer

            # Store reference
            self.moe_layers[str(idx)] = tmoe_layer

            # Determine number of experts
            if hasattr(tmoe_layer, "expert_pool"):
                num_experts = tmoe_layer.expert_pool.num_experts
            elif hasattr(tmoe_layer, "experts"):
                num_experts = len(tmoe_layer.experts)
            else:
                num_experts = getattr(tmoe_layer, "num_experts", "unknown")

            print(f"  Layer {idx}: Replaced MLP with MoE Layer ({num_experts} experts)")

        print(
            f"MoE injection complete. Total layers modified: {len(indices_to_inject)}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, Any]]]:
        """
        Forward pass with in-network MoE processing.

        Now that MoE layers have replaced MLP modules, HuggingFace's forward pass
        automatically routes through them. We just need to collect metrics if requested.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels for loss computation [batch_size, seq_len]
            return_metrics: Whether to return routing metrics

        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            loss: Language modeling loss (if labels provided)
            metrics: Optional routing metrics from MoE layers
        """
        # Forward through backbone (MoE layers now integrated)
        # Note: output_hidden_states is only needed for the fallback metrics path
        # (when cached metrics are unavailable). The MoE layers cache routing state
        # during the main forward pass, so hidden_states are rarely needed.
        with torch.set_grad_enabled(not self.freeze_backbone or bool(self.moe_layers)):
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=return_metrics,
            )

        logits = outputs.logits
        loss = outputs.loss if labels is not None else None

        # Add auxiliary losses from MoE routers
        if loss is not None and self.moe_layers:
            aux_loss = 0.0
            for moe_layer in self.moe_layers.values():
                if hasattr(moe_layer, "router"):
                    aux_loss = aux_loss + moe_layer.router.compute_aux_loss()

            # Combine losses
            loss = loss + aux_loss
        # Collect MoE metrics if requested
        all_metrics = {}
        if return_metrics and self.moe_layers:
            # Retrieve cached routing metrics from the main forward pass.
            # MoE layers cache weights/indices during forward(), so we can
            # compute metrics without a second forward pass through experts.
            for layer_idx_str, moe_layer in self.moe_layers.items():
                layer_idx = int(layer_idx_str)

                # Use cached routing state (populated during the main forward pass)
                layer_metrics = None
                if hasattr(moe_layer, "get_cached_metrics"):
                    layer_metrics = moe_layer.get_cached_metrics()

                if layer_metrics is None:
                    # Fallback: run a separate forward pass if no cache available
                    # (e.g., first call or layer type doesn't support caching)
                    layer_hidden = outputs.hidden_states[layer_idx]
                    _, layer_metrics = moe_layer(
                        layer_hidden, return_metrics=True, record_usage=False
                    )

                if layer_metrics:
                    all_metrics[f"layer_{layer_idx}"] = layer_metrics

        return logits, loss, all_metrics if return_metrics else None
