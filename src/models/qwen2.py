from typing import Optional, Dict, Any, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from src.core import ModelRegistry
from src.models.base import BaseModelBackbone
from src.project_types import ModelType
from src.training.precision import COMPUTE_DTYPE


@ModelRegistry.register(ModelType.QWEN2.value)
class Qwen2Backbone(BaseModelBackbone):
    VARIANTS = {
        "1.5b": {
            "hf_name": "Qwen/Qwen2-1.5B",
            "hidden_dim": 1536,
            "num_layers": 28,
            "intermediate_dim": 8960,
            "tokenizer_vocab_size": 151936,
            "description": "Qwen2 1.5B parameters",
        }
    }

    def __init__(
        self,
        variant: str = "1.5b",
        freeze_backbone: bool = True,
        moe_layer_indices: Optional[list[int]] = None,
        device: str = "cpu",
    ):
        if variant not in self.VARIANTS:
            available = ", ".join(self.VARIANTS.keys())
            raise ValueError(f"Invalid variant '{variant}'. Available: {available}")

        variant_config = self.VARIANTS[variant]
        super().__init__(
            model_name=variant_config["hf_name"],
            hidden_dim=variant_config["hidden_dim"],
            freeze_backbone=freeze_backbone,
            moe_layer_indices=moe_layer_indices,
        )

        self.variant = variant
        self.device = device
        self.vocab_size = None
        self.num_layers = variant_config["num_layers"]

        self.load_pretrained()

        if freeze_backbone:
            self.freeze_parameters()

    def load_pretrained(self) -> None:
        kwargs = {
            "dtype": COMPUTE_DTYPE,
        }
        # Use flash_attention_2 when available (requires CUDA + flash-attn package).
        # Falls back to HF's default (sdpa) otherwise — e.g., during tests on CPU.
        try:
            import flash_attn  # noqa: F401

            if torch.cuda.is_available():
                kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            pass
        self.backbone = AutoModelForCausalLM.from_pretrained(
            self.model_name, **kwargs
        ).to(self.device)
        self.vocab_size = self.backbone.config.vocab_size

    def get_mlp_at(self, idx: int) -> nn.Module:
        return self.backbone.model.layers[idx].mlp

    def inject_moe_layers(self, moe_layers: Dict[int, nn.Module]) -> None:
        if not moe_layers:
            return
        num_layers = len(self.backbone.model.layers)
        for idx, tmoe_layer in moe_layers.items():
            if idx < 0 or idx >= num_layers:
                raise ValueError(f"Invalid layer index {idx} (num_layers={num_layers})")
            self.backbone.model.layers[idx].mlp = tmoe_layer
            self.moe_layers[str(idx)] = tmoe_layer

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_metrics: bool = False,
        record_usage: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, Any]]]:
        # Thread record_usage into each MoE layer. HuggingFace's inner forward
        # calls mlp(hidden_states) with no kwargs, so we use a per-layer attribute
        # that LoRAMoELayer.forward() reads when no explicit kwarg is passed.
        for moe_layer in self.moe_layers.values():
            moe_layer._forced_record_usage = record_usage

        needs_grad = not self.freeze_backbone or bool(self.moe_layers)
        with torch.set_grad_enabled(needs_grad and torch.is_grad_enabled()):
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
                use_cache=False,
            )

        logits = outputs.logits
        loss = outputs.loss if labels is not None else None

        if loss is not None and self.moe_layers:
            aux_loss = sum(
                moe_layer.router.compute_aux_loss()
                for moe_layer in self.moe_layers.values()
                if hasattr(moe_layer, "router")
            )
            loss = loss + aux_loss

        all_metrics = {}
        if return_metrics and self.moe_layers:
            for layer_idx_str, moe_layer in self.moe_layers.items():
                if hasattr(moe_layer, "get_cached_metrics"):
                    layer_metrics = moe_layer.get_cached_metrics()
                    if layer_metrics:
                        # Log per-layer aux_loss: nonzero for standard/switch, zero for SPAR.
                        if hasattr(moe_layer, "router"):
                            layer_metrics["aux_loss"] = (
                                moe_layer.router.compute_aux_loss().item()
                            )
                        all_metrics[f"layer_{int(layer_idx_str)}"] = layer_metrics

        return logits, loss, all_metrics if return_metrics else None
