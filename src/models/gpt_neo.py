from typing import Optional, Dict, Any, Tuple

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig

from src.core import ModelRegistry
from src.models.base import BaseModelBackbone
from src.project_types import ModelType
from src.training.precision import COMPUTE_DTYPE


@ModelRegistry.register(ModelType.GPTNEO.value)
class GPTNeoBackbone(BaseModelBackbone):
    # Model variant configurations
    VARIANTS = {
        "125m": {
            "hf_name": "EleutherAI/gpt-neo-125m",
            "hidden_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "tokenizer_vocab_size": 50257,  # GPT-2 BPE — shared by all GPT-Neo variants
            "description": "GPT-Neo 125M parameters",
        },
        "1.3b": {
            "hf_name": "EleutherAI/gpt-neo-1.3b",
            "hidden_dim": 2048,
            "num_layers": 24,
            "num_heads": 16,
            "tokenizer_vocab_size": 50257,
            "description": "GPT-Neo 1.3B parameters",
        },
        "2.7b": {
            "hf_name": "EleutherAI/gpt-neo-2.7b",
            "hidden_dim": 2560,
            "num_layers": 32,
            "num_heads": 20,
            "tokenizer_vocab_size": 50257,
            "description": "GPT-Neo 2.7B parameters",
        },
    }

    def __init__(
        self,
        variant: str = "125m",
        freeze_backbone: bool = True,
        moe_layer_indices: Optional[list[int]] = None,
        device: str = "cpu",
    ):
        if variant not in self.VARIANTS:
            available = ", ".join(self.VARIANTS.keys())
            raise ValueError(f"Invalid variant '{variant}'. Available: {available}")

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

        self.load_pretrained()

        if freeze_backbone:
            self.freeze_parameters()

    @classmethod
    def get_variant_info(cls, variant: str) -> Dict[str, Any]:
        if variant not in cls.VARIANTS:
            available = ", ".join(cls.VARIANTS.keys())
            raise ValueError(f"Invalid variant '{variant}'. Available: {available}")
        return cls.VARIANTS[variant].copy()

    @classmethod
    def list_variants(cls) -> list[str]:
        return list(cls.VARIANTS.keys())

    def load_pretrained(self) -> None:
        self.backbone = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=COMPUTE_DTYPE,
        ).to(self.device)

        config = AutoConfig.from_pretrained(self.model_name)
        self.vocab_size = config.vocab_size

    def get_mlp_at(self, idx: int) -> nn.Module:
        return self.backbone.transformer.h[idx].mlp

    def inject_moe_layers(self, moe_layers: Dict[int, nn.Module]) -> None:
        if not moe_layers:
            return

        num_layers = len(self.backbone.transformer.h)
        for idx, tmoe_layer in moe_layers.items():
            if idx < 0 or idx >= num_layers:
                raise ValueError(f"Invalid layer index {idx} (num_layers={num_layers})")
            self.backbone.transformer.h[idx].mlp = tmoe_layer
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
        # output_hidden_states=False: MoE layers cache their own metrics; enabling this
        # wastes ~3 GB VRAM at 1.3B for data we don't use.
        # Respect outer torch.no_grad(): set_grad_enabled(True) would override it,
        # causing OOM + NCCL timeout during eval.

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
                use_cache=False,  # no KV cache — reduces memory and avoids recompilation with compile=True
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
