import inspect
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from src.routers.base import BaseRouter
from src.experts.lora import LoRAConfig, SharedBaseLoRA
from src.experts.pool import ExpertPool
from src.layers.base import BaseMoELayer
from src.project_types import ExpertType


class LoRAMoELayer(BaseMoELayer):
    """
    Drop-in replacement for a transformer MLP.

    Keeps the original layer frozen; LoRA experts add weighted deltas.
    Prefer ``from_pretrained_mlp`` over manual two-step initialization.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        router: BaseRouter,
        lora_config: LoRAConfig,
        num_experts: int = 4,
        expert_type: ExpertType = ExpertType.GPTNEO_LORA,
    ):
        super().__init__(
            hidden_dim=lora_config.hidden_dim,
            num_experts=num_experts,
            top_k=router.top_k,
        )
        # NOTE: base_layer is NOT stored as a submodule. Its weights are already
        # captured in each expert's SharedLoRALayer buffers (shared_weight/shared_bias)
        # after load_from_mlp(). Storing it would waste ~4.7M frozen params per layer
        # under FSDP (sharded/all-gathered but never used in forward).

        self.router = router
        self.expert_pool = ExpertPool(lora_config, num_experts, expert_type)
        self._router_accepts_record_usage = (
            "record_usage" in inspect.signature(router.forward).parameters
        )
        # Shared base LoRA: applies a trainable delta on c_proj to ALL tokens.
        # Instantiated here with placeholder dims; weights are set in _init_shared_base_lora()
        # called after load_from_mlp() populates expert 0's layer shapes.
        self.shared_proj_lora: Optional[SharedBaseLoRA] = None
        self._shared_base_rank = lora_config.shared_base_rank
        self._shared_base_alpha = lora_config.shared_base_alpha

    def _init_shared_base_lora(self) -> None:
        if self._shared_base_rank <= 0:
            return
        e0 = self.expert_pool.experts[0]
        out_proj = getattr(e0, "c_proj", None) or getattr(e0, "down_proj", None)
        if out_proj is None:
            return
        out_features, in_features = out_proj.shared_weight.shape
        self.shared_proj_lora = SharedBaseLoRA(
            in_features=in_features,
            out_features=out_features,
            rank=self._shared_base_rank,
            alpha=self._shared_base_alpha,
        )

    @classmethod
    def from_pretrained_mlp(
        cls,
        mlp: nn.Module,
        router: BaseRouter,
        lora_config: LoRAConfig,
        num_experts: int = 4,
        expert_type: ExpertType = ExpertType.GPTNEO_LORA,
    ) -> "LoRAMoELayer":
        """Build the layer and load base weights atomically."""
        layer = cls(mlp, router, lora_config, num_experts, expert_type)
        layer.expert_pool.load_from_mlp(mlp)
        layer._init_shared_base_lora()
        return layer

    def get_cached_metrics(self) -> Optional[Dict[str, Any]]:
        if (
            hasattr(self, "_last_routing_weights")
            and self._last_routing_weights is not None
        ):
            # compute_all_metrics already calls router.get_custom_metrics() internally
            metrics = self.router.metrics_tracker.compute_all_metrics(
                self._last_routing_indices,
                self._last_routing_weights,
            )
            # Include raw weights/indices so trainer._log_metrics() can gate on them
            metrics["weights"] = self._last_routing_weights
            if self._last_routing_indices is not None:
                metrics["indices"] = self._last_routing_indices

            # LoRA adapter metrics: per-expert delta magnitude ||B @ A||_F * scaling.
            # Batch across experts per projection to reduce CUDA syncs from N*2 to 2.
            with torch.no_grad():
                norms = None  # [num_experts] accumulated on-device
                for attr in ("c_fc", "c_proj", "gate_proj", "up_proj", "down_proj"):
                    layers = [getattr(e, attr, None) for e in self.expert_pool.experts]
                    valid = [
                        (layer, i)
                        for i, layer in enumerate(layers)
                        if layer is not None
                        and hasattr(layer, "lora_A")
                        and hasattr(layer, "lora_B")
                    ]
                    if not valid:
                        continue
                    # Stack all [rank, out] @ [in, rank] matmuls — one per expert
                    stacked = torch.stack(
                        [
                            (layer.lora_B.weight @ layer.lora_A.weight) * layer.scaling
                            for layer, _ in valid
                        ]
                    )  # [E, out, in]
                    expert_norms = stacked.flatten(1).norm(dim=1)  # [E]
                    if norms is None:
                        norms = torch.zeros(
                            len(self.expert_pool.experts),
                            device=expert_norms.device,
                            dtype=expert_norms.dtype,
                        )
                    for norm_val, (_, expert_i) in zip(expert_norms, valid):
                        norms[expert_i] = norms[expert_i] + norm_val
                if norms is not None:
                    metrics["lora_delta_norm_per_expert"] = norms.tolist()

            return metrics
        return None

    def step(self) -> None:
        if hasattr(self.router, "step"):
            self.router.step()

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_metrics: bool = False,
        record_usage: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Dict[str, Any]]]]:
        x = hidden_states
        batch, seq, hidden = x.shape

        # record_usage=None means "use whatever the layer-level override says".
        # GPTNeoBackbone.forward() sets self._forced_record_usage before calling
        # backbone() so the HuggingFace inner call (which passes no kwargs) picks
        # up the right value without requiring changes to the HF forward signature.
        effective_record_usage = (
            record_usage
            if record_usage is not None
            else getattr(self, "_forced_record_usage", True)
        )

        router_kwargs: Dict[str, Any] = {"return_metrics": return_metrics}
        if self._router_accepts_record_usage:
            router_kwargs["record_usage"] = effective_record_usage
        weights, indices, metrics = self.router(x, **router_kwargs)

        self._last_routing_weights = weights.detach()
        self._last_routing_indices = indices.detach() if indices is not None else None

        x_flat = x.view(-1, hidden).contiguous()
        combined = torch.zeros_like(x_flat)

        # Unified Dispatcher: mask-based expert accumulation over dense (N, E) weights
        for expert_idx in range(self.expert_pool.num_experts):
            expert = self.expert_pool[expert_idx]

            expert_w_col = weights[:, expert_idx]
            mask = expert_w_col > 0

            if not mask.any():
                continue

            token_ids = mask.nonzero().squeeze(-1)
            expert_in = x_flat[token_ids]
            expert_out = expert(expert_in)
            scale = expert_w_col[token_ids].unsqueeze(-1).to(expert_out.dtype)
            combined[token_ids] += expert_out * scale

        if self.shared_proj_lora is not None:
            # Compute frozen fc hidden state for ALL tokens (no grad — frozen path).
            # expert 0's shared_weight is the canonical frozen buffer post-consolidation.
            e0 = self.expert_pool.experts[0]
            fc_w = e0.c_fc.shared_weight
            fc_b = e0.c_fc.shared_bias
            if fc_w.dtype != x_flat.dtype:
                fc_w = fc_w.to(x_flat.dtype)
                if fc_b is not None:
                    fc_b = fc_b.to(x_flat.dtype)
            with torch.no_grad():
                h_all = e0.act(torch.nn.functional.linear(x_flat, fc_w, fc_b))
            # Trainable delta on c_proj for all tokens — gradient flows through lora_A/B only.
            combined = combined + self.shared_proj_lora(h_all)

        output = combined.view(batch, seq, hidden)

        if return_metrics:
            if metrics is None:
                metrics = {}
            metrics["weights"] = weights.detach()
            if indices is not None:
                metrics["indices"] = indices.detach()
            return output, metrics
        return output
