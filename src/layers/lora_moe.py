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
        """
        Instantiate shared_proj_lora after expert weights are loaded.

        Reads intermediate_dim and hidden_dim from expert 0's c_proj shape.
        Must be called after expert_pool.load_from_mlp() so c_proj is populated.
        """
        if self._shared_base_rank <= 0:
            return
        e0 = self.expert_pool.experts[0]
        if not (hasattr(e0, "c_proj") and e0.c_proj is not None):
            return
        # c_proj: [hidden_dim, intermediate_dim] in Linear convention [out, in]
        out_features, in_features = e0.c_proj.shared_weight.shape
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
            metrics["indices"] = self._last_routing_indices

            # LoRA adapter metrics: per-expert delta magnitude ||B @ A||_F * scaling.
            # Batch across experts per projection to reduce CUDA syncs from N*2 to 2.
            with torch.no_grad():
                norms = None  # [num_experts] accumulated on-device
                for attr in ("c_fc", "c_proj"):
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
        record_usage: bool = True,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Dict[str, Any]]]]:
        x = hidden_states
        batch, seq, hidden = x.shape

        router_kwargs: Dict[str, Any] = {"return_metrics": return_metrics}
        if self._router_accepts_record_usage:
            router_kwargs["record_usage"] = record_usage
        weights, indices, metrics = self.router(x, **router_kwargs)

        # Cache routing state for metric retrieval without re-running experts
        self._last_routing_weights = weights.detach()
        self._last_routing_indices = indices.detach()

        x_flat = x.view(-1, hidden)
        w_flat = weights.view(-1, weights.shape[-1])  # [BS, k]
        idx_flat = indices.view(-1, indices.shape[-1])  # [BS, k]
        combined = torch.zeros_like(x_flat)

        # Trainable shared base: pass the parameter tensors explicitly so that
        # F.linear() in SharedLoRALayer receives the parameter (not .data), allowing
        # gradients to flow back to ExpertPool.shared_fc/proj_weight.
        _fc_w = self.expert_pool.shared_fc_weight
        _proj_w = self.expert_pool.shared_proj_weight

        # Precompute active expert set — avoids iterating over experts with zero tokens.
        active_experts = idx_flat.unique()

        for expert_idx in active_experts.tolist():
            expert = self.expert_pool[expert_idx]
            # token_ids: bool mask [BS] — which tokens are assigned to this expert
            token_ids = (idx_flat == expert_idx).any(dim=1)
            if _fc_w is not None:
                expert_out = expert(
                    x_flat[token_ids], fc_weight=_fc_w, proj_weight=_proj_w
                )
            else:
                expert_out = expert(x_flat[token_ids])
            # Sum weights for this expert across the k slots (at most one slot has
            # expert_idx per token for standard top-k; sum handles k>1 correctly).
            mask = (idx_flat[token_ids] == expert_idx).to(w_flat.dtype)  # [n_tok, k]
            expert_w = (w_flat[token_ids] * mask).sum(dim=1, keepdim=True)
            combined[token_ids] += expert_out * expert_w

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
            metrics["indices"] = indices.detach()
            return output, metrics
        return output
