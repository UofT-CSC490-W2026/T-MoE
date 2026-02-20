import inspect
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import nn

from src.routers.base import BaseRouter
from src.experts.lora import LoRAConfig
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
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        self.router = router
        self.expert_pool = ExpertPool(lora_config, num_experts, expert_type)
        self._router_accepts_record_usage = (
            "record_usage" in inspect.signature(router.forward).parameters
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
        return layer

    def get_cached_metrics(self) -> Optional[Dict[str, Any]]:
        """Return metrics from most recent forward pass."""
        if (
            hasattr(self, "_last_routing_weights")
            and self._last_routing_weights is not None
        ):
            return self.router.metrics_tracker.compute_all_metrics(
                self._last_routing_indices, self._last_routing_weights
            )
        return None

    def step(self) -> None:
        """Delegate to router's step method (applies fatigue)."""
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
        w_flat = weights.view(-1, weights.shape[-1])
        idx_flat = indices.view(-1, indices.shape[-1])
        lora_delta = torch.zeros_like(x_flat)

        for expert_idx in range(self.expert_pool.num_experts):
            expert = self.expert_pool[expert_idx]
            mask = idx_flat == expert_idx
            token_ids = mask.any(dim=1)
            if not token_ids.any():
                continue
            delta = expert(x_flat[token_ids])
            expert_w = (w_flat * mask.float())[token_ids].sum(dim=1, keepdim=True)
            lora_delta[token_ids] += delta * expert_w

        output = lora_delta.view(batch, seq, hidden)

        if return_metrics:
            if metrics is None:
                metrics = {}
            metrics["weights"] = weights
            metrics["indices"] = indices
            return output, metrics
        return output
