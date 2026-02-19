from typing import Any, Dict, Tuple, Union

import torch
from torch import nn

from src.routers import BaseRouter
from src.experts.lora import LoRAConfig
from src.experts.pool import ExpertPool


class LoRAMoELayer(nn.Module):
    """
    Drop-in replacement for a transformer MLP layer.

    The original MLP is kept frozen; a set of lightweight LoRA experts
    produces additive deltas weighted by the Router.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        router: BaseRouter,
        lora_config: LoRAConfig,
        num_experts: int = 4,
        expert_type: str = "gpt_neo_lora",
    ):
        super().__init__()

        # Frozen backbone
        self.base_layer = base_layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        # Router + experts
        self.router = router
        self.expert_pool = ExpertPool(lora_config, num_experts, expert_type)

    # ── forward ──

    def forward(
        self,
        x: torch.Tensor,
        return_metrics: bool = False,
        record_usage: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Args:
            x: ``[batch, seq, hidden]``
            return_metrics: pass routing diagnostics back
            record_usage: forwarded to the router for fatigue tracking

        Returns:
            If return_metrics is True: ``(output, metrics)``
            Otherwise: plain ``output`` tensor (HuggingFace compatible)
        """
        batch, seq, hidden = x.shape

        # 1. Frozen backbone
        with torch.no_grad():
            base_out = self.base_layer(x)

        # 2. Routing
        # weights : [batch, seq, top_k]
        # indices : [batch, seq, top_k]  (values in 0 … num_experts-1)
        import inspect

        router_params = inspect.signature(self.router.forward).parameters
        router_kwargs = {"return_metrics": return_metrics}
        if "record_usage" in router_params:
            router_kwargs["record_usage"] = record_usage
        weights, indices, metrics = self.router(x, **router_kwargs)

        # 3. Expert dispatch — loop over all experts, skip inactive ones
        x_flat = x.view(-1, hidden)  # [N, H]
        w_flat = weights.view(-1, weights.shape[-1])  # [N, K]
        idx_flat = indices.view(-1, indices.shape[-1])  # [N, K]
        lora_delta = torch.zeros_like(x_flat)  # accumulator

        for expert_idx in range(self.expert_pool.num_experts):
            expert = self.expert_pool[expert_idx]
            # mask: True wherever this expert was selected  [N, K]
            mask = idx_flat == expert_idx  # stays on GPU, no .tolist() sync
            # token indices that route to this expert (any top-k slot)
            token_ids = mask.any(dim=1)
            if not token_ids.any():
                continue

            # Run expert on selected tokens
            delta = expert(x_flat[token_ids])  # [n, H]

            # Sum the routing weights across the K slots for this expert
            expert_w = (w_flat * mask.float())[token_ids].sum(
                dim=1, keepdim=True
            )  # [n, 1]

            lora_delta[token_ids] += delta * expert_w

        output = base_out + lora_delta.view(batch, seq, hidden)

        if return_metrics:
            return output, metrics
        return output
