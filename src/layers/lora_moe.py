from typing import Dict, Any, Optional, Tuple

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
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Args:
            x: ``[batch, seq, hidden]``
            return_metrics: pass routing diagnostics back

        Returns:
            output:  ``[batch, seq, hidden]``
            metrics: routing metrics dict (or ``None``)
        """
        batch, seq, hidden = x.shape

        # 1. Frozen backbone
        with torch.no_grad():
            base_out = self.base_layer(x)

        # 2. Routing
        # weights : [batch, seq, top_k]
        # indices : [batch, seq, top_k]  (values in 0 … num_experts-1)
        weights, indices, metrics = self.router(x, return_metrics=return_metrics)

        # 3. Expert dispatch — loop over *active* experts only
        x_flat = x.view(-1, hidden)  # [N, H]
        w_flat = weights.view(-1, weights.shape[-1])  # [N, K]
        idx_flat = indices.view(-1, indices.shape[-1])  # [N, K]
        lora_delta = torch.zeros_like(x_flat)  # accumulator

        for eid in idx_flat.unique().tolist():
            expert = self.expert_pool[eid]
            # mask: True wherever this expert was selected  [N, K]
            mask = idx_flat == eid
            # token indices that route to this expert (any top-k slot)
            token_ids = mask.any(dim=1)
            if not token_ids.any():
                continue

            # Run expert on selected tokens
            delta = expert(x_flat[token_ids])  # [n, H]

            # Sum the routing weights across the K slots for this expert
            # (a token might select the same expert in >1 slot, rare but valid)
            expert_w = (w_flat * mask.float())[token_ids].sum(
                dim=1, keepdim=True
            )  # [n, 1]

            lora_delta[token_ids] += delta * expert_w

        output = base_out + lora_delta.view(batch, seq, hidden)

        return output, metrics if return_metrics else None
