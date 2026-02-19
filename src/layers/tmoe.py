from typing import Optional, Type, Dict, Any, Tuple, Union

import torch
from torch import nn

from src.experts import BaseExpert
from src.layers import BaseMoELayer
from src.routers import BaseRouter, create_router
from src.project_types import RouterType


class TMoELayer(BaseMoELayer):
    """
    T-MoE Layer implementation.

    Combines metabolic routing with expert networks to create a dynamic
    mixture-of-experts layer that can be integrated into transformer models.

    Attributes:
        hidden_dim: Dimension of input/output embeddings
        num_experts: Number of expert networks
        top_k: Number of experts to route each token to
        router: The routing mechanism (metabolic or standard)
        experts: ModuleList of expert networks
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int = 8,
        expert_class: Optional[Type[BaseExpert]] = None,
        expert_kwargs: Optional[Dict[str, Any]] = None,
        router_type: RouterType = RouterType.METABOLIC,
        router_kwargs: Optional[Dict[str, Any]] = None,
        top_k: int = 2,
        use_parallel: bool = False,
    ):
        """
        Initialize TMoE layer.

        Args:
            hidden_dim: Dimension of input/output embeddings
            num_experts: Number of expert networks
            expert_class: Class to use for expert networks (optional)
            expert_kwargs: Additional arguments for expert construction
            router_type: Type of router ("metabolic" or "standard")
            router_kwargs: Additional arguments for router construction
            top_k: Number of experts to route each token to
            use_parallel: If True, use parallel batched expert computation
                         (more efficient for large num_experts). Default: False
        """
        # Initialize parent with direct parameters
        super().__init__(hidden_dim=hidden_dim, num_experts=num_experts, top_k=top_k)

        self.use_parallel = use_parallel

        # Create router
        router_kwargs = router_kwargs or {}
        self.router = self._create_router(
            hidden_dim, num_experts, top_k, router_type, router_kwargs
        )

        # Create experts
        expert_kwargs = expert_kwargs or {}
        if expert_class is not None:
            self.experts = nn.ModuleList(
                [
                    expert_class(hidden_dim=hidden_dim, **expert_kwargs)
                    for _ in range(num_experts)
                ]
            )
        else:
            # Placeholder - will be set by surgery or subclass
            self.experts = None

    def _create_router(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        router_type: Optional[RouterType],
        router_kwargs: Dict[str, Any],
    ) -> Optional[BaseRouter]:
        """
        Create router using the factory.

        Args:
            hidden_dim: Dimension of input embeddings
            num_experts: Number of experts to route to
            top_k: Number of top experts per token
            router_type: Type of router ("metabolic" or "standard"), or None to skip
            router_kwargs: Additional router configuration

        Returns:
            Configured router instance, or None if router_type is None

        Raises:
            ValueError: If router_type is not recognized
        """
        # Allow external router injection by passing router_type=None
        if router_type is None:
            return None

        return create_router(
            router_type=router_type,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            **router_kwargs,
        )

    def set_experts(self, experts: nn.ModuleList) -> None:
        """
        Set expert modules (used by surgery utilities).

        Args:
            experts: ModuleList of expert networks

        Raises:
            ValueError: If number of experts doesn't match num_experts
        """
        if len(experts) != self.num_experts:
            raise ValueError(f"Expected {self.num_experts} experts, got {len(experts)}")
        self.experts = experts

    def step(self) -> None:
        """
        Delegate to router's step method for convenience.

        This allows calling layer.step() instead of layer.router.step(),
        which is more ergonomic when managing multiple MoE layers.
        """
        if hasattr(self.router, "step"):
            self.router.step()

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_metrics: bool = False,
        record_usage: bool = True,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Dict[str, Any]]]]:
        """
        Forward pass through the MoE layer.

        Compatible with both standalone usage and GPT-Neo MLP replacement.

        Args:
            hidden_states: Input tensor [batch, seq, hidden_dim]
            return_metrics: Whether to return routing metrics
            record_usage: Whether to record usage for fatigue updates. Set to False
                         when collecting metrics from a separate forward pass.
            **kwargs: Additional arguments (for compatibility with transformer blocks)

        Returns:
            If return_metrics=False: output tensor [batch, seq, hidden_dim]
                (HuggingFace compatible - GPT-Neo block expects tensor)
            If return_metrics=True: tuple of (output, metrics dict)
        """
        x = hidden_states  # Rename for internal consistency

        # Validate inputs
        if self.experts is None:
            raise RuntimeError(
                "Experts not initialized. Either provide expert_class in __init__ "
                "or call set_experts() before forward pass."
            )

        if x.ndim != 3:
            raise ValueError(
                f"Expected 3D input [batch, seq, hidden], got shape {x.shape}"
            )

        if x.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Input hidden_dim mismatch: expected {self.hidden_dim}, got {x.shape[-1]}"
            )

        batch_size, seq_len, hidden_dim = x.shape

        # Get routing weights and indices (router expects 3D input)
        weights, indices, metrics = self.router(
            x, return_metrics=return_metrics, record_usage=record_usage, **kwargs
        )
        # weights: [batch, seq, top_k]
        # indices: [batch, seq, top_k]

        # Cache routing state for metric retrieval without re-running experts
        self._last_routing_weights = weights.detach()
        self._last_routing_indices = indices.detach()

        # Flatten to 2D for efficient expert processing
        x_flat = x.reshape(-1, hidden_dim)  # [batch*seq, hidden]
        weights_flat = weights.reshape(-1, self.top_k)  # [batch*seq, top_k]
        indices_flat = indices.reshape(-1, self.top_k)  # [batch*seq, top_k]

        # Process through experts (use parallel or sequential implementation)
        if self.use_parallel:
            expert_output_flat = self._forward_experts_parallel(
                x_flat, weights_flat, indices_flat
            )
        else:
            expert_output_flat = self._forward_experts(
                x_flat, weights_flat, indices_flat
            )
        # expert_output_flat: [batch*seq, hidden]

        # Reshape back to 3D: [batch, seq, hidden]
        output = expert_output_flat.reshape(batch_size, seq_len, hidden_dim)

        # Collect metrics if requested
        result_metrics = None
        if return_metrics and metrics is not None:
            result_metrics = {
                "weights": weights,  # Already [batch, seq, top_k]
                "indices": indices,  # Already [batch, seq, top_k]
                **metrics,  # Add router-specific metrics
            }

        # Return type depends on return_metrics flag:
        # - return_metrics=True: return (output, metrics) tuple for explicit metric collection
        # - return_metrics=False: return just output tensor for HuggingFace compatibility
        #   (GPT-Neo block expects MLP to return a tensor, not a tuple)
        if return_metrics:
            return output, result_metrics
        return output

    def _forward_experts(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Route tokens through selected experts and aggregate outputs.
        This implementation uses a loop over experts for clarity.
        For large-scale production, consider batched scatter/gather operations.

        Args:
            x: Flattened input [num_tokens, hidden]
            weights: Routing weights [num_tokens, top_k]
            indices: Expert indices [num_tokens, top_k]

        Returns:
            Aggregated output [num_tokens, hidden]
        """
        num_tokens, h_dim = x.shape
        output = torch.zeros_like(x)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find which tokens are routed to this expert (for any of the top_k slots)
            # mask: [num_tokens, top_k] - True where this expert is selected
            expert_mask = indices == expert_idx

            if not expert_mask.any():
                continue

            # Get token indices that use this expert
            # token_indices: which tokens, slot_indices: which top_k slot
            token_indices, slot_indices = torch.where(expert_mask)

            # Get the tokens for this expert (unique tokens only to avoid duplicate computation)
            unique_token_indices, inverse_indices = torch.unique(
                token_indices, return_inverse=True
            )
            expert_input = x[unique_token_indices]  # [num_unique_tokens, hidden]

            # Process through expert
            expert_output = self.experts[expert_idx](
                expert_input
            )  # [num_unique_tokens, hidden]

            # Map back to original indices for weighting
            expert_output_expanded = expert_output[
                inverse_indices
            ]  # [num_tokens, hidden]

            # Get weights for these tokens at these slots
            expert_weights = weights[token_indices, slot_indices]  # [num_tokens]

            # Weight the outputs
            weighted_output = expert_output_expanded * expert_weights.unsqueeze(-1)

            # Accumulate into output
            output.index_add_(0, token_indices, weighted_output)

        return output

    def _forward_experts_parallel(
        self,
        x: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Route tokens through selected experts using parallel batched operations.

        This approach processes all experts efficiently by:
        1. Sorting tokens by their assigned expert for coalesced memory access
        2. Splitting sorted tokens and processing each expert's batch in parallel
        3. Concatenating expert outputs and restoring original token order
        4. Aggregating outputs via reshape and sum operations

        This is more efficient than _forward_experts for large num_experts
        and production use cases, computing only the required expert outputs.

        Args:
            x: Flattened input [num_tokens, hidden]
            weights: Routing weights [num_tokens, top_k]
            indices: Expert indices [num_tokens, top_k]

        Returns:
            Aggregated output [num_tokens, hidden]
        """
        num_tokens, h_dim = x.shape

        # Expand tokens for each top_k selection
        # [num_tokens, top_k, hidden] -> [num_tokens * top_k, hidden]
        x_expanded = x.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, h_dim)

        # Flatten indices and weights: [num_tokens * top_k]
        flat_indices = indices.view(-1)
        flat_weights = weights.view(-1)

        # Sort by expert index for coalesced memory access
        sorted_expert_indices, sort_order = torch.sort(flat_indices)
        sorted_tokens = x_expanded[sort_order]  # [num_tokens * top_k, hidden]
        sorted_weights = flat_weights[sort_order]  # [num_tokens * top_k]

        # Count tokens per expert using bincount
        expert_counts = torch.bincount(
            sorted_expert_indices, minlength=self.num_experts
        )
        # Compute cumulative indices for slicing (single GPU sync via .tolist())
        cumsum = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=expert_counts.device),
                expert_counts.cumsum(0),
            ]
        )
        offsets = cumsum.tolist()  # Single GPU→CPU sync for all offsets

        # Process each expert's tokens (batched per expert)
        expert_outputs = []
        for expert_idx in range(self.num_experts):
            start_idx = offsets[expert_idx]
            end_idx = offsets[expert_idx + 1]

            if start_idx == end_idx:
                expert_outputs.append(
                    torch.empty(0, h_dim, device=x.device, dtype=x.dtype)
                )
                continue

            # Slice tokens and weights for this expert
            expert_tokens = sorted_tokens[start_idx:end_idx]
            expert_weights = sorted_weights[start_idx:end_idx]

            # Forward through expert (batched)
            out = self.experts[expert_idx](expert_tokens)  # [count, hidden]
            weighted_out = out * expert_weights.unsqueeze(-1)
            expert_outputs.append(weighted_out)

        # Concatenate all expert outputs (in sorted order)
        all_outputs = torch.cat(expert_outputs, dim=0)  # [num_tokens * top_k, hidden]

        # Unsort to restore original token order (O(n) scatter instead of O(n log n) argsort)
        restored_outputs = torch.empty_like(all_outputs)
        restored_outputs[sort_order] = all_outputs

        # Reshape and sum over top_k dimension
        restored_outputs = restored_outputs.view(num_tokens, self.top_k, h_dim)
        output = restored_outputs.sum(dim=1)  # [num_tokens, hidden]

        return output

    def get_routing_metrics(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Get routing metrics without full forward pass.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]

        Returns:
            Dictionary of routing metrics
        """
        with torch.no_grad():
            _, _, metrics = self.router(x, return_metrics=True)
        return metrics or {}

    def reset_router_state(self) -> None:
        """Reset router state (fatigue, statistics)."""
        self.router.reset_state()
