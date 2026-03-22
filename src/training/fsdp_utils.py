from __future__ import annotations

import os
import warnings
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


def init_distributed() -> Tuple[bool, int, int, int]:
    """Returns (is_distributed, rank, local_rank, world_size)."""
    if "RANK" not in os.environ:
        return False, 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA.")

    if local_rank >= torch.cuda.device_count():
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} GPU(s) visible."
        )

    dist.init_process_group(
        backend="nccl",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    torch.cuda.set_device(local_rank)
    dist.barrier()

    if rank == 0:
        print(f"Distributed training: rank={rank}, world_size={world_size}")

    return True, rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_model_for_attr_access(model: nn.Module) -> nn.Module:
    """Unwrap DDP. For FSDP, access attrs before wrapping."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    return model.module if isinstance(model, DDP) else model


def wrap_model_for_distributed(
    model: nn.Module, cfg, local_rank: int, device: torch.device
) -> nn.Module:
    """
    Dispatch to DDP or FSDP based on distributed.strategy config.

    DDP  (default) — use for models that fit in a single GPU:
        GPT-Neo 125M, Llama 1B, 3B, 8B on A100 80GB
    FSDP           — use when model DOES NOT fit in a single GPU:
        Llama 70B, 405B (requires sharding params across GPUs)

    Config:
        distributed:
          strategy: ddp    # or: fsdp
          # FSDP-only options:
          sharding_strategy: SHARD_GRAD_OP   # FULL_SHARD for 70B+
          fsdp_wrap_target: LlamaDecoderLayer # module class to wrap per-block
    """
    dist_cfg = getattr(cfg, "distributed", {})
    strategy = getattr(dist_cfg, "strategy", "ddp").lower()

    if strategy == "fsdp":
        return wrap_model_with_fsdp(model, cfg, device)

    # Default: DDP
    return wrap_model_with_ddp(model, local_rank, cfg)


def wrap_model_with_ddp(model: nn.Module, local_rank: int, cfg=None) -> nn.Module:
    """
    Wrap model with DistributedDataParallel for multi-GPU training.

    Use DDP (not FSDP) when:
    - Model fits in a single GPU (all sizes up to ~7B on A100 80GB with LoRA)
    - Backbone is mostly frozen + small trainable adapters (LoRA, router)

    Why DDP over FSDP here:
    - DDP preserves requires_grad flags correctly for mixed frozen/trainable modules
    - No FlatParameter issues that corrupt the trainable param count
    - No complex state dict API — just model.module.state_dict()
    - Standard approach for all LoRA/PEFT fine-tuning work

    find_unused_parameters: with top-k routing and large batches, all experts will
    be selected every step — set False (default) to eliminate the autograd graph
    traversal overhead. Only enable for tiny-batch debugging where an expert might
    receive zero tokens in a single step.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP

    find_unused = getattr(
        getattr(cfg, "distributed", {}), "find_unused_parameters", False
    )

    wrapped = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused,
    )

    if dist.is_initialized():
        dist.barrier()

    if is_main_process():
        print(
            f"DDP enabled | device_ids=[{local_rank}] | find_unused_parameters={find_unused}"
        )

    return wrapped


def wrap_model_with_fsdp(model: nn.Module, cfg, device: torch.device) -> nn.Module:
    """
    Wrap model with FSDP using per-GPTNeoBlock sharding.

    DO NOT call model.to(device) before this — FSDP handles device placement.
    """
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock

    from src.training.precision import COMPUTE_DTYPE, is_mixed_precision

    dist_cfg = getattr(cfg, "distributed", {})
    use_mixed_precision = getattr(dist_cfg, "mixed_precision", True)
    use_cpu_offload = getattr(dist_cfg, "cpu_offload", False)
    use_activation_checkpointing = getattr(dist_cfg, "activation_checkpointing", False)

    # Mixed precision
    mp_policy = None
    if use_mixed_precision and is_mixed_precision():
        mp_policy = MixedPrecision(
            param_dtype=COMPUTE_DTYPE,
            reduce_dtype=torch.float32,
            buffer_dtype=COMPUTE_DTYPE,
        )

    cpu_offload = CPUOffload(offload_params=True) if use_cpu_offload else None

    # Sharding strategy
    strategy_name = getattr(dist_cfg, "sharding_strategy", "SHARD_GRAD_OP").upper()
    sharding_strategy = {
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }.get(strategy_name, ShardingStrategy.SHARD_GRAD_OP)

    auto_wrap_policy = ModuleWrapPolicy({GPTNeoBlock})

    # Suppress expected mixed frozen/trainable param warning
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*has both parameters with requires_grad.*"
        )
        wrapped = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=sharding_strategy,
            mixed_precision=mp_policy,
            cpu_offload=cpu_offload,
            device_id=device,
            use_orig_params=True,
        )

    # Activation checkpointing
    if use_activation_checkpointing:
        from src.layers.lora_moe import LoRAMoELayer

        _apply_activation_checkpointing(wrapped, LoRAMoELayer)

    # Post-wrap validation
    _fsdp_count = sum(
        1 for m in wrapped.modules() if isinstance(m, FSDP) and m is not wrapped
    )

    if is_main_process():
        print(
            f"FSDP enabled | strategy={strategy_name} "
            f"| wrap_target=GPTNeoBlock "
            f"| inner_fsdp_units={_fsdp_count} "
            f"| compute_dtype={COMPUTE_DTYPE} "
            f"| mixed_precision={use_mixed_precision and is_mixed_precision()} "
            f"| cpu_offload={use_cpu_offload} "
            f"| act_ckpt={use_activation_checkpointing}"
        )
        if _fsdp_count == 0:
            print("⚠️  No inner modules wrapped — verify GPTNeoBlock instances exist.")

    if dist.is_initialized():
        dist.barrier()

    return wrapped


def _apply_activation_checkpointing(fsdp_model: nn.Module, leaf_cls: type) -> None:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=checkpoint_wrapper,
        check_fn=lambda m: isinstance(m, leaf_cls),
    )
    if is_main_process():
        print(f"Activation checkpointing applied to {leaf_cls.__name__}")
