"""
T-MoE Profiling Script
======================
Profiles the 5 identified performance bottleneck functions using cProfile.

Usage:
    # Profile the full training pipeline (short run):
    python profiling/profile_tmoe.py --mode full --config experiments/smoketest.yaml

    # Profile individual functions in isolation:
    python profiling/profile_tmoe.py --mode individual

    # Compare before/after optimization:
    python profiling/profile_tmoe.py --mode compare
"""

import cProfile
import pstats
import io
import time
import os
import sys
import argparse
from pathlib import Path
from functools import wraps

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import numpy as np  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Utility: decorator to profile any single function
# ─────────────────────────────────────────────────────────────────────
def profile_function(func):
    """Decorator that profiles a function and prints stats."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()

        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats(30)
        print(stream.getvalue())
        return result

    return wrapper


# ─────────────────────────────────────────────────────────────────────
# Helper: create realistic mock inputs matching T-MoE dimensions
# ─────────────────────────────────────────────────────────────────────
def make_mock_inputs(
    batch=4, seq_len=512, hidden_dim=768, num_experts=8, top_k=2, device="cpu"
):
    """Create tensors that match the shapes used in T-MoE forward passes."""
    hidden_states = torch.randn(batch, seq_len, hidden_dim, device=device)
    # Simulate routing indices and weights
    indices = torch.randint(0, num_experts, (batch, seq_len, top_k), device=device)
    weights = torch.softmax(torch.randn(batch, seq_len, top_k, device=device), dim=-1)
    return hidden_states, indices, weights


# ═════════════════════════════════════════════════════════════════════
# BOTTLENECK 1: LoRAMoELayer.forward()
#   - Called every training step for every MoE layer (6 layers × steps)
#   - Contains a Python for-loop over active experts
#   - Each iteration runs expert forward + weighted accumulation
# ═════════════════════════════════════════════════════════════════════
def profile_lora_moe_forward():
    """Profile the MoE layer forward pass — the single hottest function."""
    from src.experts.lora import LoRAConfig
    from src.layers.lora_moe import LoRAMoELayer
    from src.routers.stress_corrected import StressCorrectedRouter
    from src.configs.router import StressCorrectedRouterConfig

    print("=" * 70)
    print("BOTTLENECK 1: LoRAMoELayer.forward()")
    print("  Why: Called every step × every MoE layer. Contains Python for-loop")
    print("       over experts with per-expert GPU kernel launches.")
    print("=" * 70)

    hidden_dim = 768
    num_experts = 8

    lora_cfg = LoRAConfig(hidden_dim=hidden_dim, rank=16, alpha=16)
    router_cfg = StressCorrectedRouterConfig(
        hidden_dim=hidden_dim, num_experts=num_experts, top_k=2
    )
    router = StressCorrectedRouter(router_cfg)

    # Build a standalone MoE layer with mock base weights
    from src.experts.pool import ExpertPool
    from src.project_types import ExpertType

    ExpertPool(lora_cfg, num_experts, ExpertType.GPTNEO_LORA)

    # We need a pretrained MLP to initialize — create a minimal mock
    class MockMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c_fc = torch.nn.Linear(hidden_dim, 4 * hidden_dim)
            self.c_proj = torch.nn.Linear(4 * hidden_dim, hidden_dim)

    mock_mlp = MockMLP()

    moe_layer = LoRAMoELayer.from_pretrained_mlp(
        mock_mlp, router, lora_cfg, num_experts
    )
    moe_layer.eval()

    x = torch.randn(4, 512, hidden_dim)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            moe_layer(x)

    # Profile
    profiler = cProfile.Profile()
    profiler.enable()
    with torch.no_grad():
        for _ in range(20):
            moe_layer(x)
    profiler.disable()

    print("\nTop 25 functions by cumulative time:")
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(25)

    # Save for later comparison
    stats.dump_stats("profiling/results/bottleneck1_moe_forward.prof")
    print("Saved to profiling/results/bottleneck1_moe_forward.prof\n")


# ═════════════════════════════════════════════════════════════════════
# BOTTLENECK 2: StressCorrectedRouter.forward() + _update_welford()
#   - Router forward: cosine similarity matrix [B*S, E], Gumbel noise,
#     SPAR logit computation, topk selection
#   - _update_welford: per-expert Welford variance tracking with
#     scatter operations on every training step
# ═════════════════════════════════════════════════════════════════════
def profile_stress_router():
    """Profile the stress-corrected router — cosine sim + Welford updates."""
    from src.routers.stress_corrected import StressCorrectedRouter
    from src.configs.router import StressCorrectedRouterConfig

    print("=" * 70)
    print("BOTTLENECK 2: StressCorrectedRouter.forward() + _update_welford()")
    print("  Why: Computes [B*S, E] cosine similarity matrix every step,")
    print("       plus Welford variance tracking with scatter_add_ ops.")
    print("=" * 70)

    hidden_dim = 768
    num_experts = 8
    cfg = StressCorrectedRouterConfig(
        hidden_dim=hidden_dim, num_experts=num_experts, top_k=2
    )
    router = StressCorrectedRouter(cfg)
    router.train()

    x = torch.randn(4, 512, hidden_dim)

    # Warmup
    for _ in range(3):
        router(x)

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(50):
        router(x)
    profiler.disable()

    print("\nTop 25 functions by cumulative time:")
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(25)

    stats.dump_stats("profiling/results/bottleneck2_stress_router.prof")
    print("Saved to profiling/results/bottleneck2_stress_router.prof\n")


# ═════════════════════════════════════════════════════════════════════
# BOTTLENECK 3: SharedLoRALayer.forward() (inside each expert)
#   - Called num_experts × num_moe_layers × steps times
#   - Each call: F.linear(x, shared_weight) + lora_B(lora_A(x)) * scaling
#   - dtype casting (bf16 ↔ fp32) on every call
# ═════════════════════════════════════════════════════════════════════
def profile_shared_lora_forward():
    """Profile the SharedLoRALayer — the innermost computation kernel."""
    from src.experts.lora import SharedLoRALayer

    print("=" * 70)
    print("BOTTLENECK 3: SharedLoRALayer.forward()")
    print("  Why: Called E × L × steps times. Each call does base linear +")
    print("       LoRA A→B projection with dtype casting overhead.")
    print("=" * 70)

    hidden_dim = 768
    intermediate_dim = 3072

    shared_w = torch.randn(intermediate_dim, hidden_dim)
    shared_b = torch.randn(intermediate_dim)

    layer = SharedLoRALayer(
        shared_weight=shared_w, shared_bias=shared_b, rank=16, alpha=16, dropout=0.0
    )

    x = torch.randn(2048, hidden_dim)  # flattened batch*seq tokens

    # Warmup
    for _ in range(5):
        layer(x)

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(100):
        layer(x)
    profiler.disable()

    print("\nTop 20 functions by cumulative time:")
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)

    stats.dump_stats("profiling/results/bottleneck3_shared_lora.prof")
    print("Saved to profiling/results/bottleneck3_shared_lora.prof\n")


# ═════════════════════════════════════════════════════════════════════
# BOTTLENECK 4: RouterMetricsTracker.compute_all_metrics()
#   - Called at every log_interval step
#   - Chains 6+ metric computations: entropy, gini, usage distribution,
#     effective experts, confidence, custom router metrics
#   - Each sub-metric does its own tensor reductions
# ═════════════════════════════════════════════════════════════════════
def profile_metrics_computation():
    """Profile the metrics tracker — aggregates 6+ metric computations."""
    from src.routers.stress_corrected import StressCorrectedRouter
    from src.configs.router import StressCorrectedRouterConfig
    from src.metrics.router_metrics import RouterMetricsTracker

    print("=" * 70)
    print("BOTTLENECK 4: RouterMetricsTracker.compute_all_metrics()")
    print("  Why: Chains entropy + gini + usage + effective_experts +")
    print("       confidence + custom metrics. Each does tensor reductions.")
    print("=" * 70)

    hidden_dim = 768
    num_experts = 8
    cfg = StressCorrectedRouterConfig(
        hidden_dim=hidden_dim, num_experts=num_experts, top_k=2
    )
    router = StressCorrectedRouter(cfg)
    tracker = RouterMetricsTracker(router)

    # Simulate routing outputs
    batch, seq, top_k = 4, 512, 2
    indices = torch.randint(0, num_experts, (batch, seq, top_k))
    weights = torch.softmax(torch.randn(batch, seq, top_k), dim=-1)

    # Warmup
    for _ in range(3):
        tracker.compute_all_metrics(indices, weights)

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(100):
        tracker.compute_all_metrics(indices, weights)
    profiler.disable()

    print("\nTop 25 functions by cumulative time:")
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(25)

    stats.dump_stats("profiling/results/bottleneck4_metrics.prof")
    print("Saved to profiling/results/bottleneck4_metrics.prof\n")


# ═════════════════════════════════════════════════════════════════════
# BOTTLENECK 5: ShardDataset.__getitem__() + write_split_to_disk()
#   - __getitem__: called every training step × batch_size
#     Binary search over shards + memmap reads + cross-shard stitching
#   - write_split_to_disk: row-by-row iteration with json.dumps per row
# ═════════════════════════════════════════════════════════════════════
def profile_data_loading():
    """Profile the data loading pipeline — shard reads + data writing."""
    print("=" * 70)
    print("BOTTLENECK 5: ShardDataset.__getitem__() + write_split_to_disk()")
    print("  Why: __getitem__ does bisect + memmap reads per sample.")
    print("       write_split_to_disk iterates row-by-row with json.dumps.")
    print("=" * 70)

    # --- Profile ShardDataset.__getitem__ ---
    # Create a temporary shard file for profiling
    import struct
    import tempfile

    tmpdir = Path(tempfile.mkdtemp())
    shard_path = tmpdir / "train_shard_000.bin"

    num_tokens = 100_000
    tokens = np.random.randint(0, 50257, size=num_tokens, dtype=np.uint16)
    with open(shard_path, "wb") as f:
        f.write(struct.pack("<Q", num_tokens))
        f.write(tokens.tobytes())

    # Import and create dataset
    sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.train import ShardDataset

    dataset = ShardDataset(tmpdir, "train", seq_len=512)

    # Warmup
    for i in range(5):
        dataset[i]

    profiler = cProfile.Profile()
    profiler.enable()
    for i in range(1000):
        dataset[i % len(dataset)]
    profiler.disable()

    print("\nShardDataset.__getitem__ — Top 20 by cumulative time:")
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stats.print_stats(20)

    stats.dump_stats("profiling/results/bottleneck5_data_loading.prof")
    print("Saved to profiling/results/bottleneck5_data_loading.prof")

    # --- Profile write_split_to_disk ---
    import json

    mock_data = [
        {"text": f"This is sample text number {i} for profiling."}
        for i in range(10_000)
    ]

    profiler2 = cProfile.Profile()
    profiler2.enable()
    # Inline the write logic to profile it without needing the full infra setup
    output_path = tmpdir / "profile_test.jsonl"
    with open(output_path, "w", encoding="utf-8") as fh:
        for row in mock_data:
            text = row.get("text")
            if text and text.strip():
                fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    profiler2.disable()

    print("\nwrite_split_to_disk (JSONL) — Top 15 by cumulative time:")
    stats2 = pstats.Stats(profiler2)
    stats2.sort_stats("cumulative")
    stats2.print_stats(15)

    stats2.dump_stats("profiling/results/bottleneck5_write_split.prof")
    print("Saved to profiling/results/bottleneck5_write_split.prof\n")

    # Cleanup
    import shutil

    shutil.rmtree(tmpdir)


# ═════════════════════════════════════════════════════════════════════
# FULL PROGRAM PROFILING
# ═════════════════════════════════════════════════════════════════════
def profile_full_program(config_path: str):
    """Profile the entire training script using cProfile."""
    print("=" * 70)
    print("FULL PROGRAM PROFILE")
    print(f"  Config: {config_path}")
    print("  This profiles the entire train.py main() function.")
    print("=" * 70)

    output_file = "profiling/results/full_program.prof"

    # Use cProfile.run to profile the training script
    cmd = f"""
import sys
sys.argv = ['scripts/train.py', '--config', '{config_path}', '--output-dir', 'profiling/results/train_output']
from scripts.train import main
main()
"""
    cProfile.run(cmd, output_file)

    print(f"\nFull profile saved to {output_file}")
    print("To analyze interactively:")
    print(
        f"  python -c \"import pstats; p = pstats.Stats('{output_file}'); p.sort_stats('cumulative'); p.print_stats(50)\""
    )
    print(f"  # Or use snakeviz: pip install snakeviz && snakeviz {output_file}")


# ═════════════════════════════════════════════════════════════════════
# BEFORE/AFTER COMPARISON
# ═════════════════════════════════════════════════════════════════════
def compare_before_after():
    """Run profiling, apply optimizations, re-profile, and compare."""
    print("=" * 70)
    print("BEFORE/AFTER OPTIMIZATION COMPARISON")
    print("=" * 70)

    results = {}

    # --- Benchmark: SharedLoRALayer dtype casting ---
    from src.experts.lora import SharedLoRALayer

    hidden_dim = 768
    intermediate_dim = 3072
    shared_w = torch.randn(intermediate_dim, hidden_dim)
    shared_b = torch.randn(intermediate_dim)
    layer = SharedLoRALayer(
        shared_weight=shared_w, shared_bias=shared_b, rank=16, alpha=16
    )

    x_fp32 = torch.randn(2048, hidden_dim)

    # BEFORE: mixed dtype (triggers .to() casts)
    x_bf16 = x_fp32.to(torch.bfloat16)
    layer_bf16_weights = SharedLoRALayer(
        shared_weight=shared_w.to(torch.bfloat16),
        shared_bias=shared_b.to(torch.bfloat16),
        rank=16,
        alpha=16,
    )

    # Warmup
    for _ in range(5):
        layer(x_fp32)
        layer_bf16_weights(x_bf16)

    # Time BEFORE (fp32 input, fp32 weights — no casting needed)
    t0 = time.perf_counter()
    for _ in range(200):
        layer(x_fp32)
    t_same_dtype = time.perf_counter() - t0

    # Time with dtype mismatch (bf16 input, fp32 shared_weight — triggers .to())
    layer_mixed = SharedLoRALayer(
        shared_weight=shared_w,  # fp32
        shared_bias=shared_b,  # fp32
        rank=16,
        alpha=16,
    )
    t0 = time.perf_counter()
    for _ in range(200):
        layer_mixed(x_bf16)  # bf16 input → triggers w.to(x.dtype) cast
    t_mixed_dtype = time.perf_counter() - t0

    results["SharedLoRALayer (same dtype)"] = t_same_dtype
    results["SharedLoRALayer (mixed dtype — casting)"] = t_mixed_dtype

    # --- Benchmark: Gini coefficient ---
    from src.metrics.router_metrics import RouterMetricsTracker
    from src.routers.stress_corrected import StressCorrectedRouter
    from src.configs.router import StressCorrectedRouterConfig

    cfg = StressCorrectedRouterConfig(hidden_dim=768, num_experts=8, top_k=2)
    router = StressCorrectedRouter(cfg)
    tracker = RouterMetricsTracker(router)

    indices = torch.randint(0, 8, (4, 512, 2))
    weights = torch.softmax(torch.randn(4, 512, 2), dim=-1)

    # BEFORE: compute_all_metrics (recomputes usage internally for each sub-metric)
    t0 = time.perf_counter()
    for _ in range(500):
        tracker.compute_all_metrics(indices, weights)
    t_metrics = time.perf_counter() - t0
    results["compute_all_metrics (500 calls)"] = t_metrics

    # --- Benchmark: write_split_to_disk row-by-row vs batch ---
    import json
    import tempfile

    mock_data = [
        {"text": f"Sample text {i} for benchmarking write performance."}
        for i in range(10_000)
    ]
    tmpdir = Path(tempfile.mkdtemp())

    # BEFORE: row-by-row json.dumps
    t0 = time.perf_counter()
    with open(tmpdir / "before.jsonl", "w") as f:
        for row in mock_data:
            text = row.get("text")
            if text and text.strip():
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    t_row_by_row = time.perf_counter() - t0

    # AFTER: batch with StringIO buffer
    t0 = time.perf_counter()
    buf = io.StringIO()
    for row in mock_data:
        text = row.get("text")
        if text and text.strip():
            buf.write(json.dumps({"text": text}, ensure_ascii=False))
            buf.write("\n")
    with open(tmpdir / "after.jsonl", "w") as f:
        f.write(buf.getvalue())
    t_buffered = time.perf_counter() - t0

    results["write_split row-by-row"] = t_row_by_row
    results["write_split buffered (StringIO)"] = t_buffered

    # Cleanup
    import shutil

    shutil.rmtree(tmpdir)

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"{'Benchmark':<45} {'Time (s)':>10}")
    print("-" * 60)
    for name, t in results.items():
        print(f"{name:<45} {t:>10.4f}")
    print("=" * 60)


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="T-MoE Performance Profiling")
    parser.add_argument(
        "--mode",
        choices=["full", "individual", "compare"],
        default="individual",
        help="full: profile entire training run; individual: profile each bottleneck; compare: before/after",
    )
    parser.add_argument(
        "--config",
        default="experiments/smoketest.yaml",
        help="Config for full profiling",
    )
    parser.add_argument(
        "--bottleneck",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=None,
        help="Profile only a specific bottleneck (1-5)",
    )
    args = parser.parse_args()

    os.makedirs("profiling/results", exist_ok=True)

    if args.mode == "full":
        profile_full_program(args.config)

    elif args.mode == "compare":
        compare_before_after()

    else:  # individual
        bottlenecks = {
            1: profile_lora_moe_forward,
            2: profile_stress_router,
            3: profile_shared_lora_forward,
            4: profile_metrics_computation,
            5: profile_data_loading,
        }

        if args.bottleneck:
            bottlenecks[args.bottleneck]()
        else:
            for idx, fn in bottlenecks.items():
                fn()


if __name__ == "__main__":
    main()
