# T-MoE Performance Profiling Analysis

## How to Run

```bash
# Profile all 5 bottlenecks individually
python profiling/profile_tmoe.py --mode individual

# Profile just one (e.g., bottleneck 2)
python profiling/profile_tmoe.py --mode individual --bottleneck 2

# Profile the full training pipeline
python profiling/profile_tmoe.py --mode full --config experiments/smoketest.yaml

# Run before/after optimization comparison
python profiling/profile_tmoe.py --mode compare

# Visualize saved .prof files interactively (install snakeviz first)
pip install snakeviz
snakeviz profiling/results/bottleneck1_moe_forward.prof
```

---

## Understanding cProfile Output

Each row in cProfile output has these columns:

| Column | Meaning |
|--------|---------|
| `ncalls` | Number of times the function was called. Format `a/b` means `b` total calls, `a` non-recursive. |
| `tottime` | Total time spent *inside* this function only (excludes time in sub-calls). |
| `percall` | `tottime / ncalls` — average time per call (self only). |
| `cumtime` | Cumulative time inside this function *including* all sub-calls. |
| `percall` (2nd) | `cumtime / ncalls` — average wall-clock time per call. |

**How to spot bottlenecks:**
- Sort by `cumtime` to find functions that dominate wall-clock time (including their callees).
- Sort by `tottime` to find functions where the CPU actually spends time (the "leaf" work).
- High `ncalls` with moderate `percall` = death by a thousand cuts (optimize the call or reduce frequency).
- High `tottime` with low `ncalls` = heavy single operations (optimize the algorithm).

---

## The 5 Identified Bottlenecks

### Bottleneck 1: `LoRAMoELayer.forward()` — `src/layers/lora_moe.py`

**Call frequency:** Every training step × every MoE layer (6 layers × thousands of steps)

**Why it's slow:**
- Python `for expert_idx in active_experts.tolist()` loop — launches separate GPU kernels per expert
- Each iteration: boolean masking `(idx_flat == expert_idx).any(dim=1)`, expert forward pass, weighted accumulation
- `active_experts.tolist()` forces a GPU→CPU sync (blocks until GPU finishes)
- The `get_cached_metrics()` method stacks LoRA weight matrices and computes norms — another GPU sync per call

**Profiling signature to look for:**
```
ncalls  tottime  cumtime  filename:lineno(function)
  120    0.xxx    X.xxx   lora_moe.py:...(forward)        ← high cumtime
  960    0.xxx    X.xxx   gpt_neo_lora.py:...(forward)    ← ncalls = steps × experts
  960    0.xxx    X.xxx   lora.py:...(forward)            ← SharedLoRALayer inner loop
```

**Optimizations:**
1. **Batch expert dispatch** — instead of looping over experts, pad and batch all expert inputs into a single tensor, run all experts as a batched matmul, then scatter results back. Eliminates N kernel launches per step.
2. **Avoid `.tolist()` GPU sync** — keep `active_experts` on GPU and use `torch.where` / scatter operations instead of Python iteration.
3. **Fuse the weighted accumulation** — replace `combined[token_ids] += expert_out * expert_w` with a single `scatter_add_` on a pre-allocated output tensor.

```python
# BEFORE (current code — Python loop, N kernel launches):
for expert_idx in active_experts.tolist():       # GPU→CPU sync!
    expert = self.expert_pool[expert_idx]
    token_ids = (idx_flat == expert_idx).any(dim=1)
    expert_out = expert(x_flat[token_ids])
    combined[token_ids] += expert_out * expert_w

# AFTER (batched — 1 kernel launch):
# Group tokens by expert, pad to max group size, run batched forward
groups = [x_flat[idx_flat[:, 0] == e] for e in range(num_experts)]  # still a loop but can be vectorized
# Or use torch.scatter with pre-sorted indices
```

---

### Bottleneck 2: `StressCorrectedRouter.forward()` + `_update_welford()` — `src/routers/stress_corrected.py`

**Call frequency:** Every training step × every MoE layer

**Why it's slow:**
- `x_norm @ W_norm.T` computes a full `[B*S, E]` cosine similarity matrix (e.g., 2048×8 = 16K elements)
- `_update_welford()` does a second `x_flat @ W_norm.T` — **redundant** with the cosine sim already computed in `forward()`
- Gumbel noise generation: `torch.empty_like().uniform_()` + two `torch.log()` calls
- `_calibrate_lambda` appends to a Python list (`_pending_cos_sims`) — unbounded memory growth pre-calibration

**Profiling signature:**
```
ncalls  tottime  cumtime
  50     0.xxx    X.xxx   stress_corrected.py:...(forward)
  50     0.xxx    X.xxx   stress_corrected.py:...(_update_welford)    ← ~same as forward
  50     0.xxx    X.xxx   stress_corrected.py:...(_update_load_and_welford)
```

**Optimizations:**
1. **Eliminate redundant matmul in `_update_welford`** — `forward()` already computes `cos_sim = x_norm @ W_norm.T`. The Welford update recomputes `alignments = x_flat @ W_norm.T` with the same inputs. Pass `cos_sim` directly:

```python
# BEFORE (_update_welford, line 233):
alignments = x_flat @ W_norm.T          # REDUNDANT — already computed in forward()
distances = 1.0 - alignments

# AFTER: pass cos_sim from forward() and reshape
def _update_welford(self, cos_sim_flat, topk_idx):
    distances = 1.0 - cos_sim_flat       # reuse, no extra matmul
```

2. **Pre-allocate Gumbel noise** — reuse a noise buffer instead of allocating `torch.empty_like` every call.
3. **Cap `_pending_cos_sims` list** — add a max length to prevent unbounded memory growth.

---

### Bottleneck 3: `SharedLoRALayer.forward()` — `src/experts/lora.py`

**Call frequency:** `num_experts × num_moe_layers × 2 (c_fc + c_proj) × steps` = potentially 96 calls/step

**Why it's slow:**
- **dtype casting on every call**: `w.to(x.dtype)` and `x.to(self.lora_A.weight.dtype)` — two dtype conversions per forward
- Two separate linear operations: `F.linear(x, w, b)` for base + `lora_B(lora_A(x))` for adapter — could be fused
- Dropout identity check: `nn.Identity()` still has Python overhead per call

**Profiling signature:**
```
ncalls  tottime  cumtime
 1920    0.xxx    X.xxx   lora.py:...(forward)           ← SharedLoRALayer
 1920    0.xxx    X.xxx   functional.py:...(linear)      ← F.linear calls
 1920    0.xxx    X.xxx   <built-in>:...(to)             ← dtype casting
```

**Optimizations:**
1. **Ensure consistent dtypes at init** — cast `shared_weight` to the compute dtype once during `load_from_mlp()` or `consolidate_shared_weights()`, not on every forward call:

```python
# BEFORE (every forward call):
w = base_weight if base_weight is not None else self.shared_weight
if w.dtype != x.dtype:
    w = w.to(x.dtype)    # casting 3072×768 tensor EVERY call

# AFTER (cast once at init):
# In consolidate_shared_weights() or load_from_mlp():
self.shared_weight = self.shared_weight.to(COMPUTE_DTYPE)
# forward() can skip the dtype check entirely
```

2. **Fuse base + LoRA into single matmul** (inference only) — for eval, precompute `W_merged = W_base + scaling * B @ A` and do one `F.linear`.
3. **Remove Identity wrapper** — when dropout=0, skip the module call entirely with a simple `if`:

```python
# BEFORE:
self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
lora_out = self.lora_B(self.lora_dropout(self.lora_A(x)))

# AFTER:
h = self.lora_A(x)
if self.lora_dropout is not None:
    h = self.lora_dropout(h)
lora_out = self.lora_B(h)
```

---

### Bottleneck 4: `RouterMetricsTracker.compute_all_metrics()` — `src/metrics/router_metrics.py`

**Call frequency:** Every `log_interval` steps (default 10) × every MoE layer

**Why it's slow:**
- Chains 6+ metric computations, each doing independent tensor reductions
- `_compute_usage()` is called once and passed to sub-metrics (good), but `get_custom_metrics()` in `StressCorrectedRouter` recomputes `scatter_add_` for hard assignment counts — **redundant** with usage already computed
- `.item()` calls on every metric force GPU→CPU syncs (one per metric × per layer)
- `compute_gini_coefficient` sorts the usage tensor — O(E log E) per call

**Profiling signature:**
```
ncalls  tottime  cumtime
  100    0.xxx    X.xxx   router_metrics.py:...(compute_all_metrics)
  100    0.xxx    X.xxx   stress_corrected.py:...(get_custom_metrics)  ← redundant scatter
  600+   0.xxx    X.xxx   <built-in method item>                       ← GPU syncs
```

**Optimizations:**
1. **Batch `.item()` calls** — collect all scalar tensors, `torch.stack()` them, call `.tolist()` once:

```python
# BEFORE (6+ GPU→CPU syncs):
metrics["ema_load_mean"] = self.ema_load.mean().item()
metrics["ema_load_max"] = self.ema_load.max().item()
metrics["ema_load_std"] = self.ema_load.std().item()

# AFTER (1 GPU→CPU sync):
stats = torch.stack([self.ema_load.mean(), self.ema_load.max(), self.ema_load.std()])
mean_val, max_val, std_val = stats.tolist()
```

2. **Pass precomputed usage to `get_custom_metrics()`** — avoid the redundant `scatter_add_` in `StressCorrectedRouter.get_custom_metrics()`.
3. **Reduce metric frequency** — compute expensive metrics (gini, specialization) less often than cheap ones (loss, lr).

---

### Bottleneck 5: `ShardDataset.__getitem__()` + `write_split_to_disk()` — `scripts/train.py` + `infra/data_ingestion/processing_script.py`

**Call frequency:**
- `__getitem__`: `batch_size × steps × num_workers` (thousands of calls per epoch)
- `write_split_to_disk`: once per data preparation, but processes millions of rows

**Why `__getitem__` can be slow:**
- `bisect.bisect_right` + while loop for cross-shard boundary reads
- `np.astype(np.int64)` copies data on every read (memmap returns uint16, needs int64 for torch)
- `torch.from_numpy` creates a new tensor each call

**Why `write_split_to_disk` is slow:**
- Row-by-row `json.dumps()` + `fh.write()` — one syscall per row for JSONL format
- `text.strip()` called twice (once for check, once could be cached)

**Optimizations for `__getitem__`:**
1. **Pre-cast shard data** — if memory allows, load shards as int64 at init instead of casting per-read
2. **Pre-allocate the output array** — avoid `np.empty` allocation per call:

```python
# BEFORE:
tokens = np.empty(self.seq_len + 1, dtype=np.int64)

# AFTER (thread-local buffer):
# Pre-allocate in __init__ per worker
self._token_buf = np.empty(self.seq_len + 1, dtype=np.int64)
```

**Optimizations for `write_split_to_disk`:**
1. **Buffer writes with StringIO** — accumulate in memory, write once:

```python
# BEFORE: one write() syscall per row
for row in split_data:
    fh.write(json.dumps({"text": text}) + "\n")

# AFTER: batch into buffer, single write
buf = io.StringIO()
for row in split_data:
    buf.write(json.dumps({"text": text}) + "\n")
fh.write(buf.getvalue())
```

2. **Use `orjson`** for 3-5× faster JSON serialization:
```python
import orjson
fh.write(orjson.dumps({"text": text}).decode() + "\n")
```

---

## Quick Reference: Profiling Commands

```python
# Profile any function with the decorator
from profiling.profile_tmoe import profile_function

@profile_function
def my_function():
    ...

# Profile a code block inline
import cProfile, pstats, io
pr = cProfile.Profile()
pr.enable()
# ... code to profile ...
pr.disable()
stats = pstats.Stats(pr)
stats.sort_stats("cumulative")
stats.print_stats(20)

# Save and reload profiles
stats.dump_stats("my_profile.prof")
loaded = pstats.Stats("my_profile.prof")
loaded.sort_stats("tottime").print_stats(30)

# Profile from command line
python -m cProfile -s cumulative scripts/train.py --config experiments/smoketest.yaml
```
