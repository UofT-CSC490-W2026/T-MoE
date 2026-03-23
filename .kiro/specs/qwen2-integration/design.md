# Design Document: Qwen2-1.5B Integration

## Overview

This design adds `Qwen/Qwen2-1.5B` as a second supported backbone in the T-MoE framework alongside the existing GPT-Neo family. The integration follows the same pattern as GPT-Neo: a backbone class registered in `ModelRegistry`, a LoRA expert class registered in `ExpertRegistry`, and a YAML experiment config. All routing, training, checkpointing, and distributed infrastructure is reused without modification.

A critical prerequisite is fixing the shard format: Qwen2's vocabulary (151,936 tokens) overflows `uint16` (max 65,535). A versioned 10-byte shard header is introduced to support `uint32` token storage while preserving full backward compatibility with existing GPT-Neo shards.

### Key Architectural Differences: Qwen2 vs GPT-Neo

| Property | GPT-Neo 125M | Qwen2-1.5B |
|---|---|---|
| Hidden dim | 768 | 1536 |
| Layers | 12 | 28 |
| Vocab size | 50,257 | 151,936 |
| MLP type | 2-layer GELU (c_fc → c_proj) | SwiGLU (gate_proj, up_proj, down_proj) |
| Intermediate dim | 3,072 (4×768) | 8,960 (not 4×1536=6,144) |
| Linear layer type | HuggingFace `Conv1D` (transposed) | `nn.Linear` (standard) |
| MLP bias | Yes | No (`bias=False`) |
| HF model path | `transformer.h[i].mlp` | `model.layers[i].mlp` |

---

## Architecture

The integration is purely additive at the model/expert layer. The existing pipeline is unchanged:

```
YAML config
    │
    ▼
build_model() [scripts/train.py]
    │  model_lookup("qwen2-1.5b") → ModelRegistry["qwen2"] → Qwen2Backbone
    │  ExpertType("qwen2_lora")   → ExpertRegistry["qwen2_lora"] → Qwen2LoRAMLP
    ▼
Qwen2Backbone
    │  backbone = AutoModelForCausalLM("Qwen/Qwen2-1.5B")  [frozen]
    │  get_mlp_at(idx) → backbone.model.layers[idx].mlp
    ▼
LoRAMoELayer.from_pretrained_mlp(mlp, router, lora_cfg, expert_type=QWEN2_LORA)
    │  ExpertPool([Qwen2LoRAMLP × N])
    │  each expert: gate_proj, up_proj, down_proj (SharedLoRALayer, no bias)
    ▼
inject_moe_layers() → backbone.model.layers[idx].mlp = LoRAMoELayer
    ▼
DDP / training loop [unchanged]
```

### Shard Format Versioning

```
Legacy (8 bytes):   [uint64 token_count][uint16 tokens...]
New    (10 bytes):  [uint64 token_count][uint16 dtype_flag][tokens...]
                                         dtype_flag=0 → uint16 tokens
                                         dtype_flag=1 → uint32 tokens

ShardDataset detection:
  file_size == 8 + token_count * 2  → legacy uint16 (backward compat)
  file_size == 10 + token_count * 2 → new header, dtype_flag=0, uint16
  file_size == 10 + token_count * 4 → new header, dtype_flag=1, uint32
```

---

## Components and Interfaces

### New Files

**`src/models/qwen2.py`** — `Qwen2Backbone`

```python
@ModelRegistry.register("qwen2")
class Qwen2Backbone(BaseModelBackbone):
    VARIANTS = {
        "1.5b": {
            "hf_name": "Qwen/Qwen2-1.5B",
            "hidden_dim": 1536,
            "num_layers": 28,
            "intermediate_dim": 8960,
            "tokenizer_vocab_size": 151936,
        }
    }

    def load_pretrained(self) -> None: ...
    def get_mlp_at(self, idx: int) -> nn.Module: ...
    def inject_moe_layers(self, moe_layers: Dict[int, nn.Module]) -> None: ...
    def forward(self, input_ids, attention_mask=None, labels=None,
                return_metrics=False, **kwargs) -> Tuple[Tensor, Optional[Tensor], Optional[Dict]]: ...
```

`model_lookup("qwen2-1.5b")` resolves automatically via the existing VARIANTS scan in `model_lookup()` — no changes to that function.

**`src/experts/qwen2_lora.py`** — `Qwen2LoRAMLP`

```python
@ExpertRegistry.register("qwen2_lora")
class Qwen2LoRAMLP(LoRAMLPExpert):
    def __init__(self, config: LoRAConfig): ...
    def load_from_mlp(self, mlp: nn.Module) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    def get_lora_layer_names(self) -> list[str]: ...
```

The `forward` computes: `down_proj(silu(gate_proj(x)) * up_proj(x))`.  
Each projection is a `SharedLoRALayer` with `shared_bias=None`.

**`experiments/qwen2_1.5b_stress_v1-wikitext.yaml`** — experiment config for Qwen2-1.5B with SPAR router on WikiText-103.

### Modified Files

**`src/project_types.py`** — add `QWEN2_LORA = "qwen2_lora"` to `ExpertType` and `QWEN2 = "qwen2"` to `ModelType`.

**`src/models/base.py`** — add `get_mlp_at(self, idx: int) -> nn.Module` raising `NotImplementedError`.

**`src/models/gpt_neo.py`** — implement `get_mlp_at(idx)` returning `self.backbone.transformer.h[idx].mlp`.

**`src/models/__init__.py`** — add `import src.models.qwen2` to trigger registration.

**`src/experts/__init__.py`** — add `import src.experts.qwen2_lora` to trigger registration.

**`src/experts/gpt_neo_lora.py`** — add `get_lora_layer_names()` returning `["c_fc", "c_proj"]`.

**`src/experts/pool.py`** — generalize `consolidate_shared_weights()` to use `get_lora_layer_names()` protocol instead of hardcoded `c_fc`/`c_proj` attribute names.

**`src/layers/lora_moe.py`** — two minor patches:
- `get_cached_metrics()`: extend projection attribute loop to include `gate_proj`, `up_proj`, `down_proj`.
- `_init_shared_base_lora()`: check `down_proj` in addition to `c_proj` for output projection detection.

**`scripts/train.py`** — two changes:
- `build_model()`: replace `model.backbone.transformer.h[actual_idx].mlp` with `model.get_mlp_at(actual_idx)`.
- `build_model()`: pass `intermediate_dim=model_info.get("intermediate_dim")` to `LoRAConfig`.

**`scripts/prepare_data.py`** — write 10-byte versioned header; use `uint32` when `vocab_size > 65535`.

**`scripts/train.py` (`ShardDataset`)** — read 10-byte header; detect legacy vs versioned; memmap with correct dtype.

---

## Data Models

### Versioned Shard Header

```
Offset  Size   Type     Field
0       8      uint64   token_count  (number of tokens in this shard)
8       2      uint16   dtype_flag   (0 = uint16 tokens, 1 = uint32 tokens)
10      varies uint16[] OR uint32[]  token payload
```

Legacy detection: if `file_size - 8 == token_count * 2`, treat as legacy uint16 (no dtype_flag byte). This is unambiguous because a valid new-format file with `dtype_flag=0` would have `file_size = 10 + token_count * 2`, which differs from `8 + token_count * 2` by exactly 2 bytes.

### LoRAConfig Extension

`LoRAConfig` already has `intermediate_dim: Optional[int]` with `__post_init__` defaulting to `4 × hidden_dim`. No schema change needed — `build_model()` simply passes the value from VARIANTS.

### Qwen2Backbone VARIANTS Entry

```python
"1.5b": {
    "hf_name": "Qwen/Qwen2-1.5B",
    "hidden_dim": 1536,
    "num_layers": 28,
    "intermediate_dim": 8960,   # NOT 4×1536=6144
    "tokenizer_vocab_size": 151936,
    "description": "Qwen2 1.5B parameters",
}
```

The `model_lookup()` function builds the canonical key as `{model_type.replace('_','-')}-{variant}` → `"qwen2-1.5b"`, matching the YAML `model.model_key`.

### ExpertPool `consolidate_shared_weights` Protocol

After generalization, the method uses a duck-typed protocol:

```
if expert_0 has get_lora_layer_names() → use returned list
elif expert_0 has c_fc attribute        → use ["c_fc", "c_proj"]  (legacy fallback)
else                                    → return silently
```

For each layer name in the list, alias `shared_weight` and `shared_bias` buffers of experts `1..N-1` to expert `0`'s buffers.

---

## Correctness Properties


*A property is a characteristic or behavior that should hold true across all valid executions of a system — essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Shard dtype_flag matches vocabulary size

*For any* tokenizer vocabulary size, `prepare_data.py` must write `dtype_flag=1` and `uint32` tokens when `vocab_size > 65535`, and `dtype_flag=0` and `uint16` tokens when `vocab_size <= 65535`. Generating random token sequences with IDs drawn from the full vocabulary range and writing them to a shard, then reading the raw header bytes, must confirm the correct flag and token byte width.

**Validates: Requirements 1.2, 1.3**

### Property 2: Shard write/read round-trip

*For any* sequence of valid token IDs (whether in the uint16 or uint32 range), writing the sequence to a shard with `prepare_data.py`'s flush logic and reading it back with `ShardDataset` must produce a token array identical to the original sequence.

**Validates: Requirements 1.8**

### Property 3: ShardDataset always returns int64 tensors

*For any* shard file (legacy uint16, new uint16, or new uint32), the tensors returned by `ShardDataset.__getitem__` must have dtype `torch.int64`, regardless of the on-disk storage format.

**Validates: Requirements 1.7**

### Property 4: Qwen2LoRAMLP output shape invariant

*For any* number of tokens `T > 0` and any initialized `Qwen2LoRAMLP`, calling `forward(x)` with `x` of shape `[T, hidden_dim]` must return a tensor of shape `[T, hidden_dim]`. The output shape must equal the input shape for all valid `T`.

**Validates: Requirements 2.5, 2.7**

### Property 5: Qwen2LoRAMLP projection shapes match LoRAConfig

*For any* `LoRAConfig` with `hidden_dim=1536` and `intermediate_dim=8960`, a `Qwen2LoRAMLP` initialized from a matching MLP must have `gate_proj` and `up_proj` with `shared_weight` shape `[8960, 1536]` and `down_proj` with `shared_weight` shape `[1536, 8960]`. More generally, for any valid `(hidden_dim, intermediate_dim)` pair, the projection shapes must match the config values.

**Validates: Requirements 5.3**

### Property 6: consolidate_shared_weights aliases Qwen2 buffers

*For any* `ExpertPool` containing `N >= 2` `Qwen2LoRAMLP` experts after `load_from_mlp` and `model.to(device)`, calling `consolidate_shared_weights()` must result in `gate_proj.shared_weight`, `up_proj.shared_weight`, and `down_proj.shared_weight` in experts `1..N-1` having the same `data_ptr()` as the corresponding buffers in expert `0`.

**Validates: Requirements 6.1, 6.4**

### Property 7: consolidate_shared_weights aliases GPT-Neo buffers

*For any* `ExpertPool` containing `N >= 2` `GPTNeoLoRAMLP` experts after `load_from_mlp` and `model.to(device)`, calling `consolidate_shared_weights()` must result in `c_fc.shared_weight` and `c_proj.shared_weight` in experts `1..N-1` having the same `data_ptr()` as the corresponding buffers in expert `0`. This verifies backward compatibility of the generalized consolidation logic.

**Validates: Requirements 6.2, 6.3, 10.3**

---

## Error Handling

### Shard Format Errors

- `ShardDataset` raises `FileNotFoundError` if no shards match the glob pattern (existing behavior, unchanged).
- If a shard file has an unrecognized `dtype_flag` value (not 0 or 1), `ShardDataset` raises `ValueError` with a descriptive message including the file path and flag value.
- Legacy shard detection is based on file size arithmetic: if `(file_size - 8) % 2 == 0` and `file_size - 8 == token_count * 2`, treat as legacy. If the file size is inconsistent with either format, raise `ValueError`.

### Model Construction Errors

- `model_lookup()` raises `ValueError` with the list of known keys if the requested `model_key` is not found (existing behavior, unchanged).
- `Qwen2Backbone.__init__` raises `ValueError` for unknown variants, mirroring `GPTNeoBackbone`.
- `Qwen2LoRAMLP.load_from_mlp` raises `ValueError` if the MLP module is missing `gate_proj`, `up_proj`, or `down_proj` attributes, with a message listing the found attributes.
- `get_mlp_at` raises `IndexError` if `idx` is out of range for the backbone's layer list.

### Registry Errors

- `ModelRegistry.get` and `ExpertRegistry.get` raise `KeyError` if the requested type is not registered (existing behavior, unchanged).
- `ExpertType("unknown_value")` raises `ValueError` (standard Python enum behavior).

### Consolidation Edge Cases

- `consolidate_shared_weights` on a pool with `num_experts < 2` returns silently (no-op).
- `consolidate_shared_weights` on a pool whose expert `0` has neither `get_lora_layer_names()` nor `c_fc` returns silently without raising.

---

## Testing Strategy

### Dual Testing Approach

Both unit tests and property-based tests are required. Unit tests verify specific examples, structural invariants, and error conditions. Property tests verify universal correctness across randomly generated inputs.

### Unit Tests

New test files:

**`tests/experts/test_qwen2_lora.py`**
- `test_registry_entry`: verify `ExpertRegistry.get("qwen2_lora")` returns `Qwen2LoRAMLP`
- `test_get_lora_layer_names`: verify returns `["gate_proj", "up_proj", "down_proj"]`
- `test_load_from_mlp_attributes`: after `load_from_mlp`, verify `gate_proj`, `up_proj`, `down_proj` exist as `SharedLoRALayer`
- `test_load_from_mlp_no_bias`: verify `shared_bias` is `None` for all three projections
- `test_load_from_mlp_no_transpose`: verify `gate_proj.shared_weight.shape == (intermediate_dim, hidden_dim)` (not transposed)
- `test_forward_shape`: verify output shape `[T, 1536]` for several values of `T`

**`tests/models/test_qwen2_backbone.py`**
- `test_registry_entry`: verify `ModelRegistry.get("qwen2")` returns `Qwen2Backbone`
- `test_model_lookup`: verify `model_lookup("qwen2-1.5b")` returns correct `hidden_dim`, `intermediate_dim`, `model_type`
- `test_variants_fields`: verify VARIANTS `"1.5b"` entry has all required keys
- `test_get_mlp_at`: mock backbone, verify `get_mlp_at(idx)` accesses `model.layers[idx].mlp`
- `test_inject_moe_layers`: mock backbone, verify replacement at correct path
- `test_enum_entries`: verify `ModelType.QWEN2 == "qwen2"` and `ExpertType.QWEN2_LORA == "qwen2_lora"`

**`tests/experts/test_pool_consolidation.py`** (extends existing pool tests)
- `test_consolidate_qwen2_no_op_single_expert`: pool with 1 expert, consolidate is no-op
- `test_consolidate_unknown_expert_no_error`: pool with expert lacking both `c_fc` and `get_lora_layer_names`, consolidate returns silently

**`tests/test_shard_format.py`** (extends existing shard tests)
- `test_legacy_shard_backward_compat`: write legacy 8-byte shard, verify `ShardDataset` reads it correctly
- `test_new_header_uint16`: write new 10-byte shard with `dtype_flag=0`, verify correct read
- `test_new_header_uint32`: write new 10-byte shard with `dtype_flag=1` with token IDs > 65535, verify correct read
- `test_output_dtype_int64`: verify returned tensors are `torch.int64` for all shard types

### Property-Based Tests

Property-based testing library: **Hypothesis** (already a common Python PBT library; add to `requirements.txt` if not present).

Each property test runs a minimum of 100 examples (Hypothesis default `@settings(max_examples=100)`).

**`tests/test_shard_properties.py`**

```python
# Feature: qwen2-integration, Property 1: Shard dtype_flag matches vocabulary size
@given(tokens=st.lists(st.integers(min_value=0, max_value=151935), min_size=1, max_size=1000),
       vocab_size=st.integers(min_value=1, max_value=200000))
@settings(max_examples=100)
def test_shard_dtype_flag_matches_vocab_size(tokens, vocab_size): ...

# Feature: qwen2-integration, Property 2: Shard write/read round-trip
@given(tokens=st.lists(st.integers(min_value=0, max_value=151935), min_size=2, max_size=2000))
@settings(max_examples=100)
def test_shard_round_trip(tokens): ...

# Feature: qwen2-integration, Property 3: ShardDataset always returns int64 tensors
@given(tokens=st.lists(st.integers(min_value=0, max_value=151935), min_size=2, max_size=2000),
       use_uint32=st.booleans())
@settings(max_examples=100)
def test_shard_output_dtype_int64(tokens, use_uint32): ...
```

**`tests/experts/test_qwen2_properties.py`**

```python
# Feature: qwen2-integration, Property 4: Qwen2LoRAMLP output shape invariant
@given(T=st.integers(min_value=1, max_value=64))
@settings(max_examples=100)
def test_qwen2_lora_output_shape(T): ...

# Feature: qwen2-integration, Property 5: Qwen2LoRAMLP projection shapes match LoRAConfig
@given(hidden_dim=st.integers(min_value=64, max_value=2048).filter(lambda x: x % 64 == 0),
       intermediate_dim=st.integers(min_value=64, max_value=8960).filter(lambda x: x % 64 == 0))
@settings(max_examples=100)
def test_qwen2_projection_shapes_match_config(hidden_dim, intermediate_dim): ...
```

**`tests/experts/test_pool_properties.py`**

```python
# Feature: qwen2-integration, Property 6: consolidate_shared_weights aliases Qwen2 buffers
@given(num_experts=st.integers(min_value=2, max_value=8))
@settings(max_examples=100)
def test_consolidate_qwen2_aliases_buffers(num_experts): ...

# Feature: qwen2-integration, Property 7: consolidate_shared_weights aliases GPT-Neo buffers
@given(num_experts=st.integers(min_value=2, max_value=8))
@settings(max_examples=100)
def test_consolidate_gptneo_aliases_buffers(num_experts): ...
```

### Regression

All 55 existing unit tests must pass without modification. Run with:

```
pytest tests/ --run -x
```

The integration is designed to be purely additive — no existing function signatures or behaviors change, only new code paths are added alongside existing ones.
