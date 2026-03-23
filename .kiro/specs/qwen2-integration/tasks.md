# Implementation Plan: Qwen2-1.5B Integration

## Overview

Additive integration of Qwen2-1.5B into T-MoE. Steps follow the prescribed implementation order: enum extensions → abstract interface → GPT-Neo concrete impl → new Qwen2 classes → registry wiring → generalization patches → shard format fix → experiment config.

## Tasks

- [x] 1. Extend enums and abstract interface
  - [x] 1.1 Add `QWEN2_LORA = "qwen2_lora"` to `ExpertType` and `QWEN2 = "qwen2"` to `ModelType` in `src/project_types.py`
    - Purely additive; no existing values change
    - _Requirements: 8.1, 8.2_
  - [x] 1.2 Add `get_mlp_at(self, idx: int) -> nn.Module` to `BaseModelBackbone` in `src/models/base.py` raising `NotImplementedError`
    - _Requirements: 4.1_
  - [x] 1.3 Implement `get_mlp_at(idx)` on `GPTNeoBackbone` in `src/models/gpt_neo.py` returning `self.backbone.transformer.h[idx].mlp`
    - _Requirements: 4.2_
  - [x] 1.4 Add `get_lora_layer_names()` to `GPTNeoLoRAMLP` in `src/experts/gpt_neo_lora.py` returning `["c_fc", "c_proj"]`
    - _Requirements: 6.3_

- [x] 2. Create `Qwen2LoRAMLP` expert
  - [x] 2.1 Create `src/experts/qwen2_lora.py` with `Qwen2LoRAMLP` registered under `"qwen2_lora"`
    - `__init__`: initialize `gate_proj`, `up_proj`, `down_proj` as `None`; store `act_fn = nn.SiLU()`
    - `load_from_mlp`: extract weights from `mlp.gate_proj.weight`, `mlp.up_proj.weight`, `mlp.down_proj.weight` (no transpose — `nn.Linear`); pass `shared_bias=None` to each `SharedLoRALayer`
    - `forward`: compute `down_proj(silu(gate_proj(x)) * up_proj(x))`; return shape `[T, hidden_dim]`
    - `get_lora_layer_names`: return `["gate_proj", "up_proj", "down_proj"]`
    - Raise `ValueError` in `load_from_mlp` if any of the three projections are missing
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_
  - [ ]* 2.2 Write property test for `Qwen2LoRAMLP` output shape invariant (`tests/experts/test_qwen2_properties.py`)
    - **Property 4: Qwen2LoRAMLP output shape invariant**
    - **Validates: Requirements 2.5, 2.7**
    - `@given(T=st.integers(min_value=1, max_value=64))` with a mock MLP of shape `[1536, 8960]`
  - [ ]* 2.3 Write property test for `Qwen2LoRAMLP` projection shapes matching `LoRAConfig` (`tests/experts/test_qwen2_properties.py`)
    - **Property 5: Qwen2LoRAMLP projection shapes match LoRAConfig**
    - **Validates: Requirements 5.3**
    - `@given(hidden_dim=..., intermediate_dim=...)` — verify `gate_proj`/`up_proj` shape `[intermediate_dim, hidden_dim]` and `down_proj` shape `[hidden_dim, intermediate_dim]`
  - [ ]* 2.4 Write unit tests for `Qwen2LoRAMLP` (`tests/experts/test_qwen2_lora.py`)
    - `test_registry_entry`, `test_get_lora_layer_names`, `test_load_from_mlp_attributes`, `test_load_from_mlp_no_bias`, `test_load_from_mlp_no_transpose`, `test_forward_shape`
    - _Requirements: 2.1–2.7_

- [x] 3. Create `Qwen2Backbone` model wrapper
  - [x] 3.1 Create `src/models/qwen2.py` with `Qwen2Backbone` registered under `"qwen2"`
    - `VARIANTS` dict with `"1.5b"` entry: `hf_name`, `hidden_dim=1536`, `num_layers=28`, `intermediate_dim=8960`, `tokenizer_vocab_size=151936`
    - `__init__`: mirror `GPTNeoBackbone.__init__` pattern; call `load_pretrained()` then `freeze_parameters()` if `freeze_backbone`
    - `load_pretrained`: `AutoModelForCausalLM.from_pretrained(self.model_name, dtype=COMPUTE_DTYPE)`; set `self.vocab_size`
    - `get_mlp_at(idx)`: return `self.backbone.model.layers[idx].mlp`; raise `IndexError` if out of range
    - `inject_moe_layers`: replace `self.backbone.model.layers[idx].mlp` for each index; populate `self.moe_layers`
    - `forward`: identical logic to `GPTNeoBackbone.forward` — set grad context, call `self.backbone()`, accumulate `aux_loss`, collect metrics
    - _Requirements: 3.1–3.8, 4.3_
  - [ ]* 3.2 Write unit tests for `Qwen2Backbone` (`tests/models/test_qwen2_backbone.py`)
    - `test_registry_entry`, `test_model_lookup`, `test_variants_fields`, `test_get_mlp_at`, `test_inject_moe_layers`, `test_enum_entries`
    - _Requirements: 3.1–3.5, 8.1, 8.2_

- [x] 4. Wire registries
  - [x] 4.1 Add `from src.models import qwen2  # noqa: F401` to `src/models/__init__.py` and add `Qwen2Backbone` to `__all__`
    - _Requirements: 8.3, 8.5_
  - [x] 4.2 Add `import src.experts.qwen2_lora  # noqa: F401` to `src/experts/__init__.py`
    - _Requirements: 8.4, 8.6_

- [x] 5. Checkpoint — Ensure all tests pass
  - Ensure all 55 existing tests pass and new unit tests pass. Ask the user if questions arise.

- [x] 6. Generalize `consolidate_shared_weights` and patch `lora_moe.py`
  - [x] 6.1 Rewrite `consolidate_shared_weights` in `src/experts/pool.py` to use `get_lora_layer_names()` protocol
    - If `expert_0` has `get_lora_layer_names()` → use returned list; elif has `c_fc` → use `["c_fc", "c_proj"]`; else return silently
    - For each layer name, alias `shared_weight` and `shared_bias` buffers of experts `1..N-1` to expert `0`'s buffers
    - No-op when `num_experts < 2`
    - _Requirements: 6.1, 6.2, 6.4, 6.5_
  - [ ]* 6.2 Write property test for `consolidate_shared_weights` aliasing Qwen2 buffers (`tests/experts/test_pool_properties.py`)
    - **Property 6: consolidate_shared_weights aliases Qwen2 buffers**
    - **Validates: Requirements 6.1, 6.4**
    - `@given(num_experts=st.integers(min_value=2, max_value=8))` — verify `data_ptr()` equality for all three projections
  - [ ]* 6.3 Write property test for `consolidate_shared_weights` aliasing GPT-Neo buffers (`tests/experts/test_pool_properties.py`)
    - **Property 7: consolidate_shared_weights aliases GPT-Neo buffers**
    - **Validates: Requirements 6.2, 6.3, 10.3**
    - `@given(num_experts=st.integers(min_value=2, max_value=8))` — verify `data_ptr()` equality for `c_fc` and `c_proj`
  - [x] 6.4 Patch `get_cached_metrics` in `src/layers/lora_moe.py` to iterate over `("c_fc", "c_proj", "gate_proj", "up_proj", "down_proj")`
    - Existing `getattr(..., None)` guard means no logic change needed beyond extending the tuple
    - _Requirements: 7.1_
  - [x] 6.5 Patch `_init_shared_base_lora` in `src/layers/lora_moe.py` to check `down_proj` in addition to `c_proj` for output projection detection
    - If neither `c_proj` nor `down_proj` exists on expert `0`, skip initialization silently
    - _Requirements: 7.2, 7.3_

- [x] 7. Update `scripts/train.py` — `build_model` and `LoRAConfig`
  - [x] 7.1 Replace `model.backbone.transformer.h[actual_idx].mlp` with `model.get_mlp_at(actual_idx)` in `build_model`
    - _Requirements: 4.4_
  - [x] 7.2 Pass `intermediate_dim=model_info.get("intermediate_dim")` to `LoRAConfig` in `build_model`
    - When key is absent (GPT-Neo), passes `None` → `LoRAConfig.__post_init__` defaults to `4 × hidden_dim`
    - _Requirements: 5.1, 5.2_

- [x] 8. Fix shard format — uint32 write path (`scripts/prepare_data.py`)
  - [x] 8.1 Update `flush_shard` in `scripts/prepare_data.py` to write a 10-byte versioned header: `struct.pack("<QH", count, dtype_flag)` where `dtype_flag=1` when `vocab_size > 65535`, else `0`
    - Determine `vocab_size` from the tokenizer loaded in `tokenize_and_pack`; pass it into `flush_shard`
    - When `dtype_flag=1`, cast token buffer to `np.uint32` before writing; when `0`, keep `np.uint16`
    - Update `_iter_token_arrays` to yield `np.uint32` arrays when `vocab_size > 65535`
    - _Requirements: 1.1, 1.2, 1.3_
  - [ ]* 8.2 Write property test for shard dtype_flag matching vocabulary size (`tests/test_shard_properties.py`)
    - **Property 1: Shard dtype_flag matches vocabulary size**
    - **Validates: Requirements 1.2, 1.3**
    - `@given(tokens=st.lists(...), vocab_size=st.integers(min_value=1, max_value=200000))` — read raw header bytes and assert correct flag and token byte width

- [x] 9. Fix shard format — uint32 read path (`scripts/train.py` `ShardDataset`)
  - [x] 9.1 Update `ShardDataset.__init__` to detect legacy vs versioned shards and set per-shard dtype
    - Read 10 bytes; if `file_size - 8 == token_count * 2` → legacy uint16 (offset=8); else read `dtype_flag` from bytes 8–9; `dtype_flag=0` → uint16 offset=10; `dtype_flag=1` → uint32 offset=10
    - Raise `ValueError` for unrecognized `dtype_flag` values
    - Store per-shard `(dtype, offset)` tuples; update `self.mmaps` construction accordingly
    - `__getitem__` casts chunks to `np.int64` before returning (already done; verify it applies to uint32 path)
    - _Requirements: 1.4, 1.5, 1.6, 1.7, 10.1_
  - [ ]* 9.2 Write property test for shard write/read round-trip (`tests/test_shard_properties.py`)
    - **Property 2: Shard write/read round-trip**
    - **Validates: Requirements 1.8**
    - `@given(tokens=st.lists(st.integers(min_value=0, max_value=151935), min_size=2, max_size=2000))` — write then read back and assert equality
  - [ ]* 9.3 Write property test for `ShardDataset` always returning int64 tensors (`tests/test_shard_properties.py`)
    - **Property 3: ShardDataset always returns int64 tensors**
    - **Validates: Requirements 1.7**
    - `@given(tokens=..., use_uint32=st.booleans())` — verify `tensor.dtype == torch.int64` for both legacy and versioned shards

- [x] 10. Create experiment YAML
  - [x] 10.1 Create `experiments/qwen2_1.5b_stress_v1-wikitext.yaml` with `model_key: qwen2-1.5b`, `expert.type: qwen2_lora`, 6 MoE layer indices from the 28 Qwen2 layers, `router.type: stress_corrected` with `num_experts: 8` and `top_k: 2`, `dataset.dataset_key: wikitext-103`, `max_seq_len: 512`, `distributed.strategy: ddp`
    - _Requirements: 9.1–9.6_

- [x] 11. Final checkpoint — Ensure all tests pass
  - Run `pytest tests/ -x` and confirm all 55 existing tests plus all new tests pass. Ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Property tests use Hypothesis; add to `requirements.txt` if not already present
- `src/training/fsdp_utils.py` dynamic wrap target (step 14 in the integration plan) is deferred until `strategy: fsdp` is needed and is intentionally excluded from this task list
- The `make_base_trainable()` path in `pool.py` is GPT-Neo specific and requires no changes — the `hasattr(e0, "c_fc")` guard already handles the Qwen2 case silently
