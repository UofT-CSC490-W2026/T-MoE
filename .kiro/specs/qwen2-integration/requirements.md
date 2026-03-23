# Requirements Document

## Introduction

This feature adds Qwen2-1.5B (`Qwen/Qwen2-1.5B`) as a supported backbone in the T-MoE codebase, which currently supports GPT-Neo models only. Qwen2-1.5B uses a SwiGLU MLP (gate_proj, up_proj, down_proj with no bias), 1536-dimensional hidden states, 28 transformer layers, and a vocabulary of 151,936 tokens. All existing infrastructure — the SPAR router, LoRAMoELayer dispatcher, DDP training loop, and YAML experiment pipeline — must be reused without modification. Only model/expert-specific wrappers and a small number of coupling points in the training scripts need to change.

A critical blocker must be resolved first: the existing shard format stores token IDs as `uint16`, which overflows for Qwen2's vocabulary (151,936 > 65,535). A versioned shard header with a `uint32` token path must be introduced before any Qwen2 data can be tokenized.

## Glossary

- **T-MoE**: The Tokenized Mixture-of-Experts training framework this codebase implements.
- **Backbone**: A pre-trained transformer model whose weights are frozen; MoE layers are injected into it.
- **Qwen2Backbone**: The new backbone class wrapping `Qwen/Qwen2-1.5B`.
- **Qwen2LoRAMLP**: The new LoRA expert class implementing the Qwen2 SwiGLU MLP interface.
- **GPTNeoBackbone**: The existing backbone class wrapping GPT-Neo models.
- **GPTNeoLoRAMLP**: The existing LoRA expert class for GPT-Neo.
- **LoRAMoELayer**: The existing drop-in MLP replacement that routes tokens to LoRA experts.
- **ExpertPool**: The container that holds N expert instances and manages shared frozen weights.
- **SPAR Router**: The Stress-corrected Prototype-Anchored Router used for load-balanced expert routing.
- **ShardDataset**: The PyTorch Dataset that reads packed binary token shards from disk.
- **Shard**: A `.bin` file containing a packed sequence of token IDs preceded by a header.
- **Versioned Shard Header**: The new 10-byte shard header: `[uint64 token_count][uint16 dtype_flag]`.
- **dtype_flag**: A 2-byte field in the shard header: `0` = uint16 tokens (legacy), `1` = uint32 tokens.
- **ModelRegistry**: The registry that maps model type strings to backbone classes.
- **ExpertRegistry**: The registry that maps expert type strings to expert classes.
- **model_key**: A string of the form `{model_type}-{variant}` (e.g., `qwen2-1.5b`) used to look up model configuration.
- **intermediate_dim**: The hidden dimension of the MLP's intermediate projection (8,960 for Qwen2-1.5B; defaults to 4 × hidden_dim for GPT-Neo).
- **SwiGLU**: The gated activation used by Qwen2: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
- **Conv1D**: HuggingFace's transposed linear layer used in GPT-Neo; Qwen2 uses standard `nn.Linear` instead.
- **get_mlp_at(idx)**: A backbone method that returns the MLP module at transformer layer index `idx`.
- **get_lora_layer_names()**: An expert method that returns the list of LoRA projection attribute names (e.g., `["gate_proj", "up_proj", "down_proj"]`).
- **consolidate_shared_weights()**: An ExpertPool method that aliases frozen weight buffers across experts to save GPU memory.
- **DDP**: PyTorch DistributedDataParallel, the multi-GPU training strategy used.

---

## Requirements

### Requirement 1: Versioned Shard Header for uint32 Token Support

**User Story:** As a researcher, I want to tokenize datasets for models with vocabularies larger than 65,535 tokens, so that I can prepare training data for Qwen2 without token ID overflow.

#### Acceptance Criteria

1. THE `prepare_data.py` Script SHALL write a 10-byte versioned shard header consisting of a `uint64` token count followed by a `uint16` dtype_flag, replacing the previous 8-byte header.
2. WHEN the tokenizer vocabulary size is greater than 65,535, THE `prepare_data.py` Script SHALL write token IDs as `uint32` and set `dtype_flag=1` in the shard header.
3. WHEN the tokenizer vocabulary size is 65,535 or fewer, THE `prepare_data.py` Script SHALL write token IDs as `uint16` and set `dtype_flag=0` in the shard header.
4. WHEN `ShardDataset` reads a shard with `dtype_flag=0`, THE `ShardDataset` SHALL memory-map the token payload as `np.uint16` with an offset of 10 bytes.
5. WHEN `ShardDataset` reads a shard with `dtype_flag=1`, THE `ShardDataset` SHALL memory-map the token payload as `np.uint32` with an offset of 10 bytes.
6. WHEN `ShardDataset` reads a shard written with the legacy 8-byte header (no dtype_flag byte), THE `ShardDataset` SHALL treat the token payload as `np.uint16` to preserve backward compatibility with existing GPT-Neo shards.
7. THE `ShardDataset` SHALL convert all token arrays to `np.int64` before returning them as PyTorch tensors, regardless of the on-disk dtype.
8. FOR ALL valid token sequences written by `prepare_data.py`, reading the shard back with `ShardDataset` SHALL produce token ID arrays identical to the original tokenizer output (round-trip property).

---

### Requirement 2: Qwen2LoRAMLP Expert

**User Story:** As a researcher, I want a LoRA expert that wraps the Qwen2 SwiGLU MLP, so that the existing LoRAMoELayer dispatcher can route tokens to Qwen2 experts without modification.

#### Acceptance Criteria

1. THE `Qwen2LoRAMLP` SHALL be registered in the `ExpertRegistry` under the key `"qwen2_lora"`.
2. WHEN `load_from_mlp` is called with a Qwen2 MLP module, THE `Qwen2LoRAMLP` SHALL initialize three `SharedLoRALayer` instances named `gate_proj`, `up_proj`, and `down_proj` using the corresponding `nn.Linear` weight tensors from the MLP.
3. WHEN `load_from_mlp` is called with a Qwen2 MLP module, THE `Qwen2LoRAMLP` SHALL pass `None` as `shared_bias` to each `SharedLoRALayer` because Qwen2 MLP projections have `bias=False`.
4. WHEN `load_from_mlp` is called with a Qwen2 MLP module, THE `Qwen2LoRAMLP` SHALL extract weights directly from `mlp.gate_proj.weight`, `mlp.up_proj.weight`, and `mlp.down_proj.weight` without transposing, because Qwen2 uses `nn.Linear` (not HuggingFace `Conv1D`).
5. WHEN `forward` is called with an input tensor of shape `[T, hidden_dim]`, THE `Qwen2LoRAMLP` SHALL return an output tensor of shape `[T, hidden_dim]` computed as `down_proj(silu(gate_proj(x)) * up_proj(x))`.
6. THE `Qwen2LoRAMLP` SHALL implement `get_lora_layer_names()` returning `["gate_proj", "up_proj", "down_proj"]`.
7. FOR ALL input tensors `x` of shape `[T, 1536]`, the output of `Qwen2LoRAMLP.forward(x)` SHALL have shape `[T, 1536]` (shape invariant).

---

### Requirement 3: Qwen2Backbone Model Wrapper

**User Story:** As a researcher, I want a backbone class for Qwen2-1.5B that integrates with the ModelRegistry and existing training loop, so that I can train T-MoE with Qwen2 using the same YAML pipeline as GPT-Neo.

#### Acceptance Criteria

1. THE `Qwen2Backbone` SHALL be registered in the `ModelRegistry` under the key `"qwen2"`.
2. THE `Qwen2Backbone` SHALL define a `VARIANTS` dict with a `"1.5b"` entry containing: `hf_name="Qwen/Qwen2-1.5B"`, `hidden_dim=1536`, `num_layers=28`, `intermediate_dim=8960`, `tokenizer_vocab_size=151936`.
3. WHEN `model_lookup("qwen2-1.5b")` is called, THE `ModelRegistry` SHALL return a dict containing `hidden_dim=1536`, `intermediate_dim=8960`, and `model_type="qwen2"`.
4. WHEN `load_pretrained` is called, THE `Qwen2Backbone` SHALL load `Qwen/Qwen2-1.5B` weights from HuggingFace using `AutoModelForCausalLM.from_pretrained` with the configured compute dtype.
5. WHEN `get_mlp_at(idx)` is called with a valid layer index, THE `Qwen2Backbone` SHALL return `self.backbone.model.layers[idx].mlp`.
6. WHEN `inject_moe_layers` is called with a dict of `{layer_index: LoRAMoELayer}`, THE `Qwen2Backbone` SHALL replace `self.backbone.model.layers[idx].mlp` with the corresponding `LoRAMoELayer` for each index.
7. WHEN `forward` is called with `input_ids` and optional `labels`, THE `Qwen2Backbone` SHALL return `(logits, loss, metrics)` with the same signature as `GPTNeoBackbone.forward`.
8. WHEN `forward` is called and `self.moe_layers` is non-empty, THE `Qwen2Backbone` SHALL accumulate auxiliary router losses and add them to the cross-entropy loss.

---

### Requirement 4: Abstract `get_mlp_at` on BaseModelBackbone

**User Story:** As a developer, I want the training script to retrieve MLP modules through a backbone-agnostic interface, so that `build_model` in `train.py` does not contain architecture-specific attribute paths.

#### Acceptance Criteria

1. THE `BaseModelBackbone` SHALL declare `get_mlp_at(self, idx: int) -> nn.Module` as a method that raises `NotImplementedError` by default.
2. THE `GPTNeoBackbone` SHALL implement `get_mlp_at(idx)` returning `self.backbone.transformer.h[idx].mlp`.
3. THE `Qwen2Backbone` SHALL implement `get_mlp_at(idx)` returning `self.backbone.model.layers[idx].mlp`.
4. WHEN `build_model` in `scripts/train.py` retrieves the original MLP for a given layer index, THE `build_model` function SHALL call `model.get_mlp_at(actual_idx)` instead of accessing `model.backbone.transformer.h[actual_idx].mlp` directly.

---

### Requirement 5: `intermediate_dim` Propagation to LoRAConfig

**User Story:** As a researcher, I want the LoRA expert intermediate dimension to be set from the model's VARIANTS configuration, so that Qwen2's 8,960-dimensional MLP is correctly sized rather than defaulting to 4 × 1,536 = 6,144.

#### Acceptance Criteria

1. WHEN `build_model` constructs a `LoRAConfig` for a model whose `model_info` contains an `"intermediate_dim"` key, THE `build_model` function SHALL pass that value as `intermediate_dim` to `LoRAConfig`.
2. WHEN `build_model` constructs a `LoRAConfig` for a model whose `model_info` does not contain an `"intermediate_dim"` key, THE `build_model` function SHALL pass `intermediate_dim=None` to `LoRAConfig`, allowing `LoRAConfig.__post_init__` to default to `4 × hidden_dim`.
3. WHEN a `Qwen2LoRAMLP` is initialized with a `LoRAConfig` where `intermediate_dim=8960`, THE `Qwen2LoRAMLP` SHALL create `up_proj` and `gate_proj` `SharedLoRALayer` instances with `in_features=1536` and `out_features=8960`.

---

### Requirement 6: Generalized `consolidate_shared_weights` via `get_lora_layer_names`

**User Story:** As a developer, I want `ExpertPool.consolidate_shared_weights()` to work for any expert architecture, so that GPU memory is saved for Qwen2 experts the same way it is for GPT-Neo experts.

#### Acceptance Criteria

1. WHEN `consolidate_shared_weights` is called on a pool of `Qwen2LoRAMLP` experts, THE `ExpertPool` SHALL alias the `shared_weight` and `shared_bias` buffers of `gate_proj`, `up_proj`, and `down_proj` in experts `1..N-1` to point to the corresponding buffers in expert `0`.
2. WHEN `consolidate_shared_weights` is called on a pool of `GPTNeoLoRAMLP` experts, THE `ExpertPool` SHALL preserve the existing behavior of aliasing `c_fc` and `c_proj` buffers.
3. THE `GPTNeoLoRAMLP` SHALL implement `get_lora_layer_names()` returning `["c_fc", "c_proj"]`.
4. WHEN `consolidate_shared_weights` is called and expert `0` implements `get_lora_layer_names()`, THE `ExpertPool` SHALL use the returned list to determine which layer attributes to consolidate.
5. IF expert `0` does not implement `get_lora_layer_names()` and does not have a `c_fc` attribute, THEN THE `ExpertPool` SHALL return from `consolidate_shared_weights` without modifying any buffers.

---

### Requirement 7: `lora_moe.py` Attribute Name Generalization

**User Story:** As a developer, I want `LoRAMoELayer` to compute LoRA delta norms and initialize the shared base LoRA for both GPT-Neo and Qwen2 expert architectures, so that metrics and optional shared-base features work correctly with Qwen2.

#### Acceptance Criteria

1. WHEN `get_cached_metrics` iterates over expert projection attributes to compute LoRA delta norms, THE `LoRAMoELayer` SHALL check `gate_proj`, `up_proj`, and `down_proj` in addition to `c_fc` and `c_proj`.
2. WHEN `_init_shared_base_lora` detects the output projection layer to determine `in_features` and `out_features`, THE `LoRAMoELayer` SHALL check for `down_proj` in addition to `c_proj` on expert `0`.
3. WHILE an expert pool contains experts that lack both `c_proj` and `down_proj`, THE `LoRAMoELayer` SHALL skip `_init_shared_base_lora` initialization without raising an error.

---

### Requirement 8: Registry Wiring and Enum Extension

**User Story:** As a developer, I want the new Qwen2 classes to be automatically registered when the training script imports the model and expert modules, so that no changes to the registry lookup logic are needed.

#### Acceptance Criteria

1. THE `ExpertType` enum in `src/project_types.py` SHALL include a `QWEN2_LORA = "qwen2_lora"` member.
2. THE `ModelType` enum in `src/project_types.py` SHALL include a `QWEN2 = "qwen2"` member.
3. WHEN `import src.models` is executed, THE `ModelRegistry` SHALL contain an entry for `"qwen2"` mapping to `Qwen2Backbone`.
4. WHEN `import src.experts` is executed, THE `ExpertRegistry` SHALL contain an entry for `"qwen2_lora"` mapping to `Qwen2LoRAMLP`.
5. THE `src/models/__init__.py` SHALL import `src.models.qwen2` to trigger the `@ModelRegistry.register("qwen2")` decorator.
6. THE `src/experts/__init__.py` SHALL import `src.experts.qwen2_lora` to trigger the `@ExpertRegistry.register("qwen2_lora")` decorator.

---

### Requirement 9: Experiment Configuration for Qwen2-1.5B

**User Story:** As a researcher, I want a ready-to-run YAML experiment config for Qwen2-1.5B with the SPAR router on WikiText-103, so that I can launch a training run without manually constructing the config.

#### Acceptance Criteria

1. THE experiment file `experiments/qwen2_1.5b_stress_v1-wikitext.yaml` SHALL set `model.model_key: qwen2-1.5b`.
2. THE experiment file SHALL set `expert.type: qwen2_lora`.
3. THE experiment file SHALL specify `moe_layer_indices` selecting 6 of the 28 Qwen2 layers.
4. THE experiment file SHALL set `router.type: stress_corrected` with `num_experts: 8` and `top_k: 2`.
5. THE experiment file SHALL set `dataset.dataset_key: wikitext-103` with `max_seq_len: 512`.
6. THE experiment file SHALL set `distributed.strategy: ddp`.

---

### Requirement 10: Backward Compatibility with Existing GPT-Neo Shards and Training

**User Story:** As a researcher, I want all existing GPT-Neo experiments to continue working without re-tokenizing data or modifying configs, so that the Qwen2 integration does not break any current workflows.

#### Acceptance Criteria

1. WHEN `ShardDataset` opens a shard file whose size is consistent with an 8-byte legacy header (i.e., the 9th and 10th bytes are not a valid dtype_flag), THE `ShardDataset` SHALL treat the file as a legacy uint16 shard.
2. WHEN `build_model` is called with a GPT-Neo model key, THE `build_model` function SHALL produce a model functionally identical to the pre-integration behavior.
3. WHEN `consolidate_shared_weights` is called on a `GPTNeoLoRAMLP` pool, THE `ExpertPool` SHALL alias buffers using the same logic as before the generalization.
4. THE 55 existing unit tests SHALL pass without modification after the integration changes are applied.
