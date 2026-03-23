# Plan: Add Qwen2-1.5B to T-MoE

## Context
The codebase currently supports GPT-Neo models only. The user wants to replace GPT-Neo with Qwen2-1.5B (`Qwen/Qwen2-1.5B`, 1536d, 28 layers, SwiGLU MLP). All existing infrastructure — SPAR router, LoRAMoELayer, DDP training, YAML pipeline — must be reused unchanged. Only model/expert-specific wrappers and one coupling in train.py need to change.

---

## Critical Blocker: uint16 Shard Format
Qwen2 vocab is 151936 > uint16 max (65535). `scripts/prepare_data.py` writes `np.uint16` and `scripts/train.py` reads `np.uint16`. This must be fixed before Qwen2 data can be tokenized.

**Fix: versioned shard header**
- New header: `[uint64 token_count][uint16 dtype_flag]` (10 bytes total, was 8)
  - `dtype_flag=0` → uint16 tokens (legacy GPT-Neo shards still work)
  - `dtype_flag=1` → uint32 tokens
- `prepare_data.py` writes dtype_flag based on vocab_size. `ShardDataset` reads flag byte and sets `np.dtype` accordingly.

**Files changed**: `scripts/prepare_data.py` (write), `scripts/train.py` (ShardDataset read logic)

---

## Files to Create

### 1. `src/models/qwen2.py`
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

    def load_pretrained(self) -> None:
        self.backbone = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=COMPUTE_DTYPE)

    def get_mlp_at(self, idx: int) -> nn.Module:
        return self.backbone.model.layers[idx].mlp  # Qwen2 path

    def inject_moe_layers(self, moe_layers: Dict[int, nn.Module]) -> None:
        for idx, layer in moe_layers.items():
            self.backbone.model.layers[idx].mlp = layer
            self.moe_layers[str(idx)] = layer

    def forward(self, input_ids, attention_mask=None, labels=None,
                return_metrics=False, record_usage=True, **kwargs):
        # Identical logic to GPTNeoBackbone.forward():
        # set _forced_record_usage on each MoE layer, call self.backbone(),
        # accumulate aux_loss, collect metrics
```
`model_lookup("qwen2-1.5b")` auto-resolves via VARIANTS scan — no changes to `model_lookup()`.

### 2. `src/experts/qwen2_lora.py`
```python
@ExpertRegistry.register("qwen2_lora")
class Qwen2LoRAMLP(LoRAMLPExpert):
    """SwiGLU MLP: down_proj(silu(gate_proj(x)) * up_proj(x))"""

    def load_from_mlp(self, mlp: nn.Module) -> None:
        # nn.Linear (not Conv1D) — no transpose needed
        # all biases are None (bias=False in Qwen2)
        self.gate_proj = SharedLoRALayer(mlp.gate_proj.weight.detach(), None, ...)
        self.up_proj   = SharedLoRALayer(mlp.up_proj.weight.detach(), None, ...)
        self.down_proj = SharedLoRALayer(mlp.down_proj.weight.detach(), None, ...)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Output shape: [T, hidden_dim] — works with LoRAMoELayer dispatcher
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

    def get_lora_layer_names(self) -> list[str]:
        return ["gate_proj", "up_proj", "down_proj"]
```

### 3. `experiments/qwen2_1.5b_stress_v1-wikitext.yaml`
```yaml
experiment_name: qwen2_1.5b_spar_wikitext
model:
  model_key: qwen2-1.5b
  moe_layer_indices: [2, 6, 10, 14, 18, 22]  # 6 of 28 layers, every 4th
expert:
  type: qwen2_lora
  count: 8
  lora: {rank: 16, alpha: 32, dropout: 0.05}
router:
  type: stress_corrected
  num_experts: 8
  top_k: 2
  temperature: 0.5
  noise_std: 0.05
  ema_alpha: 0.01
  lambda_calib_step: 600
training:
  batch_size: 8
  gradient_accumulation_steps: 8
  steps: 5000
  lr: 3e-4
  warmup_steps: 400
  weight_decay: 0.1
dataset: {dataset_key: wikitext-103, max_seq_len: 512}
distributed: {strategy: ddp, find_unused_parameters: false}
compute: {modal: {gpu: "A100:4"}}
```

---

## Files to Modify

### 4. `src/project_types.py`
```python
class ExpertType(str, Enum):
    GPTNEO_LORA = "gpt_neo_lora"
    QWEN2_LORA  = "qwen2_lora"   # ADD

class ModelType(str, Enum):
    GPTNEO = "gpt_neo"
    QWEN2  = "qwen2"             # ADD
```

### 5. `src/models/base.py`
Add abstract method:
```python
def get_mlp_at(self, idx: int) -> nn.Module:
    raise NotImplementedError
```

### 6. `src/models/gpt_neo.py`
```python
def get_mlp_at(self, idx: int) -> nn.Module:
    return self.backbone.transformer.h[idx].mlp
```

### 7. `src/models/__init__.py`
```python
from src.models import gpt_neo   # existing
from src.models import qwen2     # ADD — triggers @ModelRegistry.register("qwen2")
```

### 8. `src/experts/__init__.py`
```python
from src.experts import qwen2_lora  # ADD — triggers @ExpertRegistry.register("qwen2_lora")
```

### 9. `src/experts/gpt_neo_lora.py`
```python
def get_lora_layer_names(self) -> list[str]:
    return ["c_fc", "c_proj"]
```

### 10. `src/experts/pool.py` — Generalize `consolidate_shared_weights()`
Replace lines 64-81 with:
```python
def consolidate_shared_weights(self) -> None:
    if self.num_experts < 2:
        return
    e0 = self.experts[0]

    if hasattr(e0, "get_lora_layer_names"):
        layer_names = e0.get_lora_layer_names()
    elif hasattr(e0, "c_fc") and e0.c_fc is not None:
        layer_names = ["c_fc", "c_proj"]
    else:
        return

    ref = {}
    for name in layer_names:
        layer = getattr(e0, name, None)
        if layer is not None:
            ref[name] = {k: v for k, v in layer._buffers.items()}

    for expert in self.experts[1:]:
        for name, buffers in ref.items():
            layer = getattr(expert, name, None)
            if layer is not None:
                for buf_name, buf_val in buffers.items():
                    layer._buffers[buf_name] = buf_val
```

### 11. `scripts/train.py` — One-line fix (line 198)
```python
# Before:
original_mlp = model.backbone.transformer.h[actual_idx].mlp

# After:
original_mlp = model.get_mlp_at(actual_idx)
```
Also pass `intermediate_dim` from VARIANTS into `LoRAConfig`:
```python
lora_cfg = LoRAConfig(
    hidden_dim=model.hidden_dim,
    intermediate_dim=model_info.get("intermediate_dim"),  # 8960 for Qwen2, None→4x for GPT-Neo
    ...
)
```

### 12. `src/layers/lora_moe.py` — Minor patches
- `get_cached_metrics()`: extend `for attr in ("c_fc", "c_proj"):` → add `"gate_proj", "up_proj", "down_proj"` (already guarded by `getattr(..., None)`)
- `_init_shared_base_lora()`: check both `"c_proj"` and `"down_proj"` for output-proj detection

### 13. `src/training/fsdp_utils.py` — FSDP wrap policy (defer until strategy=fsdp needed)
```python
def _resolve_fsdp_wrap_target(cfg) -> type:
    target = getattr(getattr(cfg, "distributed", {}), "fsdp_wrap_target", "GPTNeoBlock")
    if target == "GPTNeoBlock":
        from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock
        return GPTNeoBlock
    elif target == "Qwen2DecoderLayer":
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        return Qwen2DecoderLayer
```

---

## Implementation Order

```
1.  src/project_types.py          — add enums (purely additive)
2.  src/models/base.py            — add abstract get_mlp_at()
3.  src/models/gpt_neo.py         — implement get_mlp_at()
4.  src/experts/gpt_neo_lora.py   — add get_lora_layer_names()
5.  src/experts/qwen2_lora.py     — CREATE Qwen2LoRAMLP
6.  src/models/qwen2.py           — CREATE Qwen2Backbone
7.  src/models/__init__.py        — add import (triggers registry)
8.  src/experts/__init__.py       — add import (triggers registry)
9.  src/experts/pool.py           — generalize consolidate_shared_weights()
10. src/layers/lora_moe.py        — attr-name patches
11. scripts/train.py              — get_mlp_at() + intermediate_dim
12. scripts/prepare_data.py       — uint32 shard write (BLOCKER)
13. scripts/train.py              — ShardDataset uint32 read path
14. src/training/fsdp_utils.py    — dynamic wrap target (defer until FSDP)
15. experiments/qwen2_1.5b_...    — CREATE experiment YAML
```

---

## Key Pitfalls

| Pitfall | Detail | Fix |
|---------|--------|-----|
| uint16 overflow | Qwen2 vocab 151936 > 65535 | 10-byte versioned shard header; uint32 tokens |
| No Conv1D in Qwen2 | `nn.Linear`, not HF `Conv1D` — no `.nf` attr, no transpose | `mlp.gate_proj.weight.detach()` directly |
| No bias in Qwen2 MLP | All three projections have `bias=False` | Pass `None` as `shared_bias` to `SharedLoRALayer` |
| intermediate_dim mismatch | Qwen2: 8960 ≠ 4×1536=6144 default | Pass `intermediate_dim=8960` from VARIANTS to `LoRAConfig` |
| `make_base_trainable()` | GPT-Neo specific; never called for Qwen2 (`trainable_base=false`) | Keep as-is; silent skip via existing `hasattr(e0, "c_fc")` guard |
| model_key convention | `model_lookup()` builds key as `{model_type.replace('_','-')}-{variant}` → `"qwen2-1.5b"` | Set `model_key: qwen2-1.5b` in YAML |

---

## Verification

1. **Unit tests** (no regressions on 55 existing tests):
   - `tests/experts/test_qwen2_lora.py`: forward shape [T,1536]→[T,1536], `load_from_mlp`, consolidation
   - `tests/models/test_qwen2_backbone.py`: `model_lookup`, `inject_moe_layers`, `get_mlp_at`
   - `tests/experts/test_lora_impl.py`: `consolidate_shared_weights` via `get_lora_layer_names()` protocol

2. **Smoke test** (CPU, no GPU needed):
   ```python
   from src.configs.model import model_lookup
   info = model_lookup("qwen2-1.5b")
   assert info["hidden_dim"] == 1536
   ```

3. **Full run**: `modal run run_modal_training.py` with `qwen2_1.5b_stress_v1-wikitext.yaml`
   - Check val_ppl decreasing, `eff_E > 1` (not collapsed), `ema_load_std < 0.3`
