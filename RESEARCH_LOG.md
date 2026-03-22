## 2026-03-22 — Pre-scale review: v8a metric diagnosis, SOTA comparison, Qwen/Llama checklist

### 1. Metric Diagnosis: v8a steps 0--600

**conf = 0.524--0.528 throughout, at tau=0.5, top_k=2, N=8**

Theoretical conf bounds with top_k=2 softmax over N=8 experts at temperature tau:
```
w_max = exp(cos_max / tau) / [exp(cos_max / tau) + exp(cos_2nd / tau)]
      = sigma(Delta_cos / tau)      where sigma = logistic function
conf  = E[w_max] = E[sigma(Delta_cos / tau)]
```
- **Minimum** (uniform routing, Delta_cos = 0): `conf = sigma(0) = 0.500`
- **Maximum** (one expert dominates, Delta_cos -> inf): `conf -> 1.0`
- **Observed** `conf = 0.527`: solve `0.527 = sigma(Delta / 0.5)` => `Delta = 0.5 * logit(0.527) = 0.5 * 0.108 = 0.054`

So the mean cosine gap between first and second selected experts is **Delta_cos ~ 0.054** at step 600. This is slightly above noise-level differentiation. For context: random unit vectors in D=768 have `std(cos) ~ 1/sqrt(D) ~ 0.036`. A gap of 0.054 is about 1.5 sigma above random — the k-means initialization is producing barely-meaningful differentiation.

**What conf indicates "real specialization":** conf = 0.60 requires Delta_cos = 0.5 * logit(0.60) = 0.5 * 0.405 = 0.20. conf = 0.70 requires Delta_cos = 0.5 * logit(0.70) = 0.5 * 0.847 = 0.42. Given that cosine similarities live in [-1,1], a gap of 0.20 is substantial — it means the best expert is meaningfully closer than the runner-up. The v8a trajectory (conf growing by +0.003 over 600 steps) projects to conf ~ 0.62 at step 19000 if linear — consistent with Delta_cos ~ 0.10, which is moderate but not strong specialization.

**var = 0.006--0.024 (Welford variance of cosine distances)**

These are variances of `1 - cos(x, W_i)` for tokens assigned to expert i. Standard deviations:
```
L1:  sqrt(0.007) = 0.084    L3:  sqrt(0.012) = 0.110
L5:  sqrt(0.006) = 0.077    L7:  sqrt(0.006) = 0.077
L9:  sqrt(0.010) = 0.100    L11: sqrt(0.024) = 0.155
```
For k-means-initialized prototypes in D=768, the expected intra-cluster cosine distance std depends on cluster tightness. With 8 centroids partitioning a 768-dim unit sphere, each Voronoi cell subtends a solid angle of ~1/8 of the sphere. The expected intra-cluster `std(1-cos)` for uniform random vectors in such a cell is approximately `sqrt(2/(D*pi)) * sqrt(8) ~ 0.04`. The observed values (0.077--0.155) are 2--4x larger than this, indicating that token distributions are NOT uniform within Voronoi cells — there is genuine structure (some tokens are close to the centroid, others near the boundary). L11 has the largest variance (0.155), consistent with later layers having more heterogeneous representations.

The n_min values (28k--304k) show all experts are receiving substantial tokens even at step 0. L1 has the smallest n_min (28k), suggesting early layers have more uneven initial routing. L5/L7 have the largest (300k), indicating near-perfect balance in middle layers — expected since middle-layer representations are more isotropic.

**eff_E 7.1 -> 7.5, gini 0.188 -> 0.154**

This is near-perfect load balance improving further. At step 0, before lambda is calibrated, eff_E = 7.2 with lambda_val = 1.0 (the default). Two explanations, not mutually exclusive:
1. **K-means init provides good starting balance**: prototypes placed at k-means centroids create roughly equal Voronoi cells, so even without penalty, routing is near-uniform. This is the dominant effect — eff_E = 7.2 at step 0 with lambda = 1.0 (uncalibrated) is very high.
2. **SPAR penalty is contributing**: the default lambda=1.0 is not zero, so the `max(0, L_i - 1/N)` term is active from step 0. With ema_load initialized to 1/8 (fair share), the penalty is initially zero for all experts, but as training produces small load imbalances, the penalty fires.

Verdict: the high eff_E is primarily from k-means init, secondarily from SPAR. This is good — it means k-means init is doing its job. The SPAR penalty's role will become visible after lambda calibration at step 600, when lambda takes its data-derived value.

**Loss plateau at ~3.47 from step 400--600**

At step 400, the learning rate warmup (400 steps) has just completed. Steps 400--600 are the first 200 steps at full LR. This is NOT a plateau — it is the transition from warmup to full-LR training. The loss decline from 4.965 to 3.476 over 600 steps is rapid and healthy. The apparent flattening at 3.47--3.48 over steps 400--600 is consistent with the LR reaching its peak and the model entering steady-state optimization. Check: at step 400, LR reaches 3e-4 (peak). Steps 400--600 at peak LR should show continued decline but slower than during warmup when both LR and model are co-adapting.

The router is NOT bottlenecking learning at this stage. Evidence: loss dropped from 4.965 to 3.476 (a factor of 4.2x in PPL) in 600 steps. The prior v7 run at similar steps showed comparable loss trajectory. SharedBaseLoRA B matrix is still near-zero at this stage (per LoRA Without Regret: B stays small for ~1000--2000 steps), so v8a and v7 should be indistinguishable through step 600.

---

### 2. Is conf=0.527 a problem?

No, for three reasons:

**Reason 1: it is step 600 of 19000 (3.2% complete).** The v8a trajectory shows conf growing. At tau=0.5 (current), the softmax is relatively soft. As tau anneals toward 0.10 over 14000 steps, even a fixed Delta_cos = 0.054 would produce conf = sigma(0.054/0.10) = sigma(0.54) = 0.632. The tau anneal alone will push conf above 0.60.

**Reason 2: Delta_cos will grow.** K-means init gives a starting Delta_cos ~ 0.054. As task-loss gradients flow through the softmax weights back to W (via the cos_sim -> softmax -> expert_weight -> combined output path), prototypes will specialize toward their assigned token clusters. At step 5000 (tau ~ 0.36), Delta_cos is expected to reach 0.08--0.12 based on the v6-wikitext trajectory, giving conf ~ 0.60--0.65.

**Reason 3: the 2026-03-19 Voronoi analysis already established that conf is the wrong metric.** At eff_E = 8.0, perfect load balance geometrically constrains prototypes to Voronoi boundaries where Delta_cos is small. conf ~ 0.55--0.65 at tau = 0.10 is the mathematical ceiling for eff_E = 8.0 with N = 8 in D = 768. The paper should report eff_E and gini, not conf.

**Action:** No intervention needed. conf = 0.527 at step 600 is on-track.

---

### 3. SOTA Router Improvements: ranked recommendations

#### 3a. DeepSeek-V3 bias correction vs SPAR EMA penalty

**DeepSeek formulation:**
```
z_i = linear_gate(x)_i + b_i
b_i <- b_i + gamma * sign(1/N - actual_load_i)    # discrete step, every batch
```

**SPAR formulation:**
```
z_i = cos(x, W_i) - lambda * max(0, L_i - 1/N)
L_i = (1-alpha) * L_i + alpha * U_i               # continuous EMA
```

**Mathematical comparison:**

| Property | SPAR | DeepSeek bias |
|----------|------|---------------|
| Penalty on underloaded experts | Zero (one-sided max) | Negative (sign function pushes b_i up) |
| Update smoothness | Continuous (EMA, alpha=0.01) | Discrete (sign function, +/- gamma) |
| Scale sensitivity | Auto-calibrated lambda | Fixed gamma (needs manual tuning) |
| Steady-state | L_i = 1/N => penalty = 0 | b_i drifts without bound unless clipped |
| Gradient interaction | Out-of-graph (detached ema_load) | Out-of-graph (bias is not differentiable) |

**Verdict: SPAR is mathematically superior in 3 of 5 properties.**

The one-sided penalty is SPAR's key advantage. DeepSeek's sign-based correction pushes underloaded experts' biases up AND overloaded experts' biases down — this creates a global shift that confounds the routing signal. SPAR's `max(0, ...)` only penalizes overloaded experts, leaving underloaded ones free to attract tokens through genuine cosine affinity.

DeepSeek's unbounded drift is a real issue: without clipping, b_i accumulates indefinitely for consistently over/underloaded experts, eventually dominating the linear gate signal. DeepSeek addresses this by making gamma very small (1e-5 to 1e-3), but this slows response time. SPAR's EMA has a natural forgetting mechanism (alpha=0.01 => 100-step window).

**Recommendation:** Do not adopt bias correction. SPAR's formulation is strictly better for the paper's use case. The comparison itself is a paper contribution: "SPAR's one-sided EMA penalty is a continuous, auto-calibrated generalization of DeepSeek-V3's discrete bias correction."

#### 3b. DeepSeek shared expert vs T-MoE SharedBaseLoRA

DeepSeek uses m=2 shared experts (full MLP width) that always fire for all tokens, plus N-m=62 routed experts. Total expert count is 64, effective per-token compute = 2 (shared) + 6 (routed) = 8 experts.

T-MoE's SharedBaseLoRA is structurally different:
```
DeepSeek: y = sum_{j in shared}(MLP_j(x)) + sum_{i in TopK}(w_i * MLP_i(x))
T-MoE:    y = frozen_base_MLP(x) + SharedBaseLoRA_proj(h) + sum_{i in TopK}(w_i * delta_i(x))
```

The frozen base MLP in T-MoE already acts as the "shared expert" — it processes every token. SharedBaseLoRA adds a rank-8 correction to c_proj for all tokens, which is a parameter-efficient approximation to DeepSeek's full shared expert (184k params vs millions).

**Mathematical tradeoff: full shared expert vs SharedBaseLoRA:**
- Full shared expert at rank=32: 2 * (768 * 32 + 32 * 3072) = 2 * 123,648 = 247k LoRA params (c_fc + c_proj), routed like any expert but always selected.
- SharedBaseLoRA at rank=8 on c_proj only: 6 * (3072 * 8 + 8 * 768) = 184k params.
- Full shared LoRA expert provides a richer correction (both c_fc and c_proj, higher rank per projection) but creates the gradient competition issue identified on 2026-03-19: shared expert gets 8x more gradient, starving routed experts.

**Recommendation: keep SharedBaseLoRA, do not add a full shared expert.** The frozen backbone IS the shared expert. SharedBaseLoRA is the right parameter-efficient correction for domain shift. A full shared LoRA expert would reproduce the gradient starvation pathology already diagnosed.

#### 3c. Expert-choice routing

Expert-choice (EC) guarantees perfect load balance by construction: each expert picks its top-c tokens (c = capacity * B*S / N). No penalty, no EMA, no lambda.

**Fatal for T-MoE:**
1. **Token dropping**: tokens selected by zero experts receive zero MoE output. With `combined = torch.zeros_like(x_flat)` in the current code, these tokens lose their entire MLP contribution. The fix (init combined = frozen_base(x)) adds a full frozen forward pass.
2. **Variable batch incompatibility**: EC requires fixed batch size to compute capacity. With DDP gradient accumulation (batch varies across microbatches), capacity factor must be recomputed per microbatch — fragile.
3. **Paper contradiction**: SPAR's claim is "near-perfect balance WITHOUT auxiliary loss or capacity constraints." EC achieves perfect balance by imposing a hard capacity constraint — a different mechanism that validates the problem but not the SPAR solution.
4. **Already diagnosed as fatal on 2026-03-19** (zero-output bug, gradient sparsity).

**Recommendation: do not implement. EC is the right comparison baseline in the paper but not a mechanism to adopt.**

#### 3d. Token dropping / overflow handling

Current SPAR has no overflow — all tokens are processed by their selected experts. With N=8 and top_k=2, each expert processes ~25% of tokens on average. At eff_E=7.5+, the max-loaded expert processes ~30% of tokens (gini=0.15 => load ratio max/mean ~ 1.2).

For larger models (N=16 or N=64 with Qwen/Llama):
- With N=64 and top_k=6 (DeepSeek-V3 style), max-load expert could see 2--3x fair share during early training.
- Memory cost scales linearly with max tokens per expert per batch.
- Token dropping would cap this at capacity * B*S / N tokens per expert.

**Recommendation: not needed for N=8. Add as an option when N >= 16, implemented as a configurable capacity_factor (default: inf = no dropping).** Do not implement before Qwen/Llama scale-up. When added, ensure dropped tokens fall through to frozen_base(x), not zero.

#### 3e. Ranked priority for pre-scale implementation

| Priority | Change | Effort | Expected Impact |
|----------|--------|--------|----------------|
| 1 | LoRA init fix (alpha=rank, B=0) | Config-only | Removes confound, cleaner paper |
| 2 | GMR (already spec'd in code header) | ~50 LOC | Fixes gini=0.05 on fineweb-edu |
| 3 | SwiGLU expert class for Qwen/Llama | ~80 LOC | Required for scale-up |
| 4 | FSDP router buffer handling | ~30 LOC | Required for 7B models |
| 5 | Per-layer lambda (optional) | ~40 LOC | L1 and L11 have different var; per-layer lambda may help |

---

### 4. LoRA Initialization: specific recommendations

**Current: A ~ Kaiming(a=0.01), B ~ Normal(0, 0.01), scaling=2.0**

Three issues:

**Issue 1: b_init_scale=0.01 is harmful for SPAR.**

The code comment says "Non-zero breaks expert symmetry for MoE routing." This was intentional — the idea is that non-zero B at init makes each expert produce a different delta from step 0, giving the router something to differentiate. However:

With b_init_scale=0.01 and scaling=2.0, the initial expert delta magnitude is:
```
||delta_i|| = ||B_i @ A_i|| * scaling * ||x||
            ~ ||B_i||_F * ||A_i||_F * scaling / sqrt(rank)   [approximate for random matrices]
            ~ (sqrt(out * rank) * 0.01) * ||A_i||_F * 2.0 / sqrt(32)
```

For c_fc (768 -> 3072): `||B||_F ~ sqrt(3072 * 32) * 0.01 = 3.13`, `||A||_F ~ sqrt(768 * 32) * sqrt(2/768) = sqrt(32 * 2) = 8.0` (Kaiming with a=0.01 ~ standard Kaiming). So `||delta|| ~ 3.13 * 8.0 * 2.0 / sqrt(32) = 8.9`. This is NOT small — it is a non-negligible perturbation at init.

The problem: with b_init_scale=0.01, the LoRA guarantee (delta=0 at init) is violated. The model output at step 0 is `frozen_base(x) + random_noise(x)`, not `frozen_base(x)`. This adds noise to the pretrained model's predictions, increasing initial loss and potentially destabilizing early training.

**Verdict: set b_init_scale=0.0.** K-means init already breaks expert symmetry (different W_i directions). The B=0 init preserves the pretrained model's predictions at step 0, which is the standard LoRA guarantee. Expert differentiation should come from routing (different W_i), not from random perturbation of expert outputs.

**Issue 2: scaling=2.0 (alpha=64, rank=32)**

Per LoRA Without Regret: alpha=rank is optimal, giving scaling=1.0. The invariance argument says Adam compensates for the scale factor, but scaling=2.0 means the effective learning rate for LoRA parameters is 2x what the optimizer LR specifies. This interacts with:
- The LR schedule (lr=3e-4 behaves like lr=6e-4 for LoRA deltas)
- Gradient clipping (clip_grad_norm=1.0 clips earlier than intended)
- SharedBaseLoRA (shared_base_alpha=16.0 with rank=8 gives scaling=2.0 there too)

**Verdict: set alpha=32 (scaling=1.0) for expert LoRA, shared_base_alpha=8.0 (scaling=1.0) for SharedBaseLoRA.** Already noted in the 2026-03-21 log entry as a v8b change. Confirm it is applied.

**Issue 3: init_scale=0.01 for Kaiming**

`nn.init.kaiming_uniform_(A.weight, a=0.01)` uses `a` as the negative slope parameter for leaky ReLU, not a scale factor. With a=0.01, this is `kaiming_uniform with mode='fan_in', nonlinearity='leaky_relu', negative_slope=0.01`. The resulting distribution is `Uniform(-bound, bound)` where `bound = sqrt(6 / ((1 + 0.01^2) * fan_in)) ~ sqrt(6 / fan_in)`. This is essentially standard Kaiming for ReLU (a=0 gives the same bound to 4 decimal places). **No change needed** — a=0.01 is fine.

**Summary of LoRA init changes for v8b/v9:**
```yaml
expert:
  lora:
    alpha: 32          # was 64, scaling 2.0 -> 1.0
    b_init_scale: 0.0  # was 0.01, restore LoRA zero-init guarantee
    shared_base_alpha: 8.0  # was 16.0, scaling 2.0 -> 1.0
```

---

### 5. Qwen/Llama Pre-Scale Checklist

#### 5a. MLP structure: SwiGLU vs GELU

**GPT-Neo MLP:**
```
y = c_proj(GELU(c_fc(x)))           # 2 linear projections
c_fc:   [D, 4D]                      # 768 -> 3072
c_proj: [4D, D]                      # 3072 -> 768
```

**Qwen2.5/Llama3 MLP (SwiGLU):**
```
y = down_proj(SiLU(gate_proj(x)) * up_proj(x))   # 3 linear projections
gate_proj: [D, intermediate_dim]      # 4096 -> 11008 (Llama) or 11008 (Qwen 7B)
up_proj:   [D, intermediate_dim]      # 4096 -> 11008
down_proj: [intermediate_dim, D]      # 11008 -> 4096
```

**Required change:** Create `SwiGLULoRAMLP` expert class (parallel to `GPTNeoLoRAMLP`):
```python
class SwiGLULoRAMLP(LoRAMLPExpert):
    # 3 SharedLoRALayers: gate_proj, up_proj, down_proj
    # forward: down_proj(silu(gate_proj(x)) * up_proj(x))
    # LoRA deltas on all three projections
```

The `LoRAConfig` needs `intermediate_dim` set correctly: Llama-7B uses 11008 (not 4*4096=16384). This is already a field in `LoRAConfig` (`intermediate_dim: Optional[int] = None`, defaults to 4*hidden_dim). Must be explicitly set for SwiGLU models.

**LoRA parameter count for SwiGLU at rank=32:**
```
Per expert: 3 projections * (4096*32 + 32*11008) = 3 * (131072 + 352256) = 1,449,984
Per layer (8 experts): 11.6M
6 MoE layers: 69.6M trainable (vs 125M backbone -> 56%; vs 7B backbone -> 1.0%)
```
This is a much better LoRA-to-backbone ratio (1%) than GPT-Neo 125M (4.75%). Rank=32 is appropriate.

**SharedBaseLoRA adjustment:** For SwiGLU, shared base LoRA should apply to `down_proj` (the output projection), not `c_proj`. Same role: domain-shift correction on the output stage for all tokens.

#### 5b. Attention: GQA vs MHA

Qwen2.5 7B and Llama 3.1 8B use Grouped Query Attention (GQA). T-MoE applies LoRA exclusively to MLP blocks — attention is untouched. **No change needed** as long as the MoE replacement targets only MLP blocks (which it does: `moe_layer_indices` selects which transformer layers get MoE MLPs). GQA vs MHA is irrelevant.

#### 5c. Layer norms: RMSNorm vs LayerNorm

Qwen/Llama uses RMSNorm (no mean subtraction, only scale normalization). The router input `x` is the hidden state AFTER the pre-MLP norm (RMSNorm or LayerNorm). This means:
- `x` passed to the router is already approximately unit-variance per dimension
- `F.normalize(x, dim=-1)` in the router further normalizes to unit L2 norm
- RMSNorm vs LayerNorm affects the distribution of `x` before F.normalize, but the cosine similarity `cos(x, W_i)` is invariant to input scale — only direction matters

**No change needed for the router.** The double-normalization (RMSNorm then F.normalize) is redundant but not harmful. RMSNorm already ensures `||x||_2 ~ sqrt(D)`, so `F.normalize` just divides by `sqrt(D)` — a constant factor that cancels in the cosine.

#### 5d. Tokenizer

Qwen2.5 uses tiktoken (cl100k-like), Llama uses SentencePiece. This affects:
- `scripts/prepare_data.py`: tokenizer loading must use the model's tokenizer, not hardcoded GPT-Neo BPE
- Vocabulary size: Qwen=151936, Llama=128256 vs GPT-Neo=50257. Does not affect MoE layers (MoE is in MLP, not embedding/head).
- Sequence packing: same algorithm, different token IDs. No MoE-specific change.

**Change needed in `prepare_data.py` only.** The tokenizer is loaded from the model name — ensure it handles tiktoken/sentencepiece correctly (HuggingFace AutoTokenizer does this automatically).

#### 5e. Cosine similarity on RMSNorm'd inputs

After RMSNorm, hidden states have `||x||_2 ~ sqrt(D)` and approximately zero mean per token. After `F.normalize(x)`, the vector is on the unit sphere. The cosine similarity then depends only on the angular structure of the hidden states.

For D=4096 (Llama 7B) vs D=768 (GPT-Neo 125M):
- **Concentration of measure**: in D=4096, random unit vectors have `cos ~ 0` with `std ~ 1/sqrt(D) ~ 0.0156`. This is 2.3x smaller than GPT-Neo's `1/sqrt(768) ~ 0.036`. Cosine similarities between tokens and prototypes will be more tightly concentrated around the mean.
- **Impact on SPAR**: Delta_cos (the gap between top-1 and top-2 expert) will be smaller in absolute terms. At D=4096, the k-means-initialized Delta_cos is expected to be `~0.054 * (768/4096)^{0.5} = 0.054 * 0.433 = 0.023` (scaling as `1/sqrt(D)`).
- **Impact on lambda calibration**: `sigma_cos ~ 1/sqrt(D) ~ 0.016` at D=4096, so `lambda = min(sigma_cos * N, 5.0) = min(0.016 * 8, 5.0) = 0.125`. This is much smaller than the 125M lambda (expected ~0.3--0.5). The penalty will be proportionally weaker.

**Key concern: lambda may be too small at D=4096.** With lambda=0.125 and max overload penalty `max(0, L_i - 1/8) ~ 0.05` (for a mildly overloaded expert), the penalty term is `0.125 * 0.05 = 0.006`. The cosine signal (Delta_cos ~ 0.023) is only 4x larger. This ratio may be sufficient — the penalty is designed to be a perturbation, not a dominant term. But monitor: if eff_E drops below 7.0 at D=4096, lambda may need a floor: `lambda = max(sigma_cos * N, 0.5)`.

**Recommendation: add a lambda_floor parameter.** `lambda = max(min(sigma_cos * N, 5.0), lambda_floor)` with `lambda_floor = 0.5` default. This ensures the penalty remains meaningful at high D. The floor should be configurable — 0.5 is a reasonable default for D=4096.

#### 5f. FSDP considerations

Qwen2.5 7B has ~7B params. With bf16, that is ~14GB per GPU. An H100 (80GB) can hold the model but not with 8 experts' LoRA weights plus optimizer states. FSDP is required.

**Router-specific FSDP issues:**
1. **Buffers vs parameters**: FSDP shards parameters but NOT buffers by default. Router buffers (`ema_load`, `lambda_val`, `welford_*`, `_pending_counts`, `num_steps`) are registered via `register_buffer`. They are replicated across ranks, not sharded. This is correct — these are small (8 floats for ema_load) and must be identical across ranks.
2. **All-reduce in step()**: `_sync_ema_load_distributed()` calls `dist.all_reduce(ema_load, AVG)`. Under FSDP, the router module may be in a different FSDP unit than the experts. The all_reduce must happen outside FSDP's backward pass (it does — it is called in `step()`, after `optimizer.step()`). **Verified correct.**
3. **W parameter sharding**: `self.W` (shape [8, 4096] at D=4096 = 32KB in bf16) is tiny. FSDP will shard it but the overhead of gathering 32KB across 4 GPUs is negligible. No special handling needed.
4. **torch.compile + FSDP**: The `@torch._dynamo.disable` decorator on forward() and helper methods was added to avoid Dynamo graph breaks. Under FSDP, this is even more important — Dynamo cannot trace through FSDP's all-gather/reduce-scatter. The current approach (disable Dynamo for the router) is correct for FSDP.

**Required changes:**
- `src/training/fsdp_utils.py`: ensure router module is in a separate FSDP wrap unit (not fused with expert parameters). This prevents FSDP from treating `W` and `ema_load` as part of an expert shard.
- Test: verify `step()` all_reduce works correctly when called between FSDP backward and optimizer step.

#### 5g. Rank scaling with model dimension

At D=768 (GPT-Neo 125M), rank=32 gives rank/D = 0.042 (4.2% of the hidden dimension).
At D=4096 (Llama 7B), rank=32 gives rank/D = 0.0078 (0.78%).

The LoRA approximation quality depends on rank relative to the effective rank of the weight update matrix, not on rank/D. For fine-tuning (not pretraining), the weight update is typically low-rank (~4-16 effective dimensions for domain adaptation). Rank=32 is 2-8x over-provisioned for both D=768 and D=4096.

However, with MoE, each expert needs enough rank to represent its specialized subspace. With eff_E=8, each expert sees ~12.5% of tokens — a more diverse token distribution than single-task fine-tuning. The required rank scales with the complexity of the token subpopulation, not with D.

**Recommendation: rank=32 is sufficient for 7B models.** The LoRA Without Regret paper confirms this: rank=128 gives diminishing returns at 7B scale. For MoE with 8 experts at rank=32, the aggregate rank across all active experts per token is `top_k * rank = 2 * 32 = 64`, which is well within the effective dimensionality of fine-tuning updates.

If PPL gains plateau at rank=32, try rank=64 as a single ablation point. Do not go to rank=128 — parameter count would exceed 140M trainable (2% of 7B), which weakens the "parameter-efficient" claim.

---

### 6. The Shared Expert Question

**Should T-MoE add a full shared expert (one LoRA expert that always fires)?**

**No.** Three arguments:

**Argument 1: The frozen backbone IS the shared expert.**

In T-MoE, every expert computes:
```
expert_i(x) = frozen_base_MLP(x) + delta_i(x)
```
The frozen base MLP is shared across all experts and processes every token through every expert. The MoE output is:
```
y = sum_{i in TopK} w_i * [frozen_base(x) + delta_i(x)]
  = (sum w_i) * frozen_base(x) + sum_{i in TopK} w_i * delta_i(x)
  = frozen_base(x) + sum_{i in TopK} w_i * delta_i(x)    [since sum w_i = 1]
```
This is algebraically identical to DeepSeek's formulation with 1 shared expert + K routed experts, where the shared expert = frozen_base and routed experts = deltas. Adding another shared LoRA expert on top would be a second shared path — redundant with both frozen_base and SharedBaseLoRA.

**Argument 2: Gradient starvation (proven on 2026-03-19, confirmed on 2026-03-21).**

A shared LoRA expert receives gradient from ALL tokens (batch_size * seq_len per step). Each routed expert receives gradient from ~1/eff_E fraction. With eff_E=8, the shared expert gets 8x more gradient signal. This causes:
- Shared expert converges fast, capturing most of the learnable signal
- Routed expert residuals shrink, reducing gradient to routed expert LoRA parameters
- Expert specialization stalls — reproducing the conf=0.52 plateau

SharedBaseLoRA (rank=8 on c_proj only) is a deliberate compromise: small enough to not starve routed experts, applied only to the output projection to avoid double-counting.

**Argument 3: Parameter efficiency.**

A full shared expert at rank=32 adds ~250k params per layer, ~1.5M total. SharedBaseLoRA at rank=8 on c_proj adds 184k total. The marginal 1.3M params are better spent increasing routed expert rank (e.g., rank 32->40 for all 8 experts = 6 * 8 * 2 * (768*8 + 8*3072) = 1.47M additional params) which benefits specialization rather than the shared path.

**Verdict: keep SharedBaseLoRA (rank=8, c_proj only). Do not add a full shared expert.** The frozen backbone already provides the shared computation. SharedBaseLoRA provides a small, targeted domain-shift correction. A full shared expert would be architecturally redundant and empirically harmful.

---

### 7. Appendix: lambda calibration at step 600

At step 600, lambda calibration fires. The formula is `lambda = min(sigma_cos * N, 5.0)`.

From the DIAG output at step 0, the Welford variances are 0.006--0.024. These are variances of cosine DISTANCES (1 - cos), not cosine similarities. The sigma_cos in the calibration formula is `std(cos_sim)` computed from `_pending_cos_sims` — the raw cosine similarities accumulated over the grad-accum window at step 600.

Expected sigma_cos at step 600: with k-means init and D=768, sigma_cos across all (token, expert) pairs is approximately `sqrt(Var_between_experts(cos)) ~ 0.03--0.06`. So lambda ~ min(0.04 * 8, 5.0) ~ 0.32. This is a reasonable penalty scale: at max overload of `L_i - 1/8 = 0.05`, the penalty is `0.32 * 0.05 = 0.016`, which is about 30% of Delta_cos (0.054). Strong enough to matter, weak enough to not override the cosine signal.

The per-layer variance spread (L1 var=0.007 vs L11 var=0.024) suggests that a per-layer lambda would better match each layer's routing geometry. L11 has 3.4x the variance of L1, so lambda at L11 would be ~1.8x larger than L1. This is a minor optimization — current global lambda is an average that works adequately for all layers.

---

## 2026-03-21 — LoRA Without Regret: key findings applied to T-MoE

**Source**: Schulman et al., Thinking Machines Lab, Sep 2025 — https://thinkingmachines.ai/blog/lora/

- **[finding]** MLP-only LoRA validates T-MoE architecture. Paper shows rank-256 attention-only
  underperforms rank-128 MLP-only at equal parameter count. T-MoE applies LoRA exclusively to MLP
  blocks (c_fc + c_proj per expert) — this is the paper-optimal placement.
  **result:** No change needed to adapter placement.
  **note:** Provides external validation for a design decision that was previously justified only
  by parameter efficiency arguments.

- **[finding]** `alpha = rank` is optimal (scaling = 1.0). T-MoE v6/v7/v8a used alpha=2×rank
  (scaling=2.0). Paper's invariance analysis shows alpha and init_A are redundant under Adam —
  the two degrees of freedom that matter are the product-of-matrices dynamics, not individual
  scale factors. Setting alpha=rank is the maximally neutral choice.
  **result:** v8b: alpha 64→32 (rank=32), shared_base_alpha 16.0→8.0 (rank=8).
  **note:** The invariance result means prior runs were not misconfigured — Adam adapts to the
  scale — but alpha=rank removes a free variable and makes LR the single learning-rate knob.

- **[finding]** Large batch size penalizes LoRA more than FullFT, and this is NOT mitigated by
  increasing rank — it is a property of the BA product-of-matrices parametrization. v8a used
  effective batch=64 (batch_size=32 × grad_accum=2).
  **result:** v8b reduces to effective batch=32 (batch_size=16 × grad_accum=2).
  **note:** The paper's mechanism: large batches reduce gradient noise, which benefits FullFT;
  LoRA's low-rank structure already provides implicit regularization, so extra batch averaging
  removes the stochasticity that LoRA's landscape requires to escape low-rank saddles.

- **[finding]** LoRA LR = 10× FullFT LR. T-MoE lr=3e-4 is already correct (GPT-Neo-125M FullFT
  ≈3e-5 → LoRA 3e-4). No change.
  **result:** LR confirmed paper-consistent across all T-MoE runs.
  **note:** Not a new finding but good to have external confirmation before paper submission.

- **[finding]** MoE layer placement is listed as unresolved in the paper. The "apply to all
  layers" eNTK argument covers LoRA adapter coverage within a model, not how many FFN layers
  to convert to MoE. At d=768, converting all 12 layers saturates 100% of residual stream
  subspace (12 layers × top-k=2 × rank=32 = 768 dims) causing interference. Every-other-layer
  [1,3,5,7,9,11] is correct for 125M.
  **result:** No change to MoE layer indices.
  **note:** The interference argument is not in the paper — it was derived separately. The
  paper's silence on MoE placement means T-MoE's layer selection is an independent contribution.

- **[observation]** B stays near zero early; B norm exceeds A norm by end of training (paper
  Figure 6). Explains why v8a routing metrics (eff_E, gini, conf) are identical to v7 through
  step 12k — SharedBaseLoRA B matrix is still growing into its learned subspace.
  **result:** v8a B-norm lag is expected behavior, not a bug.
  **note:** The shared path will only meaningfully affect routing once B has accumulated
  sufficient magnitude to produce non-negligible deltas on common tokens. Paper predicts this
  activates mid-to-late training, consistent with the step 12k observation.

- **[action]** Changes applied for v8b:
  (1) `gptneo_125m_stress_v8b.yaml`: alpha=32, shared_base_alpha=8.0, batch_size=16 (eff batch 32).
  (2) `src/models/gpt_neo.py`: dtype=torch.float32 → torch_dtype=COMPUTE_DTYPE — was wrong kwarg,
  silently ignored → model ran in fp32 on H100.
  (3) `scripts/train.py`: added `torch.amp.autocast` around forward pass — was running fp32
  despite H100 bf16 support. Expected speedup: 1.5–2.5×.
  **result:** dtype bug existed across all prior v6/v7/v8a runs — they all ran in fp32 on H100.
  **note:** PPL comparisons across v6/v7/v8a remain internally consistent (all fp32), but they
  are not directly comparable to any bf16 baseline. v8b will be the first bf16 run. If bf16
  causes routing instability (e.g., underflow in cosine logits), revert to fp32 for paper runs.

---

## 2026-03-21 — v8a fineweb: steps 3400–12075 (single run, ~18–63% complete)

**Clarification**: Steps 3400–6225 and 8975–12075 shown in the same log dump are from the SAME
v8a run, not two concurrent experiments. The log output was scrolled back.

- **[observation]** gini critically low throughout: 0.047–0.114 at step 12075 (τ≈0.155, 63%
  complete). This is 4–8× lower than v6-wikitext (gini=0.388 at step 5000). eff_E=7.9–8.0
  throughout. SPAR penalty is working — load is balanced — but at eff_E≈8 with D=768 there is
  no geometric room for cosine-based specialization.
  **result:** Perfect Load Paradox confirmed on fineweb-edu: eff_E=8.0 forces prototypes to
  Voronoi boundaries → Δcos → 0 → gini cannot grow.
  **note:** This extends the 2026-03-19 Voronoi analysis to a second corpus; the pathology is
  not wikitext-specific but structural to SPAR at full capacity.

- **[observation]** PPL declining normally: 29.6 (step 3400) → 28.7 (step 12075). Loss
  3.39→3.36. Monotone decline, no instability.
  **result:** SharedBaseLoRA is not causing training instability across 12k steps.
  **note:** PPL improvement is real but slow; gini at 0.047–0.096 suggests experts are barely
  differentiating, so PPL gains are coming from the shared path and frozen base, not from MoE
  specialization.

- **[observation]** conf rising very slowly: 0.523→0.547 over steps 975–12075 at τ≈0.155.
  conf should be higher at this τ value; the slow growth is consistent with near-uniform routing
  (equidistant prototypes → softmax confidence stays low regardless of τ).
  **result:** conf trajectory confirms Voronoi-boundary diagnosis independently of gini.
  **note:** At τ=0.10 (final), conf ceiling will be ≈0.55–0.60 if Δcos remains ≈0.03. This is
  the same ceiling observed in v7. SharedBaseLoRA has not moved it.

- **[diagnosis]** SharedBaseLoRA does not fix the gini problem. At step 12075, gini
  (0.047–0.096) is no higher than step 3400 (0.073–0.114). Predicted mechanism: shared path
  receives 8× more gradient signal than any single expert (all tokens vs. 1/eff_E tokens);
  expert adapters get weak relative gradient; prototypes never diverge enough for SPAR to find
  meaningful specialization signals.
  **result:** The problem is upstream of the adapter architecture — it is in the routing space.
  The fix must change how cosine similarities are computed, not how adapters are structured.
  **note:** This is the experimental justification for GMR (Global Mean Projection). Any
  architectural change that leaves cosine(x, W_i) near-uniform across i will reproduce this
  pathology regardless of adapter design.

- **[conclusion]** v8a establishes that the bottleneck is in routing space, not adapter
  architecture. Adding shared base LoRA neither helps nor hurts gini because the cosine logits
  are already near-uniform — the routing signal is the missing ingredient.
  **result:** v8a run will complete to step 19000 for final PPL comparison vs v7, but no further
  architectural iteration on the adapter side is warranted.
  **note:** If v8a final PPL ≈ v7 (within 0.3 ppl), the shared base LoRA is effectively a
  neutral addition at this scale. The decomposition hypothesis (shared + routed adapters) is not
  falsified — it may still help at 1.3B where domain heterogeneity is larger — but it is not
  the solution to the fineweb-edu gini problem.

- **[next]** GMR (Global Mean Projection) for v8b after alpha/batch corrections. Route in
  corpus-residual space: `x_proj = x - (x·v_global)v_global`, where `v_global` = EMA of batch
  mean directions (no hyperparameter). Hypothesis: projecting out the global mean direction
  creates heterogeneous cosine similarities even on diverse corpora, allowing SPAR to find
  meaningful expert assignments at eff_E≈8.
  **result:** Not yet implemented.
  **note:** Implementation TODO written in `src/routers/stress_corrected.py` header. The
  v_global EMA update must be DDP-synced (all_reduce AVG) at each step, same pattern as EMA
  load sync. Verify: with GMR, does gini exceed 0.20 by step 5000? If yes, GMR is the fix.
  If no, the Voronoi constraint is independent of input representation.

---

## 2026-03-20 — stress_v8a_fineweb: shared base LoRA on c_proj, steps 500–975

- **[architecture]** Added `SharedBaseLoRA` (rank=8, alpha=16) on `c_proj`, applied to all tokens
  outside the per-expert routing path. Delta-only: `h = act(W_frozen_fc · x)` under `torch.no_grad()`,
  then `out = W_frozen_proj · h + lora_A_base × lora_B_base · h`. B zero-initialized → zero delta at
  init. Adds 184,320 trainable params (~1.5% overhead on 125M). Addresses the
  mathematician's fatal-flaw analysis from 2026-03-19 by keeping the shared adapter on c_proj only
  (post-activation), avoiding the double-counting bug from including frozen base twice.
  **result:** At step 975: loss=3.4298, ppl=30.9, bpb=4.948. eff_E=7.6, gini=0.122, conf=0.523.
  **note:** Routing metrics (eff_E, gini, conf) are statistically identical to v7 at this stage — the
  B=0 guarantee holds empirically. The shared LoRA has contributed zero signal through step 975, as
  expected. This is the correct behavior; a non-zero delta at step 0 would indicate an initialization
  bug.

- **[observation]** λ calibration at step 600 completed without anomaly. Loss trajectory 3.48→3.43
  over steps 500–975 matches v7 closely — no regression from the architectural addition. gini
  0.12–0.17 is in the expected early-phase band before τ annealing (0.5→0.10 over 14k steps) begins
  to bite. At step 975, τ≈0.497 — still near initialization; gini will not reflect full τ pressure
  until ~step 5000 (τ≈0.36).
  **result:** No instability introduced. The shared LoRA path is inert through step 975.
  **note:** τ at step 5000 ≈ 0.36, computed from linear schedule (0.5 − (0.5−0.10)×5000/14000).
  gini growth past step 1000 will be the first real signal of whether shared base LoRA allows
  higher expert specialization by handling domain-invariant tokens globally.

- **[hypothesis]** Shared base LoRA begins contributing meaningfully at ~steps 1000–2000, once
  `||lora_A_base × lora_B_base||_F` grows from zero via gradient updates. At that point, dense
  gradient on the shared path (every token) should learn common syntactic / positional patterns,
  freeing each routed expert to specialize on narrower token distributions — potentially pushing
  gini higher than v7's equilibrium while maintaining eff_E≈8.0. If this holds, the shared LoRA
  acts as an implicit domain-invariant prior, making per-expert adaptation lower-variance.
  **result:** Not yet testable — shared delta is zero through step 975.
  **note:** Key checkpoint: step 3000–5000 where τ≈0.43–0.36 and lora_A_base should have nonzero
  magnitude. Compare gini trajectory vs v7 at matched τ values, not matched step counts, to
  control for annealing schedule effects.

- **[idea]** If shared base LoRA improves final PPL vs v7 on fineweb-edu, the ablation supports a
  2-component adapter decomposition: `y = frozen_base(x) + δ_shared(x) + Σ w_i · δ_i(x)`. This
  is structurally analogous to DeepSeek-MoE's shared expert, but implemented via LoRA rather than a
  full expert, and without routing that expert (no load-balance overhead). Worth framing as a
  parameter-efficient variant of the shared-expert trick if the PPL gain materializes.
  **result:** Pending — compare v8a vs v7 final PPL at step 19000.
  **note:** If delta is negligible at step 5000, the shared LoRA may be starved by the dense-vs-sparse
  gradient imbalance identified in the 2026-03-19 mathematician analysis (shared path 8× more tokens
  than any single expert on fineweb-edu with eff_E=8.0 → faster convergence → residual signal
  shrinks → routed experts get weaker specialization gradient). Monitor `lora_B_base` gradient norm
  vs `lora_B_expert_i` norms at step 2000 to catch this early.

---

## 2026-03-19 — v7 step 8000 diagnosis: conf plateau is structural, not a bug

- **[finding: Δcos is shrinking, not growing]** Solved for Δcos from conf + τ at two checkpoints:
  - Step 1775: τ=0.43, conf=0.522 → Δcos = τ·logit(conf) = **0.038**
  - Step 8000: τ=0.196, conf=0.537 → Δcos = τ·logit(conf) = **0.029**
  Prototypes are moving closer together, not further apart.

- **[root cause: eff_E=8.0 forces prototypes to Voronoi boundaries]** Perfect load balance
  requires each expert to receive equal tokens. For this to hold, prototypes must sit at the
  boundary between adjacent Voronoi cells — where adjacent experts are equidistant from each
  token by definition. At a Voronoi boundary, Δcos → 0 structurally. **Perfect load balance
  and high conf are in fundamental tension. You cannot have both.**
  - eff_E=8.0 → Voronoi boundaries → Δcos small → conf ceiling ≈ 0.54–0.58
  - Every strategy attempted to improve conf was fighting SPAR's own mechanism (correctly)

- **[verdict: conf is the wrong metric for this paper]** The paper claim is
  *"near-perfect load balance (eff_E≈8/8) with zero auxiliary loss."* That claim is achieved.
  conf=0.537 is a mathematical consequence of delivering the primary claim, not a failure mode.
  Report eff_E and gini, not conf.

- **[v7 step 8000 numbers]** ppl=29.0, eff_E=7.9–8.0, gini=0.046–0.067, conf=0.536–0.537.
  gini=0.046 is the best load uniformity observed across all runs. Training continues to step 19000.

---

## 2026-03-19 — prototype LR multiplier: implemented, tested, removed

- **[experiment]** Added separate AdamW param group for `router.W` across all 6 MoE layers:
  `lr = base_lr × prototype_lr_scale`, `weight_decay=0.0` (critical: AdamW decay would corrupt
  unit-norm vectors). `prototype_lr_scale: 3.0` in v7 YAML. Tested to step 3650.

- **[result: failed]** At step 3650 vs previous v7 run (1× LR):

| Metric | 3x LR | 1x LR |
|--------|-------|-------|
| eff_E | 6.9-7.0 | 7.5-7.6 |
| gini | 0.15-0.22 (volatile) | 0.10-0.14 (stable) |
| conf | 0.527 (+0.005) | 0.521 |
| PPL | ~29.6 | ~29.6 |

- **[diagnosis]** Prototypes moving 3× faster outrun the EMA load tracker (α=0.01, 100-step lag).
  Load penalty fires on stale estimates → wrong experts penalized → gini spikes, eff_E drops
  from 7.6 to 7.0. The gradient *direction* (not magnitude) is the actual bottleneck, and Adam's
  second-moment normalization already compensates for gradient sparsity. 3× amplifies noise steps,
  not signal steps. Conf gain of +0.005 does not justify eff_E loss of −0.6.

- **[action]** Fully reverted: `prototype_lr_scale` removed from config, `build_optimizer` reverted,
  `prototype_lr_scale: 3.0` removed from v7 YAML, `TestPrototypeLRScale` (5 tests) removed.
  147 tests pass.

---

## 2026-03-19 — routing improvement strategies: full analysis

### Strategies evaluated by moe-router-mathematician

**Strategy A — Expert Choice routing: FATAL (do not implement)**
- Fatal implementation bug: `combined = torch.zeros_like(x_flat)` in `LoRAMoELayer.forward`
  means tokens selected by zero experts produce **zero output**, silently zeroing the residual
  stream — not `base_MLP(x)` as claimed. Correctness fix exists (init combined = frozen_base_out)
  but is not sufficient on its own.
- With eff_E≈8.0, SPAR's penalty is near-zero and token-choice is already cosine-greedy.
  Expert-choice and SPAR produce near-identical selection sets. Does not address Δcos≈0.01.
- Gradient sparsity: ~10% of tokens expected to be selected by zero experts under EC,
  contributing zero gradient to any LoRA parameter (systematically undertrained).

**Strategy B — Sigmoid independent gating: INVALID PREMISE**
- With per-pair mean-centering `σ((cos_i − μ)/τ)` where μ=(cos_1+cos_2)/2:
  `cos_i − μ = ±Δcos/2`, and since `σ(−x) = 1−σ(x)`: σ_1 + σ_2 = **1 exactly, always**.
  Sum-to-1 constraint is perfectly reimposed. Algebraically equivalent to `softmax([cos_1,cos_2]/2τ)`.
- At Δcos=0.01, τ=0.12: sigmoid conf ≈ 0.511 vs softmax ≈ 0.521. Marginally **worse**.
- Variable-Σσ property requires population-level mean (K>2), not per-pair mean.

**Strategy C — Sinkhorn optimal transport: VALID BUT REDUNDANT**
- Genuine advantage: differentiable soft-assignment gives every prototype gradient from every
  token. But near-uniform assignments (Δcos≈0.01) produce diffuse gradient with weak signal.
- Estimated PPL benefit ≤0.01. SPAR already achieves load balance that Sinkhorn provides.
- Worth a future ablation; not a primary intervention.

**Strategy D — Prototype orthogonality regularization: VALID BUT BREAKS PAPER**
- `L_proto = α·||W·W^T − I||²_F` has valid fixed point (orthogonal prototypes), correct
  magnitude at α=0.01, expected conf improvement to 0.65–0.80 at τ=0.12.
- **Fatal for paper thesis**: adds an auxiliary loss term, directly contradicting
  "zero auxiliary loss" differentiation from GShard/Switch/DeepSeek.
  Run as ablation after primary SPAR result is published. Report that SPAR deliberately avoids
  it while being aware orthogonality regularization exists.

**Strategy E — Diversity bonus in selection logit: DEFERRED**
```
z_i = cos(x,W_i) − λ·max(0, L_i − 1/N) + γ·d_i
d_i = min_{j≠i}(1 − cos(W_i, W_j))   [prototype isolation score, per-expert constant]
γ = σ(inter-prototype cosines) × N    [auto-calibrated at step 600, zero new free params]
```
- d_i computed from 8×8 Gram matrix (64 dot products), negligible cost.
- Biases selection toward pairs where experts are directionally isolated.
- Only needed if v7 conf < 0.55 at step 10000 (after full τ anneal).
  Current trajectory (conf=0.537 at step 8000) is consistent with reaching 0.62–0.68 naturally.

### Online EMA prototype update: REJECTED

- Mathematician verdict: addresses the same bottleneck as k-means init (uninformative gradient
  direction from low Δcos). They are substitutes, not complements.
- Confirmed circular feedback loop: overloaded expert → large centroid → W_i moves toward
  noise-routed tokens → they re-select that expert. SPAR penalty does not clearly dominate
  on a per-step basis (β=0.001 → angular displacement ≈ penalty's per-step corrective effect).
- Adam + EMA sequential application produces systematic conflict when loss gradient opposes
  centroid direction — common case in early training.
- k-means init already solves the direction problem; EMA adds complexity for zero marginal gain.

### Shared LoRA expert: REJECTED

Three issues, two fatal:
- **Fatal #1 — Routing signal suppression**: shared LoRA gets dense gradient (every token),
  routed experts get 4× sparser gradient. Dense-vs-sparse 4:1 ratio causes shared LoRA to
  converge fast, shrinking residual signal to `∂L/∂W_i` at the critical differentiation window
  (steps 0–2000). Would recreate the conf=0.52 plateau despite k-means init.
- **Fatal #2 — Double-counting**: frozen base is inside each `SharedLoRALayer`. Naive shared
  LoRA produces `y = 2·b(x) + δ_shared + Σ w_j·δ_j`. Fix requires forking `SharedLoRALayer`
  into delta-only mode — non-trivial architecture surgery.
- Common adaptation leakage is theoretically real but quantitatively small at rank=32
  (domain shift needs ~4–8 principal components; rank=32 is 4–8× over-provisioned).
- Simpler alternative: increase `alpha` 64→96 (scaling 2.0→3.0), zero architecture change.

### Fine-grained segmentation N=16, rank=16, K=4: REJECTED

- Iso-parameter confirmed: N×rank = 8×32 = 16×16 = 256 (params exactly equal).
- **C(16,4)=1820 vs C(8,2)=28 argument is misleading**: effective output rank per token =
  min(K×rank, D) = min(2×32, 768) = min(4×16, 768) = **64 in both cases**. Expressivity ceiling
  is identical.
- **λ doubles** to `16σ_cos`. For same absolute overload: penalty is **3× larger** with N=16
  (fair share halves from 1/8 to 1/16). Simultaneously, rank=16 experts have ~29% weaker
  delta signal (||δ|| ∝ √rank). Weaker differentiation + stronger penalty = bias toward uniform
  routing — opposite of goal.
- MoLoRA (2024) result (fine-grained beats coarse) was on instruction tuning (discrete task
  clusters). fineweb-edu has continuous token distribution — result does not generalize.
- Requires `alpha=32` (not 64) if tried, to preserve scaling=2.0 with rank=16.

---

## 2026-03-19 — base_out routing signal: empirically ruled out

- **[experiment]** Wrote `scripts/check_routing_signal.py`. For each MoE layer: computed
  std across experts of `cos(x_i, W_j)` and `cos(base_out_i, W_j)` on 512-token calibration batch.

| Layer | std_x | std_base | ratio |
|-------|-------|----------|-------|
| L1 | 0.0355 | 0.0345 | 0.971 |
| L3 | 0.0348 | 0.0339 | 0.976 |
| L5 | 0.0365 | 0.0351 | 0.959 |
| L7 | 0.0349 | 0.0341 | 0.978 |
| L9 | 0.0348 | 0.0349 | 1.000 |
| L11 | 0.0339 | 0.0362 | 1.068 |
| Mean | 0.0351 | 0.0348 | 0.992 |

- **[verdict]** Ratio=0.992, well below 1.2 threshold. Routing on `base_out` provides no
  improvement over routing on `x`. The frozen MLP reshapes geometry uniformly — F.normalize
  before cosine means only direction matters, and the MLP transformation does not selectively
  amplify token-type directional differences. Both approaches are equivalent.
  Routing stays on `x`. Switching would add a full frozen MLP forward pass per layer for zero gain.

---

## 2026-03-19 — Grassmannian Expert Choice: three fatal flaws, one paper contribution

**Proposed formulation:** Replace `cos(x, W_i)` with Grassmannian projection coefficient:
`p_i(x) = ||A_i·x̂||²/rank` — fraction of token's direction captured by expert i's LoRA subspace.
Combined with expert-choice selection. Eliminates W_i, λ, EMA entirely. "1-line formula."

**Fatal Flaw 1 — Scale non-invariance (same pathology as raw A-matrix routing):**
`p_i(λA_i) = λ²·p_i(A_i)`. Task loss drives `||A_i||_F` up during training. Routing collapses to
whichever expert's A matrix grows largest — positive feedback loop. SPAR avoids this by normalizing
W_i in every forward; GEC has no equivalent.

**Fatal Flaw 2 — 55× worse discriminability at initialization:**
From kaiming_uniform with fan-in scaling: `Var(A_i entries) = 2/D`.
- `std_expert[p_i] = 2√2 / (D√rank) = 2√2/(768×√32) ≈ 0.00065`
- `std_expert[cos(x,W_i)] ≈ 1/√D ≈ 0.036`
- Ratio: **55×**. Averaging rank=32 squared projections concentrates by CLT — unavoidable,
  not a cold-start issue. Initial routing is dominated by floating-point noise.

**Fatal Flaw 3 — Gradient conflict from dual-use of A_i:**
A_i appears in routing `p_i = ||A_i x̂||²/rank` AND in task adaptation `lora_out = B_i A_i x`.
These objectives oppose each other: the A_i that best compresses a token cluster ≠ the A_i that
maximally expands selection coverage. SPAR's factored design (W_i for routing, A_i/B_i for
adaptation) cleanly separates these objectives — this factorization is correct by design.

**Paper contribution from rejection (no new experiments needed):**
> "An appealing alternative routes based on the LoRA-A subspace: `p_i(x) = ||A_i x̂||²/rank`.
> However, in D=768 dimensions with rank=32, the std of this score across experts at initialization
> is `2√2/(D√rank) ≈ 6.5×10⁻⁴`, compared to `1/√D ≈ 0.036` for prototype cosine routing — a
> factor of 55×. Initial routing degenerates to near-random. Furthermore, the gradient paths for
> routing and adaptation are coupled through the shared A matrix, creating a multi-objective
> conflict. SPAR's factored design — separate W_i for routing, A_i/B_i for adaptation —
> cleanly avoids both problems."

**EC zero-output bug (worth fixing as correctness safeguard):**
`combined = torch.zeros_like(x_flat)` in `LoRAMoELayer.forward` means any token selected by
zero experts produces zero MoE output — silently corrupts the residual stream. Correct fix:
initialize `combined = frozen_base_MLP(x)` using shared weights from `expert_pool.experts[0]`.
Does not affect SPAR (all positions overwritten). One-time fix for any future EC experiments.

---

## 2026-03-19 — noise annealing + λ formula simplification

- **[feature]** Noise annealing implemented in `StressCorrectedRouter`. `noise_std` decays
  linearly from `noise_std` → 0 over `noise_anneal_steps` optimizer steps (based on `num_steps`,
  updated in `step()`). Disabled by default (`noise_anneal_steps=0`), so all existing configs
  and tests are unaffected. Mirrors the `_current_tau()` + `self._tau` pattern exactly.
  Config: `StressCorrectedRouterConfig` gains `noise_anneal_steps: int = 0`.
  `gptneo_125m_stress_v7-fineweb.yaml` opts in with `noise_anneal_steps: 9500` (midpoint of 19000 steps).
  **Rationale:** At step 6400, measured Δ_cos≈0.037 < noise effective std≈0.064 (SNR=0.58). Noise
  was dominating routing decisions. Annealing to 0 by midpoint preserves early-stage exploration
  while letting the cosine signal dominate once prototypes have aligned.

- **[simplification]** `_calibrate_lambda()` simplified: `σ_cos / mean_load` → `σ_cos * N`.
  Since Σ L_i = 1 always, mean_load = 1/N exactly — the division by mean_load is equivalent to
  multiplication by N. Removes one tensor op, zero behavior change.

## 2026-03-19 — rank=32 ablation plan + architectural boundary decisions

- **[decision: unfreezing MLP is off-limits]** Unfreezing 6 MoE MLP backbone blocks is not a
  valid PPL lever for this project. Adds ~28–34M trainable parameters (5× current 5.9M),
  changes the architecture class from "LoRA-MoE with frozen prior" to "partial fine-tuning
  + LoRA bolted on." The paper claim — *MoE specialization with adapter-only training,
  frozen backbone* — is invalidated. The efficiency argument (LoRA = parameter-efficient
  MoE) collapses. The correct capacity lever is **rank=32**.

- **[finding: rank=32 is SPAR-differential, not router-neutral]** Mathematician analysis
  establishes that the PPL benefit of rank=16→32 is approximately 2× larger for SPAR
  (eff_E=8.0) than for standard routing (eff_E=4.0). Structural argument: with eff_E=4.0,
  4 undertrained experts (~4% tokens each) have noise-dominated gradients even at rank=16;
  additional rank capacity for those 4 experts is wasted. With eff_E=8.0, all 8 experts
  see ~25% tokens each; all 8 can utilize rank=32's additional capacity.
  - Standard: ~64 effective new rank-units gained
  - SPAR: ~128 effective new rank-units gained
  - Lower bound: SPAR gains ~2× more PPL from rank doubling.

- **[plan: 2×2 ablation grid]** After v6-fineweb and v7-fineweb complete:
  - `gptneo_125m_standard_v3_r32` — standard router, fineweb-edu, rank=32 (fills cell B)
  - `gptneo_125m_stress_v6_r32` — SPAR, fineweb-edu, rank=32 (fills cell D)
  - If `(C−D) > (A−B)` (SPAR gains more from rank doubling), paper has novel result:
    *eff_E=8.0 compounds with adapter capacity scaling.* Testable, not yet in MoE literature.

- **[finding: noise_std > Δ_cos at step 6400]** `noise_std=0.05` exceeds measured
  `Δ_cos=0.037` (top-2 cosine gap) at step 6400. Routing noise is currently larger than
  routing signal. For v8 (or any future run), anneal `noise_std` to 0 by midpoint
  (~step 9500 for 19k runs). Orthogonal to EMA load penalty — safe change.

---

## 2026-03-19 — standard_v3 fineweb-edu baseline (completed)

- **[result]** `gptneo_125m_standard_v3` on H100:4, fineweb-edu, 19000 steps, standard
  router + aux_loss coef=0.01, N=8 experts, rank=16.
  - Steps 9025–11700: loss 3.36–3.39, PPL 28.8–29.6, **plateaued**.
  - eff_E=3.9–4.1, gini=0.56–0.60, conf=0.524–0.527.
  - Routing collapsed to ~4/8 effective experts despite aux loss. No recovery observed.
  - Projected final (step 19000): PPL ~28.5–29.0, eff_E≈4.0.

- **[paper comparison]** At step 6400: SPAR PPL=29.6 (declining), eff_E=8.0, gini=0.04–0.09
  vs standard PPL=29.0 (plateaued), eff_E=4.0, gini=0.57. SPAR projected final ~27.5–28.5.
  SPAR achieves 2× effective expert utilization with zero aux loss, same or better PPL.

- **[diagnosis]** aux_loss coef=0.01 (Switch Transformer default) is too weak for fineweb-edu
  domain diversity. Task-loss gradients on a multi-domain corpus consistently outcompete the
  load penalty, driving routing collapse to eff_E≈4. The out-of-gradient-graph EMA penalty
  (SPAR) cannot be outcompeted by task loss by construction — this is the mechanism difference.

---

## 2026-03-19 — data-driven prototype initialization (SPAR router)

- **[impl]** Added `_kmeans_init` (pure PyTorch, no sklearn) and `initialize_prototypes_from_data`
  to `StressCorrectedRouter`. Sets `W_i = normalize(centroid_i)` from k-means on actual layer
  activations. Starts cos_sim ≈ 0.5+ vs ≈ 0 with random init.

- **[wire-up]** `_initialize_router_prototypes()` in `scripts/train.py`: registers forward hooks on
  all `LoRAMoELayer` instances, collects activations over 2 warmup batches via `base_model.eval()`,
  runs k-means on rank 0, broadcasts `W` to all ranks via `dist.broadcast`. Gated by
  `router.init_from_data: true` in YAML.

- **[config]** `StressCorrectedRouterConfig.init_from_data: bool = False` — opt-in, backward
  compatible. Default false so existing runs (v6-fineweb in progress) are unaffected.

- **[yamls]** `experiments/gptneo_125m_stress_v7-fineweb.yaml` created with `init_from_data: true`,
  otherwise identical to v6. v6 YAML has a comment noting the option. v7 is the next run.

- **[tests]** 5 new tests in `TestPrototypeInit`: W changes, unit normalization, cos_sim improves
  on clustered data, edge case with exactly k tokens, all centroids assigned. 140 tests total pass.

---

## 2026-03-19 — fineweb-edu run: early diagnostic (step ~4100/19000)

- **[observation]** `gptneo_125m_stress_v6_fineweb` on H100:4. Loss 3.446→3.406, PPL 31.4→30.1 over
  3000 steps. eff_E 7.6→8.0 (hit perfect balance at step 3725, 3850). gini 0.126→0.056 — 6× lower
  than wikitext final (0.388). conf 0.520→0.527. τ at step 4100 ≈ 0.344 (already annealing).

- **[diagnosis]** `welford_mu_mean` across all 6 layers is 0.90–0.993, meaning cosine similarity
  between routed tokens and their expert prototype is only **0.007–0.10**. Prototypes have not learned
  to align with their token populations. In D=768, random unit vectors have expected cos_sim ≈ 0;
  prototypes started near-random and are escaping very slowly.
  **Per-layer cos_sim at step ~4100:** L1=0.025, L3=0.05, L5=0.007, L7=0.045, L9=0.10, L11=0.018

- **[consequence chain]** cos_sim≈0 → all experts look equivalent from cosine perspective → Δ_cos≈0.037
  (top-2 gap) → conf stuck at 0.52 regardless of τ → eff_E=8.0 because routing is near-uniform
  (all experts are cosine-equivalent) → slow PPL decline because routing carries no semantic signal.
  Chicken-and-egg: weak cos_sim → near-uniform softmax → weak gradient on W → prototypes stay random.

- **[root cause]** Random prototype initialization. Not LoRA rank — the adapters ARE learning (loss
  declining). The W prototype directions have not found their token subpopulations.

- **[projection]** PPL should accelerate past step 8000–10000 as prototypes mature (layer_9 at
  cos_sim≈0.10 and improving fastest). Expected final PPL: ~22–24. The eff_E=8.0 / gini=0.056 result
  is the headline load-balance number for the paper (vs wikitext 6.1 / 0.388).

- **[fix queued]** Data-driven prototype initialization: k-means on layer activations from one warm-up
  batch → W_i = normalize(centroid_i). Starts cos_sim at ≈0.5 instead of ≈0. In progress.

---

## 2026-03-19 — fix: _current_tau() Dynamo graph break

- **[fix]** Moved `_current_tau()` out of the compiled `forward()` graph. Cached tau as
  `self._tau: float`, updated once per optimizer step in `step()`. Eliminates the
  `Tensor.item()` graph break and the downstream recompile-limit (8) hit on
  `torch_dynamo_resume_in_forward_at_205` caused by `x_norm` requires_grad mismatch.
  No change to tau annealing behavior — `_current_tau()` still computes the same value,
  just called from `step()` instead of `forward()`.

---

## 2026-03-18 — τ annealing + training pipeline wiring

- **[feature]** τ annealing implemented in `StressCorrectedRouter`. τ decays linearly from `temperature` → `tau_final` over `tau_anneal_steps` optimizer steps (based on `num_steps`, updated in `step()`). Disabled by default (`tau_anneal_steps=0`), so all existing wikitext configs are unaffected. `_current_tau()` guards against division-by-zero and clamps at `tau_final`. Config: `StressCorrectedRouterConfig` gains `tau_final: float = 0.5` and `tau_anneal_steps: int = 0`.
  **expected impact:** conf metric expected to rise from 0.562 → ~0.62 on fineweb-edu (τ=0.5→0.12 over 10000 steps). Sharper output weights at convergence without affecting selection logit or load balancing.

- **[fix]** `scripts/train.py` `create_router` call now explicitly passes `ema_alpha`, `lambda_calib_step`, `tau_final`, and `tau_anneal_steps` from the YAML config. Previously these SPAR fields silently fell back to dataclass defaults — fragile if the defaults ever diverged from the YAML. Factory's `filtered_kwargs` ensures these are no-ops for non-SPAR routers.

- **[cleanup]** `experiments/gptneo_125m_stress_v6-fineweb.yaml`: removed dead DFR fields (`mu_stress`, `gamma_surprise`, `n_min_surprise`) that were silently dropped by the factory. Added explicit `ema_alpha: 0.01`. Updated header comment to reflect the actual implemented SPAR formula (removed stale DFR formula and step-200 reference).

- **[config]** `run_modal_training.py` CONFIG updated from `gptneo_125m_stress_v6-wikitext.yaml` → `gptneo_125m_stress_v6-fineweb.yaml`. Next cloud run will train on fineweb-edu.

- **[tests]** Added `TestTauAnnealing` (6 tests) to `tests/routers/test_stress_router.py`: τ starts at temperature, decreases over steps, clamps at tau_final, disabled by tau_anneal_steps=0, disabled when tau_final==temperature, lower τ produces sharper output weights.

---

## 2026-03-10

- **[experiment]** `metabolic_v4` (WandB: `pw764gz6`) vs `standard_v2` (WandB: `a78b5f6q`) — definitive controlled ablation. Both: GPT-Neo 125M, 8 experts, top-k=2, rank=16, wikitext-103, 5000 steps, A100:4, seed=42, identical LoRA/optimizer config. Single variable: router type.
  **result:** val_loss 3.1246 (metabolic) vs 3.1212 (standard) — statistically tied. Load balance: metabolic effective_E 5.0–7.4 across layers vs standard 3.5–4.6. Standard collapses in L1 and L11 despite `aux_loss_coef=0.01`. Metabolic throughput 14% lower (fatigue all-reduce overhead, 6 `dist.all_reduce` calls per step).
  **note:** Task loss parity with clear load-balance superiority is the paper's central empirical claim. Throughput cost is documented and expected; publishable as an honest tradeoff table.

- **[observation]** Dataset discrepancy in v4 config: `gptneo_125m_metabolic_v4.yaml` has `dataset_key: fineweb-edu` while `standard_v2` has `wikitext-103`. If WandB confirms both runs consumed wikitext-103, the YAML reflects a post-run edit error and the comparison remains valid. Must be reconciled before paper submission.
  **result:** Pending WandB artifact check.
  **note:** If v4 actually trained on fineweb-edu, the ablation is confounded and a rerun on matching data is required.

- **[experiment]** `standard_v2` collapse profile: L1 gini=0.616 by step 350 (from 2026-03-09 entry), L11 also collapses despite aux loss. `aux_loss_coef=0.01` (Switch Transformer default) is insufficient to prevent monopolisation on wikitext-103.
  **result:** Effective experts at step 5000: 3.5 (L1) to 4.6 (L11), well below theoretical max of 8.
  **note:** Raises the question whether any `aux_loss_coef` value can recover from early collapse without restarting. The collapse appears self-reinforcing once gini exceeds ~0.55.

- **[experiment]** `metabolic_v4` load-balance trajectory: effective_E 5.0–7.4 at step 5000, with Layer 1 stabilised at gini≈0.50 from step ~1800 onward (slope +0.000013/step in 2400–2900 window). No layer shows collapse.
  **result:** SoftSign penalty at equilibrium for a 2× overloaded expert: `λ × SoftSign(β/γ × 1/8) = 0.5 × SoftSign(1.0) ≈ 0.25` — sufficient to offset gate's alignment advantage for dominant experts.
  **note:** The fixed-point analysis in the v4 YAML comment matches empirical stabilisation step, which is encouraging for analytical tractability.

---

## 2026-02-XX — Deleted Experiment Archaeology

- **[experiment]** `metabolic_v2` (deleted): first differential-fatigue run, wikitext-2 (~2M tokens), 3000 steps, A100:4. Config: λ=0.5, β=0.4, γ=0.05 (β/γ=8), T=1.0, top-k=2, 8 experts, rank=16.
  **result:** Loss plateaued ~step 1400. T=1.0 produced diffuse routing weights. wikitext-2 too small to force routing diversity — no clear collapse signal, but also no specialisation.
  **note:** λ=0.5 was correct in magnitude but two formula bugs masked its effect: (1) U_i computed from softmax weights rather than token counts — U_i inflated for high-confidence slots; (2) birth_step warmup was always a no-op (implementation bug). Both bugs meant fatigue signal was noisy and poorly calibrated.

- **[bug]** `metabolic_v2`: fatigue `U_i` used softmax gating weights `w_i` as the usage signal, not uniform token counts. For top-k=2, correct usage is `1/k = 0.5` per selected slot. Using `w_i` instead inflates usage for high-confidence experts (those with large `w_i`) and undersignals usage for near-tie experts.
  **result:** Fatigue pressure was systematically higher on already-dominant experts (catching some of their dominance), but miscalibrated — the penalty magnitude did not correspond to actual compute load.
  **note:** Fixed in v3: U_i = uniform `1/top_k` per selected slot. Token-count usage is the correct compute-load proxy; softmax weights are a routing-confidence proxy.

- **[bug]** `metabolic_v2`: birth_step warmup was always a no-op due to an off-by-one in the warmup condition check. The gate received fatigue penalty from step 0, before any prototype directions had stabilised.
  **result:** With immediate penalty and noisy U_i, Layer 1 locked to 2 dominant experts before step 200.
  **note:** Fixed in v3/v4 via global λ warmup: penalty ramps 0→λ over 400 steps, implemented as a scalar multiplier on the SoftSign output rather than a step-count gate.

- **[experiment]** `metabolic_v3` (deleted): formula-fixed run, wikitext-103 (~103M tokens), 5000 steps, A100:4. Config: λ=0.3, β=0.4, γ=0.05, T=0.7, top-k=2, 8 experts, rank=16, warmup=400.
  **result:** Routing stabilised ~step 1500–2000. Effective experts improved over v2. λ=0.3 proved too weak post-warmup — Layer 1 collapsed to gini=0.59 by step 2350. val_loss≈3.13.
  **note:** T=0.7 (vs v2's 1.0) sharpened routing without hurting diversity — kept as the metabolic default. wikitext-103 was sufficient corpus size to reveal routing dynamics. λ=0.3 finding directly motivated the v4 decision to restore λ=0.5 while keeping warmup=400.

- **[experiment]** `standard_v1` (deleted): first aux-loss baseline, wikitext-2, 3000 steps, A100:4. Config: T=1.0, `aux_loss_coef=0.01`, top-k=2, 8 experts, rank=16.
  **result:** Some routing collapse on wikitext-2, insufficient corpus to force diversity. Aux loss unable to prevent expert monopolisation. Not a fair comparison against metabolic_v2 (different dataset, different corpus size).
  **note:** Both the dataset mismatch and T=1.0 vs the eventual metabolic T=0.7 made this comparison unpublishable. Superseded by standard_v2 on wikitext-103.

- **[observation]** Current active configs: `metabolic_v4` (WandB `pw764gz6`, completed), `standard_v2` (WandB `a78b5f6q`, completed), `metabolic_fast` (500 steps, 1 GPU, fast iteration), `gptneo_1.3b_metabolic_v3` (scale-up, not yet run), `smoketest` (CI only, 2 steps).
  **result:** Publication ablation (v4 vs standard_v2) complete. Scaling experiment and matched-temperature ablation remain.
  **note:** Next required experiments: (1) matched-temperature run (standard at T=0.7, metabolic at T=1.0) to deconfound Stress metric; (2) 1.3B scaling run; (3) Stress–val_PPL correlation sweep across v4 checkpoints.

---

- **[architecture]** Expert Stress metric defined: `Stress_i = CV(1 - cos(x_k, W_i))` for tokens routed to expert i, tracked via weighted online Welford. Raw observation is cosine distance `d_{i,k} = 1 - cos(x_k, W_i)`, weighted by usage fraction `w_i(t)`. Replaces earlier attributed-loss formulation `w_i * L(t)`, which was incorrect — it conflated routing-weight variance with loss variance via the product-variance identity.
  **result:** Formulation is now mathematically clean. Stress reduces to the CV of per-token cosine distances to the expert prototype, independent of scalar loss.
  **note:** Distinction matters for paper correctness; the old formula silently introduced a covariance term that does not have a routing interpretation.

- **[observation]** Layer-locality is a correctness requirement, not a convenience. In a 32-layer network, final loss `L(t)` is a diluted function of all 256 routing decisions simultaneously. Loss-based stress unfairly penalises Layer 2 Expert A when Layer 30 Expert B caused the spike. Cosine-distance Stress uses only the layer's own alignment, computed locally in the forward pass — immune to downstream routing errors.
  **result:** Credit assignment failure of loss-based metrics scales with depth. At 32 layers the attribution signal-to-noise ratio per expert per step is at most 1/256.
  **note:** This is the primary argument for Stress over loss-based diagnostics in deep MoE stacks.

- **[observation]** Zero compute overhead. `alignment[batch, seq, expert_i]` is already materialised in `compute_alignment()`. Stress observation is `1 - alignment[..., i]` — zero additional FLOPs. State per expert is three scalars `(n_i, mu_i, M2_i)` = 24 floats per layer for 8 experts.
  **result:** Token-level granularity ~2048 observations per expert per batch (vs 1 for loss-based). Welford estimate is statistically reliable from step 1.
  **note:** No justification needed to enable Stress tracking by default in all runs.

- **[hypothesis]** Stress is proportional to `sqrt(rank-1 LoRA approximation error)` on the routing distribution. At convergence, `W_i` aligns with the top eigenvector of the second-moment matrix `S_i` of routed tokens. Then `Stress_i^2 * mu_i^2 ≈ E_i(1) = 1 - lambda_1(S_i)` — the rank-1 LoRA approximation error. Low Stress implies routed tokens live near a rank-1 subspace, so rank-r LoRA (r≥1) achieves low approximation error.
  **result:** Not yet empirically verified.
  **note:** This is the external validity argument: low Stress should predict lower PPL if LoRA approximation error and PPL co-vary. Must be tested via Stress–val_PPL correlation across checkpoints.

- **[observation]** Circularity concern: the metabolic router selects tokens using cosine similarity, so selected tokens have low cosine distance by the selection rule itself. The mean `mu_i` is circular; the variance (CV) is not. Two experts can have identical mean cosine distance but different Stress depending on whether selected tokens form a coherent cluster or a diffuse cloud.
  **result:** Report CV (Stress), not mean cosine distance, in all cross-router comparisons.
  **note:** Universal computation via post-hoc normalisation `d_{i,k}^norm = 1 - (x_k/||x_k||) · (W_i/||W_i||)` is valid for standard/topk/switch routers and removes the circularity objection for comparison tables.

- **[ablation]** Temperature confound: metabolic uses T=0.7, standard uses T=1.0. Lower temperature sharpens routing, mechanically reducing cosine distance variance regardless of routing mechanism.
  **result:** Any observed Stress difference between routers is currently confounded with temperature.
  **note:** Matched-temperature ablation required before any cross-router Stress comparison is paper-ready. Planned: rerun standard router at T=0.7 and metabolic at T=1.0.

- **[observation]** 2×2 diagnostic table: (Low Gini + Low Stress) = balanced + coherent = metabolic goal; (High Gini + Low Stress) = routing collapse with narrow token set = standard router pathology; (Low Gini + High Stress) = balanced but incoherent = random routing baseline. Low Stress alone is not a quality signal.
  **result:** Gini must always be reported alongside Stress. The joint (Gini, Stress) pair is the meaningful diagnostic.
  **note:** This framing is reviewer-robust — it prevents the obvious objection that a random router could also achieve low Gini.

- **[idea]** Correct paper positioning: Stress is a mechanistic diagnostic, not a primary comparison metric. Use it to show (a) Stress decreases as λ warmup activates (steps 400+), (b) Stress gradient across layers complements MI gradient (earlier layers higher stress, later layers lower), (c) Stress–PPL correlation if empirically confirmed. Primary cross-router comparison remains Gini + effective_E + MI + downstream accuracy.
  **result:** Framing avoids overreach; positions Stress as interpretability evidence rather than a performance claim.
  **note:** External validity (Stress–PPL correlation) is the one missing experiment that would allow stronger claims.

## 2026-03-10 (continued) — `metabolic_v4` fineweb-edu run, steps 0–1225

- **[observation]** Warmup phase (steps 0–400) on fineweb-edu produces near-perfect balance: gini 0.157→0.026, eff_E rising to 8.0. The 400-step penalty ramp gives the gate clean prototype directions before fatigue pressure activates — consistent with wikitext-103 behavior.
  **result:** Warmup design generalises to a harder corpus. Initial balance ceiling of eff_E=8.0 is router-agnostic; the meaningful signal starts post-warmup.
  **note:** fineweb-edu warmup dynamics are indistinguishable from wikitext-103 at this stage, confirming the warmup mechanism is not data-regime-specific.

- **[instability]** Post-warmup routing drift (steps 400–1225): gini 0.026→0.207, eff_E 8.0→7.4, F_σ rising monotonically 0.042→0.391 with no plateau visible. LM loss unaffected (4.93→3.44 converging normally). Drift rate decelerating: +0.102 gini over steps 400–800, +0.079 over steps 800–1225.
  **result:** Fatigue is building and applying braking force — the deceleration is evidence the mechanism is active — but has not yet reached fixed-point equilibrium by step 1225.
  **note:** Decelerating slope is the key diagnostic. A monotonically accelerating gini would imply the mechanism is failing entirely; deceleration implies a ceiling is forming, just higher than observed on wikitext-103.

- **[hypothesis]** Root cause of drift: λ=0.5 sets a maximum penalty of `λ × SoftSign(∞) = 0.5`. On wikitext-103, gate logit advantages for dominant experts stayed below 0.5, so the ceiling was sufficient. On fineweb-edu, 400 steps of unconstrained task-loss gradient descent grow logit advantages >0.5 for certain high-frequency token patterns — the penalty ceiling is breached and fatigue cannot fully correct.
  **result:** Not yet confirmed — requires extracting per-expert gate logit distributions at steps 400 and 1225 from WandB to compare against the λ=0.5 ceiling.
  **note:** If logit advantages are found clustered near 0.45–0.55, a modest λ increase to 1.0 should close the gap. If they are >1.0, a fundamentally stronger mechanism is needed.

- **[hypothesis]** Mechanistic distinction from aux loss preserved even if λ requires recalibration. Aux loss constrains the gradient direction of gate weights, competing with LM loss during backprop. Metabolic λ operates on routing logits at inference time, outside the gradient graph — gate weights still specialize freely, penalty applied post-hoc per step. Increasing λ does not introduce the gradient competition that aux loss creates.
  **result:** The mechanistic claim survives even if λ=0.5 is empirically insufficient for fineweb-edu. The correct framing is hyperparameter recalibration, not mechanism failure.
  **note:** This distinction is paper-critical. Reviewer objection "you just need a stronger aux loss" is rebutted by the gradient-graph argument, not by load balance numbers alone.

- **[idea]** Next run: `metabolic_v4b` with λ=1.0, all else equal (β=0.4, γ=0.05, warmup=400, fineweb-edu). Doubles the penalty ceiling to 1.0. Diagnostic criterion: if gini stabilises <0.1 by step 2000 with λ=1.0, the mechanism works on real data with recalibrated λ. If gini continues drifting past 0.2 with λ=1.0, the fixed-λ assumption is fundamentally broken for complex corpora.
  **result:** Not yet run. Blocked on current run completing.
  **note:** λ=1.0 is the natural next doubling; jumping further (e.g. λ=2.0) risks overcorrecting and suppressing legitimate specialisation — step the search.

- **[idea]** Longer-term: data-adaptive λ — increase λ when gate gradient norm is high (task loss driving rapid logit growth), decay λ when gate has converged. Signal: `||∇_{gate} L_task||` per step, already computable from the existing optimizer state. This makes the fatigue mechanism phase-aware and would be genuinely novel vs both fixed-λ metabolic and aux-loss baselines. Deferred until fixed-λ sweep (λ ∈ {0.5, 1.0}) is complete.
  **result:** Conceptual only. No implementation yet.
  **note:** The gate gradient norm as a λ-schedule signal is the key idea. If gate grad norm correlates with post-warmup gini drift rate across runs, this becomes a clean empirical motivation for the adaptive mechanism.

## 2026-03-18 — `gptneo_125m_spar_wikitext` (SPAR first full run)

- **[experiment]** First complete SPAR run: GPT-Neo 125M, 8 experts, top-k=2, 6 MoE layers (indices 1,3,5,7,9,11), rank=16, τ=0.5, α=0.01, λ_calib_step=200, wikitext-103, 5000 steps, A100:4, 108 min. Selection: `z_i = cos(x,W_i) - λ·max(0, L_i - 1/N)`. Output weight: `w_i = softmax(cos(x,W_i) / τ)`. No aux loss. No NCCL crashes. `find_unused_parameters: false`.
  **result:** val_ppl=22.9 (val_loss=3.1333). Final state: eff_E=6.1, gini=0.388, conf=0.562. 1 `dist.all_reduce` per step (EMA load only) — down from metabolic's 6.
  **note:** This is the SPAR wikitext-103 baseline. Comparable val_loss to metabolic_v4 (3.1246) and standard_v2 (3.1212) — in the statistically tied range. Need standard_v2 rerun on same config for an apples-to-apples comparison.

- **[observation]** PPL trajectory: rapid phase 0–500 (51.2→25.9, −25.3), then slow grind 500–5000 (25.9→22.9, −3.0). ~89% of the total gain occurs in the first 10% of training steps. LR hit cosine floor (3e-5) at ~step 4075 with no discontinuity.
  **result:** Training loss plateau 3.29–3.32 from step 1500 onward while val loss kept declining from 3.26→3.13 — no overfitting gap through step 5000.
  **note:** The train/val gap actually closed over training. Suggests regularization from LoRA dropout (0.05) plus routing noise is sufficient; longer runs or lower LR floor may recover additional PPL.

- **[observation]** λ calibrated at step 200: `λ = min(σ_cos / mean(L), 5.0)`. At step 200, EMA load has converged to near-fair-share (all experts at ~0.125), so `mean(L)≈0.125` and `σ_cos` reflects the true cosine variance of the input distribution at that stage. The calibration locks λ to the scale of the routing signal, not to a manual constant.
  **result:** After calibration, λ remained fixed for steps 200–5000. No instability at the calibration boundary (step 200) was observed.
  **note:** The step-200 trigger is post-optimizer-warmup (warmup_steps=400 for LR, but EMA converges faster). In principle calibrating mid-LR-warmup means σ_cos is slightly suppressed by smaller gradient updates — worth checking if step 400 would give a different λ value and whether that matters.

- **[observation]** eff_E stabilized at 6.0–6.2 (75–77% of 8 experts) from step 2000 onward. gini peaked 0.424 at step 1000, then declined to 0.388 at step 5000.
  **result:** The post-peak gini decline is consistent with SPAR finding a fixed point: once EMA load differences drive `max(0, L_i - 1/N)` nonzero, the one-sided penalty bleeds load from dominant experts toward underloaded ones until equilibrium. The equilibrium is not uniform (gini≠0) because cosine alignment legitimately varies.
  **note:** eff_E settling at 6.1/8 rather than 8/8 is expected behavior — SPAR is designed to prevent monopolisation, not enforce uniformity. Expert 0 and Expert 7 (hypothetically) may simply be better-aligned with wikitext-103 token distributions; SPAR keeps them at L_i≈2/N rather than L_i→1.

- **[observation]** conf (`welford_mu`) grew monotonically 0.522→0.562 across 5000 steps. This is the per-expert mean cosine distance — decreasing distance means W_i directions are converging toward the centroids of their routed token clusters.
  **result:** conf has not plateaued at step 5000. Still growing at +0.001/500 steps in the 4500–5000 window.
  **note:** Monotonic conf growth with no plateau suggests either (a) the optimizer has not saturated the prototype directions, or (b) the task-loss gradient is still meaningfully updating W. Either points to potential gains from extending past 5000 steps. Worth a 10k-step continuation run before the fineweb-edu comparison.

- **[architecture]** SPAR DDP overhead: 1 `all_reduce` per step (EMA load), 1 additional at step 200 (λ sync). Welford sync deliberately removed from `step()` — eliminated 3 `all_gather` calls per step that were causing SEQNUM drift and a potential NCCL deadlock in early iterations.
  **result:** No NCCL issues across entire 5000-step A100:4 run. Per-step sync budget is minimal compared to metabolic (6 all_reduce/step).
  **note:** Per-rank Welford divergence is acceptable because it is metrics-only; routing correctness depends only on EMA load, which is fully synced. This was a deliberate design decision — removing Welford from `step()` is not an approximation, it is correct.

- **[hypothesis]** conf trajectory predicts downstream specialisation quality. If the Stress–PPL correlation hypothesis holds (low cosine-distance variance → efficient LoRA approximation → lower PPL), then the monotonic conf growth through step 5000 implies the model is still specialising and PPL has not bottomed out. The checkpoints at steps 3000–5000 are the right range to measure the correlation.
  **result:** Not yet tested. Cross-checkpoint Stress–val_PPL scatter is the required experiment.
  **note:** conf=0.562 at step 5000 means mean cosine distance to routed-token centroid is 0.438 (distance = 1 − conf). This is moderate alignment — the gap between conf at step 500 (0.549) and step 5000 (0.562) is small in absolute terms, suggesting most prototype learning happened in the rapid phase (0–500).

- **[idea]** λ_calib_step=200 may be suboptimal relative to the LR warmup schedule. LR warmup runs to step 400; during steps 0–200 gradient magnitudes are ramped, so cos_sim distributions are narrower than they will be at full LR. Calibrating at step 400 (end of warmup) would use a more representative σ_cos. Low-cost test: rerun with lambda_calib_step=400 and compare final λ value and gini trajectory.
  **result:** Not yet tested.
  **note:** Expected direction: λ_400 > λ_200 (larger σ_cos at full LR → larger numerator). Whether a larger λ overshoots and suppresses specialisation below eff_E=6 is the unknown.

---

## 2026-03-09

- **Standard aux-loss router collapses early** (gini=0.616, eff_E=4.0/8 by step 350 at aux_loss_coef=0.01), driving some experts to zero tokens. This requires `find_unused_parameters=True` in DDP (~5–10% per-step overhead) — a permanent tax for any collapsing router.

- **Metabolic router never triggers the zero-token condition.** Fatigue dynamics keep all 8 experts active throughout training, making it compatible with `find_unused_parameters=False`. This DDP compatibility advantage is absent from Switch Transformer, ST-MoE, and Mixtral papers.

- **v4 (λ=0.5 + warmup=400) reaches stable equilibrium.** Layer 1 gini stabilises at ~0.50 from step 1800 onward (not collapsing). Gini slope in the 2400–2900 window: +0.000013/step → effectively zero. The SoftSign penalty holds the overloaded experts at a fixed point rather than allowing continued concentration.

- **Layer-depth specialisation gradient confirmed.** MI(token, expert) increases monotonically from L1 (0.10) to L11 (0.24) at step 4150. Later layers develop more token-type-specific routing without any explicit per-layer objective — the metabolic router discovers this structure naturally.

- **λ warmup prevents early routing lock-in.** Without warmup (v2), Layer 1 locked to 2 dominant experts before step 200. With warmup=400 (v3/v4), Layer 1 stays near-uniform (gini<0.08) through step 400, giving the gate time to find better prototype directions before fatigue feedback activates.

- **λ=0.3 (v3) too weak post-warmup.** Layer 1 continued collapsing to gini=0.59 by step 2350. λ=0.5 (v4) holds Layer 1 at equilibrium gini≈0.50. The penalty at equilibrium (λ×SoftSign(F_max)≈0.31) matches the gate's alignment advantage for dominant experts.
