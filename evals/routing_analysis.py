from __future__ import annotations

import os
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer

from evals.loading import load_model_for_eval
from evals.results_schema import build_results_payload, write_results_json
from src.configs.model import model_lookup


class _RoutingHookState:
    def __init__(self) -> None:
        # layer_idx -> expert_idx -> list[token_id]
        self.token_log: Dict[int, Dict[int, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._handles: list = []

    def clear(self) -> None:
        self.token_log = defaultdict(lambda: defaultdict(list))

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _attach_routing_hooks(
    model: Any,
    state: _RoutingHookState,
    input_ids_ref: list,
) -> None:
    from src.layers.lora_moe import LoRAMoELayer

    def _make_hook(layer_idx: int):
        def _hook(module: LoRAMoELayer, inputs, outputs):
            current_ids = input_ids_ref[0]
            if current_ids is None:
                return

            # All routers return (expert_weights [B*S, E], None, metrics) — the
            # indices slot is always None. Read dispatch from _last_routing_weights
            # which LoRAMoELayer sets before the hook fires.
            weights = getattr(module, "_last_routing_weights", None)
            if weights is None:
                return

            hidden = inputs[0]
            B, T, _ = hidden.shape
            # weights: [B*S, E] — nonzero entries identify active (token, expert) pairs
            with torch.no_grad():
                active = weights.nonzero(
                    as_tuple=False
                )  # [nnz, 2]: (flat_token, expert)
            for row in active:
                flat_tok = int(row[0].item())
                expert_idx = int(row[1].item())
                b, t = divmod(flat_tok, T)
                if b >= B or t >= T:
                    continue
                tok_id = int(current_ids[b, t].item())
                state.token_log[layer_idx][expert_idx].append(tok_id)

        return _hook

    for name, module in model.named_modules():
        if isinstance(module, LoRAMoELayer):
            parts = name.split(".")
            layer_idx = None
            for p in parts:
                if p.isdigit():
                    layer_idx = int(p)
                    break
            if layer_idx is None:
                continue
            handle = module.register_forward_hook(_make_hook(layer_idx))
            state._handles.append(handle)


def _decode_tokens(token_ids: List[int], tokenizer: Any) -> List[str]:
    return [tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]


def analyze_expert_token_distributions(
    token_log: Dict[int, Dict[int, List[int]]],
    tokenizer: Any,
    top_n: int = 50,
) -> Dict[int, Dict[int, Dict]]:
    results: Dict[int, Dict[int, Dict]] = {}

    for layer_idx, expert_map in sorted(token_log.items()):
        results[layer_idx] = {}
        for expert_idx, token_ids in sorted(expert_map.items()):
            total = len(token_ids)
            if total == 0:
                results[layer_idx][expert_idx] = {
                    "total_tokens": 0,
                    "unique_tokens": 0,
                    "type_token_ratio": 0.0,
                    "top_tokens": [],
                }
                continue

            counter = Counter(token_ids)
            unique = len(counter)
            ttr = unique / total

            top_tokens = [
                {
                    "token": tokenizer.decode([tid], skip_special_tokens=False),
                    "token_id": tid,
                    "count": cnt,
                    "freq": cnt / total,
                }
                for tid, cnt in counter.most_common(top_n)
            ]

            results[layer_idx][expert_idx] = {
                "total_tokens": total,
                "unique_tokens": unique,
                "type_token_ratio": round(ttr, 4),
                "top_tokens": top_tokens,
            }

    return results


def compute_specialization_score(stats: Dict) -> float:
    # 1 - TTR: high score = narrow/repetitive token distribution (specialist)
    ttr = stats.get("type_token_ratio", 1.0)
    return round(1.0 - ttr, 4)


def run_routing_analysis(
    config: Any,
    checkpoint_path: str | Path,
    *,
    model: Any | None = None,
    checkpoint_info: Dict[str, Any] | None = None,
    texts: Optional[List[str]] = None,
    n_samples: int = 200,
    max_length: int = 512,
    top_n_tokens: int = 50,
    device: str = "cuda",
    output_path: str | Path | None = None,
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)

    if model is None:
        model, checkpoint_info = load_model_for_eval(
            config=config,
            checkpoint_path=checkpoint_path,
            device=device,
            dtype=autocast_dtype if device.startswith("cuda") else None,
        )

    model_key = (
        config.get("model", {}).get("model_key", "gpt-neo-125m")
        if isinstance(config, dict)
        else getattr(getattr(config, "model", None), "model_key", "gpt-neo-125m")
    )
    model_info = model_lookup(model_key)
    tokenizer = AutoTokenizer.from_pretrained(
        model_info["hf_name"],
        token=os.environ.get("HF_TOKEN"),
    )

    if texts is None:
        try:
            from datasets import load_dataset

            ds = load_dataset(
                "wikitext",
                "wikitext-103-raw-v1",
                split="validation",
                streaming=True,
            )
            texts = []
            for row in ds:
                t = row.get("text", "").strip()
                if len(t) > 100:
                    texts.append(t)
                if len(texts) >= n_samples:
                    break
        except Exception:
            texts = [
                "The transformer architecture has become the dominant approach "
                "in natural language processing tasks.",
            ] * min(n_samples, 10)

    hook_state = _RoutingHookState()
    input_ids_ref: list = [None]
    _attach_routing_hooks(model, hook_state, input_ids_ref)

    model.eval()

    use_cuda = device.startswith("cuda")

    processed = 0
    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            input_ids = enc["input_ids"].to(device)
            if input_ids.size(1) < 2:
                continue

            input_ids_ref[0] = input_ids

            ctx = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if use_cuda
                else nullcontext()
            )
            with ctx:
                model(input_ids=input_ids)

            processed += 1

    hook_state.remove_hooks()
    input_ids_ref[0] = None

    print(f"\nRouting analysis: processed {processed} samples.")

    analysis = analyze_expert_token_distributions(
        hook_state.token_log,
        tokenizer,
        top_n=top_n_tokens,
    )

    summary_rows = []
    for layer_idx, expert_map in sorted(analysis.items()):
        for expert_idx, stats in sorted(expert_map.items()):
            summary_rows.append(
                {
                    "layer": layer_idx,
                    "expert": expert_idx,
                    "total_tokens": stats["total_tokens"],
                    "unique_tokens": stats["unique_tokens"],
                    "type_token_ratio": stats["type_token_ratio"],
                    "specialization_score": compute_specialization_score(stats),
                    "top_5_tokens": [t["token"] for t in stats["top_tokens"][:5]],
                }
            )

    print("\n── Expert Token Distribution Summary ───────────────────────────────")
    print(
        f"{'Layer':>6} {'Expert':>7} {'Tokens':>8} {'Unique':>8} "
        f"{'TTR':>6} {'Spec.':>6}  Top-5 tokens"
    )
    print("─" * 80)
    for row in summary_rows:
        top5 = ", ".join(repr(t) for t in row["top_5_tokens"])
        print(
            f"{row['layer']:>6} {row['expert']:>7} {row['total_tokens']:>8} "
            f"{row['unique_tokens']:>8} {row['type_token_ratio']:>6.3f} "
            f"{row['specialization_score']:>6.3f}  {top5}"
        )
    print("────────────────────────────────────────────────────────────────────\n")

    payload = build_results_payload(
        task="routing_analysis",
        checkpoint_path=checkpoint_path,
        checkpoint_info=checkpoint_info,
        config=config,
        results={
            "samples_processed": float(processed),
            "layers_analysed": float(len(analysis)),
        },
        metadata={
            "n_samples": n_samples,
            "max_length": max_length,
            "top_n_tokens": top_n_tokens,
            "device": device,
            "summary": summary_rows,
            "full_analysis": {
                str(layer): {
                    str(expert): {
                        "total_tokens": s["total_tokens"],
                        "unique_tokens": s["unique_tokens"],
                        "type_token_ratio": s["type_token_ratio"],
                        "specialization_score": compute_specialization_score(s),
                        "top_tokens": s["top_tokens"],
                    }
                    for expert, s in expert_map.items()
                }
                for layer, expert_map in analysis.items()
            },
        },
    )

    if output_path is not None:
        write_results_json(payload, output_path)
        print(f"Results written to {output_path}")

    return payload
