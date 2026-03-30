"""Part 3 variant: three long docs, key-value facts in exactly 2/3 docs."""

import argparse
import json
import random
import re
from typing import Callable, List, Optional

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init
from nanochat.engine import Engine


REAL_PROSE_PASSAGES = [
    (
        "A neighborhood library started a weekly science reading circle for students and parents. "
        "Participants compared short articles, discussed methods used in each study, and summarized key points. "
        "The librarian tracked attendance, noticing that sessions with concrete examples led to stronger engagement."
    ),
    (
        "At a small coastal research station, technicians calibrated sensors each morning before collecting tide and temperature data. "
        "Their reports highlighted gradual seasonal shifts and occasional spikes linked to storms. "
        "Local teachers used these reports in class to connect statistics with real environmental observations."
    ),
    (
        "A city museum redesigned one exhibit by adding clearer labels and a guided route through each section. "
        "Visitors spent more time reading background context and asked more focused questions at the end. "
        "Staff concluded that structure and narrative flow can significantly improve comprehension."
    ),
]

FUN_QA_PROMPTS = [
    "Q: What is the capital of France?\nA:",
    "Q: If 2 + 3 = 5, what is 7 + 8?\nA:",
    "Q: Complete the phrase: The opposite of hot is\nA:",
    "Q: Name one planet in our solar system.\nA:",
]


def parse_csv(text: str, cast: Callable[[str], object]) -> List[object]:
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def extract_choice_letter(text: str, option_letters: List[str]) -> Optional[str]:
    """Extract the first multiple-choice answer letter from allowed options."""
    allowed = "".join(option_letters)
    # Accept bare letters, "(A)", "A)", "A.", etc.
    m = re.search(rf"(?<![A-Z0-9])([{allowed}])(?=[^A-Z0-9]|$)", text.upper())
    return m.group(1) if m else None


def extract_value_token(
    text: str, candidate_values: Optional[List[str]] = None
) -> Optional[str]:
    """Extract value token from model output with robust fallbacks."""
    upper = text.upper()

    # Fast path: exact V###, including minor punctuation variants like V 123 or V-123.
    m = re.search(r"\bV\s*[-_:]?\s*(\d{1,4})\b", upper)
    if m:
        return f"V{m.group(1)}"

    # If we know the candidate values, look for exact mentions first.
    if candidate_values:
        for value in candidate_values:
            if re.search(rf"\b{re.escape(value.upper())}\b", upper):
                return value.upper()

        # Fallback: model may output just digits (e.g. "123"). Map by suffix.
        digit_to_values = {}
        for value in candidate_values:
            vm = re.match(r"V(\d{1,4})$", value.upper())
            if vm:
                d = vm.group(1)
                digit_to_values.setdefault(d, []).append(value.upper())
        for d in re.findall(r"\b(\d{1,4})\b", upper):
            vals = digit_to_values.get(d, [])
            if len(vals) == 1:
                return vals[0]

    return None


def parse_prediction(
    answer_text: str, option_values: List[str], option_letters: List[str]
) -> Optional[str]:
    """Return predicted letter from either letter output or direct value output."""
    letter = extract_choice_letter(answer_text, option_letters)
    if letter is not None:
        return letter

    value = extract_value_token(answer_text, candidate_values=option_values)
    if value is None:
        return None

    try:
        return option_letters[option_values.index(value)]
    except ValueError:
        return None


def build_option_values(
    rng: random.Random,
    target_value: str,
    other_value: str,
    num_options: int,
) -> List[str]:
    option_values = [target_value, other_value]
    while len(option_values) < num_options:
        candidate = f"V{rng.randint(100, 999)}"
        if candidate not in option_values:
            option_values.append(candidate)
    rng.shuffle(option_values)
    return option_values


def build_doc_tokens(
    tokenizer,
    base_tokens: List[int],
    budget: int,
    fact_text: Optional[str],
) -> List[int]:
    """Build one long document, optionally injecting a key-value fact."""
    budget = max(0, budget)
    if budget == 0:
        return []

    if fact_text is None:
        return base_tokens[:budget]

    fact_tokens = tokenizer.encode(f" {fact_text} ")
    if len(fact_tokens) >= budget:
        return fact_tokens[:budget]

    free = budget - len(fact_tokens)
    before = free // 2
    after = free - before
    return base_tokens[:before] + fact_tokens + base_tokens[:after]


def evaluate_one_model(
    engine: Engine,
    tokenizer,
    model_seq_len: int,
    context_lengths: List[int],
    trials: int,
    max_new_tokens: int,
    seed: int,
    show_example: bool,
    task_format: str,
    num_options: int,
    preset: str,
    debug_trials: int,
) -> List[dict]:
    rng = random.Random(seed)
    bos = tokenizer.get_bos_token_id()

    if preset == "easy":
        prefix = (
            "Read the three documents carefully.\n"
            "Find the value that matches the asked key.\n\n"
            "Context:\n"
        )
    else:
        prefix = (
            "Read the three long documents.\n"
            "Exactly two documents contain key-value facts.\n"
            "Use the facts to answer the multiple-choice question.\n\n"
            "Context:\n"
        )
    prefix_tokens = tokenizer.encode(prefix)
    doc_headers = [tokenizer.encode(f"\n\nDocument {i + 1}:\n") for i in range(3)]

    repeated_prose = " ".join(REAL_PROSE_PASSAGES)
    prose_repeat = 120 if preset == "easy" else 400
    prose_tokens = tokenizer.encode((" " + repeated_prose) * prose_repeat)
    example_printed = False

    rows = []
    for length in context_lengths:
        effective_length = length

        correct = 0
        parsed = 0
        target_value_hits = 0
        failed = False
        error_message = ""
        for trial_idx in range(trials):
            target_key = f"K{rng.randint(100, 999)}"
            target_value = f"V{rng.randint(100, 999)}"

            other_key = f"K{rng.randint(100, 999)}"
            if other_key == target_key:
                other_key = f"K{rng.randint(100, 999)}"
            other_value = f"V{rng.randint(100, 999)}"
            if other_value == target_value:
                other_value = f"V{rng.randint(100, 999)}"

            target_fact = f"Key {target_key} maps to value {target_value}."
            other_fact = f"Key {other_key} maps to value {other_value}."

            if preset == "easy":
                # Stable structure in easy mode: target always in Doc 2, distractor in Doc 1.
                target_doc = 1
                other_doc = 0
            else:
                keyed_docs = sorted(rng.sample([0, 1, 2], 2))
                target_doc = rng.choice(keyed_docs)
                other_doc = (
                    keyed_docs[0] if keyed_docs[1] == target_doc else keyed_docs[1]
                )

            option_letters = [chr(ord("A") + i) for i in range(num_options)]
            option_values = build_option_values(
                rng, target_value, other_value, num_options
            )
            correct_letter = option_letters[option_values.index(target_value)]

            if task_format == "direct":
                suffix = (
                    f"\n\nQuestion: What is the value for key {target_key}?\n"
                    "Answer with the value token only (example: V123).\n"
                    "Answer:"
                )
            else:
                options = "\n".join(
                    f"{option_letters[i]}) {option_values[i]}"
                    for i in range(num_options)
                )
                suffix = (
                    f"\n\nQuestion: What is the value for key {target_key}?\n"
                    f"{options}\n"
                    f"Answer with one letter only ({', '.join(option_letters)}).\n"
                    "Answer:"
                )
            suffix_tokens = tokenizer.encode(suffix)

            trial_budget = (
                effective_length - 1 - len(prefix_tokens) - len(suffix_tokens)
            )
            header_budget = sum(len(h) for h in doc_headers)
            body_budget = max(0, trial_budget - header_budget)
            if body_budget == 0:
                continue

            base = body_budget // 3
            doc_budgets = [base, base, body_budget - 2 * base]

            docs = []
            for i in range(3):
                fact_text: Optional[str] = None
                if i == target_doc:
                    fact_text = (
                        f"{target_fact} {target_fact}"
                        if preset == "easy"
                        else target_fact
                    )
                elif i == other_doc:
                    fact_text = other_fact
                docs.extend(doc_headers[i])
                docs.extend(
                    build_doc_tokens(tokenizer, prose_tokens, doc_budgets[i], fact_text)
                )

            prompt = (
                [bos] + prefix_tokens + docs[: max(0, trial_budget)] + suffix_tokens
            )
            if show_example and not example_printed:
                rendered_prompt = tokenizer.decode(prompt[1:])
                print("\n--- Example trial prompt ---")
                print(rendered_prompt)
                print(f"\n[expected correct letter: {correct_letter}]")
                print("--- End example ---\n")
                example_printed = True
            try:
                sample, _ = engine.generate_batch(
                    prompt,
                    num_samples=1,
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                    top_k=1,
                )
            except Exception as exc:
                failed = True
                error_message = str(exc)
                break
            generated = sample[0][len(prompt) :]
            answer = tokenizer.decode(generated).strip()
            candidate_values_for_parse = [target_value, other_value] + option_values
            parsed_value = extract_value_token(
                answer, candidate_values=candidate_values_for_parse
            )
            if parsed_value == target_value:
                target_value_hits += 1

            if task_format == "direct":
                predicted_value = parsed_value
                if predicted_value is not None:
                    parsed += 1
                if predicted_value == target_value:
                    correct += 1
                predicted_for_debug = predicted_value
                expected_for_debug = target_value
                is_correct = predicted_value == target_value
            else:
                predicted = parse_prediction(answer, option_values, option_letters)
                if predicted is not None:
                    parsed += 1
                if predicted == correct_letter:
                    correct += 1
                predicted_for_debug = predicted
                expected_for_debug = correct_letter
                is_correct = predicted == correct_letter

            if trial_idx < debug_trials:
                print(
                    "[debug] "
                    f"L={length} trial={trial_idx + 1} "
                    f"key={target_key} expected={expected_for_debug} "
                    f"predicted={predicted_for_debug} correct={is_correct} "
                    f'raw="{answer}"'
                )

        if failed:
            rows.append(
                {
                    "context_len": length,
                    "effective_context_len": effective_length,
                    "trials": 0,
                    "accuracy": None,
                    "note": f"runtime error: {error_message}",
                }
            )
            continue

        row = {
            "context_len": length,
            "effective_context_len": effective_length,
            "trials": trials,
            "accuracy": correct / max(1, trials),
            "parsed_rate": parsed / max(1, trials),
            "target_value_rate": target_value_hits / max(1, trials),
        }
        rows.append(row)

    return rows


def run_fun_qa(
    engine: Engine,
    tokenizer,
    seed: int,
    count: int,
    max_new_tokens: int,
) -> List[dict]:
    """Run a few simple general QA prompts and return model outputs."""
    rng = random.Random(seed + 999)
    bos = tokenizer.get_bos_token_id()
    prompts = FUN_QA_PROMPTS.copy()
    rng.shuffle(prompts)
    prompts = prompts[: max(0, min(count, len(prompts)))]
    rows: List[dict] = []
    for prompt_text in prompts:
        prompt_tokens = [bos] + tokenizer.encode(prompt_text)
        sample, _ = engine.generate_batch(
            prompt_tokens,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=0.0,
            top_k=1,
        )
        generated = sample[0][len(prompt_tokens) :]
        answer = tokenizer.decode(generated).strip()
        rows.append({"prompt": prompt_text, "output": answer})
    return rows


def run_phase(
    phase_name: str,
    tag: str,
    step: Optional[int],
    device,
    context_lengths: List[int],
    trials: int,
    max_new_tokens: int,
    seed: int,
    show_example: bool,
    task_format: str,
    num_options: int,
    preset: str,
    debug_trials: int,
    fun_qa_count: int,
    fun_qa_max_new_tokens: int,
) -> dict:
    model, tokenizer, meta = load_model(
        "base", device, phase="eval", model_tag=tag, step=step
    )
    seq_len = meta["model_config"]["sequence_len"]
    print(f"[{phase_name}] loaded tag={tag} step={meta['step']} seq_len={seq_len}")
    engine = Engine(model, tokenizer)
    rows = evaluate_one_model(
        engine=engine,
        tokenizer=tokenizer,
        model_seq_len=seq_len,
        context_lengths=context_lengths,
        trials=trials,
        max_new_tokens=max_new_tokens,
        seed=seed,
        show_example=show_example,
        task_format=task_format,
        num_options=num_options,
        preset=preset,
        debug_trials=debug_trials,
    )
    fun_qa = run_fun_qa(
        engine=engine,
        tokenizer=tokenizer,
        seed=seed,
        count=fun_qa_count,
        max_new_tokens=fun_qa_max_new_tokens,
    )
    return {
        "phase": phase_name,
        "model_tag": tag,
        "loaded_step": meta["step"],
        "model_sequence_len": seq_len,
        "rows": rows,
        "fun_qa": fun_qa,
    }


def main():
    p = argparse.ArgumentParser(
        description="Part 3 eval variant: 2 of 3 long docs contain key-value facts"
    )
    p.add_argument("--phase1-tag", type=str, required=True)
    p.add_argument("--phase1-step", type=int, default=None)
    p.add_argument("--phase2-tag", type=str, required=True)
    p.add_argument("--phase2-step", type=int, default=None)
    p.add_argument("--context-lengths", type=str, default="256,512,1024,1536,2048")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=4)
    p.add_argument("--task-format", type=str, default="mcq", choices=["mcq", "direct"])
    p.add_argument(
        "--num-options",
        type=int,
        default=4,
        help="Number of MCQ options (2-6). Ignored for direct.",
    )
    p.add_argument("--preset", type=str, default="easy", choices=["easy", "standard"])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--show-example",
        action="store_true",
        help="Print one concrete trial prompt per model",
    )
    p.add_argument(
        "--debug-trials",
        type=int,
        default=0,
        help="Print scoring details for first N trials per context",
    )
    p.add_argument(
        "--fun-qa-count",
        type=int,
        default=2,
        help="Number of simple general QA prompts to sample per model",
    )
    p.add_argument("--fun-qa-max-new-tokens", type=int, default=24)
    p.add_argument(
        "--device-type", type=str, default="", choices=["", "cuda", "cpu", "mps"]
    )
    p.add_argument("--output-json", type=str, default="part3_eval_2of3_results.json")
    args = p.parse_args()

    context_lengths = parse_csv(args.context_lengths, int)
    if args.preset == "easy":
        # Easier, straightforward retrieval check by default.
        args.task_format = "direct"
        args.num_options = 2
        args.max_new_tokens = max(args.max_new_tokens, 8)
    if args.task_format == "mcq" and not (2 <= args.num_options <= 6):
        raise ValueError("--num-options must be between 2 and 6 for mcq format.")
    device_type = (
        autodetect_device_type() if args.device_type == "" else args.device_type
    )
    ddp, _ddp_rank, _ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    if ddp_world_size != 1:
        raise RuntimeError(
            "part3_eval_2of3 is single-process. Use python -m, not torchrun."
        )

    out = {
        "config": {
            "context_lengths": context_lengths,
            "trials": args.trials,
            "max_new_tokens": args.max_new_tokens,
            "preset": args.preset,
            "task_format": args.task_format,
            "num_options": args.num_options,
            "seed": args.seed,
            "fun_qa_count": args.fun_qa_count,
            "fun_qa_max_new_tokens": args.fun_qa_max_new_tokens,
        },
        "models": [],
    }

    try:
        runs = [
            ("phase1", args.phase1_tag, args.phase1_step),
            ("phase2", args.phase2_tag, args.phase2_step),
        ]
        for name, tag, step in runs:
            res = run_phase(
                phase_name=name,
                tag=tag,
                step=step,
                device=device,
                context_lengths=context_lengths,
                trials=args.trials,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                show_example=args.show_example,
                task_format=args.task_format,
                num_options=args.num_options,
                preset=args.preset,
                debug_trials=max(0, args.debug_trials),
                fun_qa_count=max(0, args.fun_qa_count),
                fun_qa_max_new_tokens=max(1, args.fun_qa_max_new_tokens),
            )
            out["models"].append(res)
            print(f"\n== {name} results ==")
            for row in res["rows"]:
                if row.get("note"):
                    acc_text = (
                        "n/a" if row["accuracy"] is None else f"{row['accuracy']:.3f}"
                    )
                    print(
                        f"  L={row['context_len']:>4} (effective {row['effective_context_len']:>4}) "
                        f"acc={acc_text} [{row['note']}]"
                    )
                else:
                    print(
                        f"  L={row['context_len']:>4} acc={row['accuracy']:.3f} "
                        f"parsed={row['parsed_rate']:.3f} target_value={row['target_value_rate']:.3f}"
                    )
            if res["fun_qa"]:
                print(f"\n== {name} fun QA ==")
                for qa in res["fun_qa"]:
                    print(f"  {qa['prompt']}")
                    print(f"  -> {qa['output']}")
    finally:
        compute_cleanup()

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {args.output_json}")


if __name__ == "__main__":
    main()
