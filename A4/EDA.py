import os
import re
import json
import argparse
from collections import Counter

from nanochat.common import compute_init, compute_cleanup, autodetect_device_type, get_base_dir
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K, extract_answer

def detect_repetition(text, ngram_size=4, threshold=3):
    """
    Detect if the model output contains repetitive loops.
    Returns (is_repetitive: bool, repetition_score: float, repeated_phrase: str)
    """
    words = text.split()
    if len(words) < ngram_size * 2:
        return False, 0.0, ""

    # Count n-gram occurrences
    ngrams = []
    for i in range(len(words) - ngram_size + 1):
        ngram = " ".join(words[i:i + ngram_size])
        ngrams.append(ngram)

    counts = Counter(ngrams)
    if not counts:
        return False, 0.0, ""

    most_common_phrase, most_common_count = counts.most_common(1)[0]
    # Repetition score: ratio of most-repeated n-gram to total n-grams
    repetition_score = most_common_count / len(ngrams)

    is_repetitive = most_common_count >= threshold
    return is_repetitive, round(repetition_score, 4), most_common_phrase if is_repetitive else ""


def detect_format_compliance(text):
    """Check if the model used the #### answer format."""
    has_hash = "####" in text
    extracted = extract_answer(text)
    return {
        "has_hash_marker": has_hash,
        "extracted_answer": extracted,
    }


def detect_answer_proximity(text, ref_answer):
    """
    Check if the correct number appears anywhere in the response,
    even if it wasn't correctly formatted with ####.
    """
    if ref_answer is None:
        return {"correct_number_in_text": False, "near_miss": False}

    correct_in_text = ref_answer in text
    # Check if a close number appears (within 10% for numeric answers)
    near_miss = False
    try:
        ref_val = float(ref_answer.replace(",", ""))
        # Find all numbers in the text
        numbers = re.findall(r'\-?[\d,]+\.?\d*', text)
        for num_str in numbers:
            try:
                num_val = float(num_str.replace(",", ""))
                if ref_val != 0 and abs(num_val - ref_val) / abs(ref_val) < 0.1:
                    near_miss = True
                    break
                elif ref_val == 0 and abs(num_val) < 1:
                    near_miss = True
                    break
            except ValueError:
                continue
    except ValueError:
        pass

    return {
        "correct_number_in_text": correct_in_text,
        "near_miss": near_miss,
    }


def classify_error(output, ref_answer, is_correct, format_info, rep_info, proximity_info):
    """
    Classify the error into a category for clustering.
    Categories:
      - correct: Model got it right
      - format_error: Right number present but wrong format (no ####)
      - repetition_loop: Model got stuck in a repetitive loop
      - wrong_answer: Model produced a number but it was wrong
      - no_answer: Model didn't produce any identifiable number
      - near_miss: Model produced a number close to the correct answer
      - truncated: Model output appears cut off (hit max tokens)
    """
    if is_correct:
        return "correct"

    is_rep, _, _ = rep_info
    if is_rep:
        return "repetition_loop"

    if proximity_info["correct_number_in_text"] and not format_info["has_hash_marker"]:
        return "format_error"

    if proximity_info["correct_number_in_text"] and format_info["has_hash_marker"]:
        # Has #### and correct number is in text, but still wrong
        # This means it extracted a different number after ####
        return "wrong_answer"

    if proximity_info["near_miss"]:
        return "near_miss"

    if format_info["extracted_answer"] is not None:
        return "wrong_answer"

    # Check if output seems truncated (ends mid-sentence, no period/####)
    stripped = output.strip()
    if stripped and not stripped[-1] in '.!?':
        return "truncated"

    return "no_answer"


def estimate_reasoning_steps(text):
    """
    Count approximate number of reasoning/calculation steps in the response.
    Looks for patterns like equations, "Step X", numbered lists, etc.
    """
    # Count lines that look like calculation steps
    lines = text.strip().split('\n')
    step_count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Lines with = sign (equations)
        if '=' in line and any(c.isdigit() for c in line):
            step_count += 1
        # Lines starting with numbers or "Step"
        elif re.match(r'^(\d+[\.\):]|Step\s)', line, re.IGNORECASE):
            step_count += 1
    return step_count


def compute_problem_difficulty(question):
    """
    Estimate problem difficulty based on question characteristics.
    Returns a difficulty category: easy, medium, hard
    """
    # Count numbers mentioned in the question
    numbers = re.findall(r'\d+', question)
    num_count = len(numbers)

    # Count sentences (rough proxy for complexity)
    sentences = re.split(r'[.?!]', question)
    sentence_count = len([s for s in sentences if s.strip()])

    # Large numbers suggest harder arithmetic
    has_large_numbers = any(int(n) > 100 for n in numbers if n.isdigit())

    if num_count <= 2 and sentence_count <= 2 and not has_large_numbers:
        return "easy"
    elif num_count >= 5 or sentence_count >= 4 or has_large_numbers:
        return "hard"
    else:
        return "medium"



def dump_mistakes(source, model_tag, step, max_questions):
    device_type = autodetect_device_type()
    ddp, _ddp_rank, _ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    if ddp_world_size != 1:
        raise RuntimeError("This dump script must be run single-process.")

    print(f"Loading {source} model (tag: {model_tag})...")
    model, tokenizer, meta = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
    engine = Engine(model, tokenizer)

    # Load GSM8K test set
    task = GSM8K(subset="main", split="test")
    num_questions = min(max_questions, len(task))

    results = []
    print(f"Evaluating {num_questions} GSM8K questions...")

    for i in range(num_questions):
        conversation = task[i]
        encoded_prompt = tokenizer.render_for_completion(conversation)

        # Generate answer
        sample, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=1,
            max_tokens=512,
            temperature=0.0,
            top_k=50,
        )

        prefix_length = len(encoded_prompt)
        generated_tokens = sample[0][prefix_length:]
        output_str = tokenizer.decode(generated_tokens).strip()

        is_correct = bool(task.evaluate(conversation, output_str))

        # Extract question and reference answer
        question_str = conversation['messages'][0]['content']
        assistant_content = conversation['messages'][1]['content']
        if isinstance(assistant_content, list):
            expected_full = "".join(
                p['text'] if p['type'] == 'text' else f"<<{p['text']}>>"
                for p in assistant_content
            )
        else:
            expected_full = str(assistant_content)

        # Get the reference numerical answer
        last_text_part = assistant_content[-1]['text'] if isinstance(assistant_content, list) else str(assistant_content)
        ref_answer = extract_answer(last_text_part)

        # ── Run all EDA diagnostics ──
        format_info = detect_format_compliance(output_str)
        is_rep, rep_score, rep_phrase = detect_repetition(output_str)
        rep_info = (is_rep, rep_score, rep_phrase)
        proximity_info = detect_answer_proximity(output_str, ref_answer)
        error_category = classify_error(output_str, ref_answer, is_correct, format_info, rep_info, proximity_info)
        reasoning_steps = estimate_reasoning_steps(output_str)
        difficulty = compute_problem_difficulty(question_str)

        results.append({
            "id": i,
            "question": question_str,
            "expected_answer": ref_answer,
            "expected_full_solution": expected_full,
            "generated_output": output_str,
            "correct": is_correct,
            "output_length": len(output_str),
            "output_word_count": len(output_str.split()),
            # EDA fields
            "error_category": error_category,
            "difficulty": difficulty,
            "reasoning_steps_detected": reasoning_steps,
            "format_compliance": format_info,
            "repetition": {
                "is_repetitive": is_rep,
                "repetition_score": rep_score,
                "repeated_phrase": rep_phrase,
            },
            "answer_proximity": proximity_info,
        })

        if (i + 1) % 10 == 0:
            correct_so_far = sum(1 for r in results if r["correct"])
            print(f"  [{i+1}/{num_questions}] Accuracy so far: {correct_so_far}/{i+1} ({100*correct_so_far/(i+1):.1f}%)")

    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])

    # Error category distribution
    category_counts = Counter(r["error_category"] for r in results)

    # Difficulty breakdown
    difficulty_counts = Counter(r["difficulty"] for r in results)
    difficulty_accuracy = {}
    for diff in ["easy", "medium", "hard"]:
        diff_items = [r for r in results if r["difficulty"] == diff]
        if diff_items:
            diff_correct = sum(1 for r in diff_items if r["correct"])
            difficulty_accuracy[diff] = {
                "total": len(diff_items),
                "correct": diff_correct,
                "accuracy": round(diff_correct / len(diff_items), 4),
            }

    # Format compliance stats
    format_with_hash = sum(1 for r in results if r["format_compliance"]["has_hash_marker"])
    repetition_count = sum(1 for r in results if r["repetition"]["is_repetitive"])

    # Average output length
    avg_length = sum(r["output_length"] for r in results) / total if total > 0 else 0
    avg_word_count = sum(r["output_word_count"] for r in results) / total if total > 0 else 0
    avg_reasoning_steps = sum(r["reasoning_steps_detected"] for r in results) / total if total > 0 else 0

    summary = {
        "model_source": source,
        "model_tag": model_tag,
        "total_questions": total,
        "correct": correct_count,
        "accuracy": round(correct_count / total, 4) if total > 0 else 0,
        "error_category_distribution": dict(category_counts),
        "difficulty_breakdown": difficulty_accuracy,
        "format_compliance_rate": round(format_with_hash / total, 4) if total > 0 else 0,
        "repetition_rate": round(repetition_count / total, 4) if total > 0 else 0,
        "avg_output_length_chars": round(avg_length, 1),
        "avg_output_word_count": round(avg_word_count, 1),
        "avg_reasoning_steps": round(avg_reasoning_steps, 2),
    }

    output_data = {
        "summary": summary,
        "results": results,
    }

    print("\n" + "=" * 60)
    print(f"  GSM8K EDA Summary — {source} (tag: {model_tag})")
    print("=" * 60)
    print(f"  Accuracy: {correct_count}/{total} ({100*correct_count/total:.1f}%)")
    print(f"  Format compliance (has ####): {format_with_hash}/{total} ({100*format_with_hash/total:.1f}%)")
    print(f"  Repetition loops: {repetition_count}/{total} ({100*repetition_count/total:.1f}%)")
    print(f"  Avg output length: {avg_length:.0f} chars, {avg_word_count:.0f} words")
    print(f"  Avg reasoning steps: {avg_reasoning_steps:.1f}")
    print(f"\n  Error categories:")
    for cat, count in category_counts.most_common():
        print(f"    {cat:20s}: {count:4d} ({100*count/total:.1f}%)")
    print(f"\n  Difficulty breakdown:")
    for diff in ["easy", "medium", "hard"]:
        if diff in difficulty_accuracy:
            d = difficulty_accuracy[diff]
            print(f"    {diff:8s}: {d['correct']}/{d['total']} ({100*d['accuracy']:.1f}%)")
    print("=" * 60)

    cat_printed = 0
    print("\n  [Visual Examples of Mistakes]")
    print("-" * 60)
    for cat in ["format_error", "near_miss", "repetition_loop", "wrong_answer", "truncated"]:
        cat_results = [r for r in results if r["error_category"] == cat]
        if cat_results:
            ex = cat_results[0]
            print(f"\n  ➤ CATEGORY: {cat.upper()} (Problem #{ex['id']})")
            print(f"  Q: {ex['question']}")
            print(f"  Expected Number: {ex['expected_answer']}")
            
            # Print output, truncated if too long
            out_str = ex['generated_output']
            if len(out_str) > 300:
                print(f"  Model Output: {out_str[:300]}...\n  [...truncated...]")
            else:
                print(f"  Model Output: {out_str}")
            cat_printed += 1
            
    if cat_printed == 0:
        print("  No mistake examples to show.")
    print("=" * 60)

    report_dir = os.path.join(get_base_dir(), "report")
    os.makedirs(report_dir, exist_ok=True)
    tag_suffix = model_tag or "default"
    out_name = f"gsm8k_mistakes_{source}_{tag_suffix}.json"
    out_path = os.path.join(report_dir, out_name)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    compute_cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GSM8K Mistakes Dump + EDA for Part 3 & 4")
    parser.add_argument("--source", type=str, required=True, choices=["sft", "rl"])
    parser.add_argument("--model-tag", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=200, help="Number of GSM8K problems to evaluate")
    args = parser.parse_args()

    dump_mistakes(args.source, args.model_tag, args.step, args.num_samples)
