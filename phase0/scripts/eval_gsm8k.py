"""
phase0/scripts/eval_gsm8k.py — GSM8K math reasoning benchmark

So sánh accuracy của:
  1. True baseline  — Qwen3-1.7B gốc (n_iter=0)
  2. LoopedLM n_iter=N với connect layer đã train

GSM8K: 1319 grade-school math problems, ground truth là số cuối sau "####".
Metric: exact match accuracy (số cuối greedy decode == ground truth).

Chạy:
    python phase0/scripts/eval_gsm8k.py \\
        --checkpoint phase0/checkpoints/residual_seq4096_think_3epoch/best_connect.pt \\
        --connect_type residual --n_iter 3

    # Nhanh hơn — chỉ 100 examples
    python phase0/scripts/eval_gsm8k.py \\
        --checkpoint ... --max_samples 100

    # So sánh nhiều n_iter
    python phase0/scripts/eval_gsm8k.py \\
        --checkpoint ... --eval_all_iters
"""

import argparse, sys, os, re, math, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from phase0.src import Phase0Model, Phase0Config


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_gt_answer(answer_str: str) -> str | None:
    """Extract ground-truth number from '#### 18' at end of GSM8K answer."""
    m = re.search(r"####\s*([\d,\.\-]+)", answer_str)
    if m:
        return m.group(1).replace(",", "").strip()
    return None


def extract_pred_answer(generated: str) -> str | None:
    """
    Extract predicted answer from model output.
    Priority:
      1. Last number after '####' (model follows GSM8K format)
      2. Last boxed number: \\boxed{N}
      3. Last standalone number in text
    """
    # Pattern 1: #### N
    m = re.search(r"####\s*([\d,\.\-]+)", generated)
    if m:
        return m.group(1).replace(",", "").strip()

    # Pattern 2: \boxed{N}
    m = re.search(r"\\boxed\{([^\}]+)\}", generated)
    if m:
        return m.group(1).replace(",", "").strip()

    # Pattern 3: last number in text
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", generated)
    if nums:
        return nums[-1].replace(",", "")

    return None


def answers_match(pred: str | None, gt: str | None) -> bool:
    if pred is None or gt is None:
        return False
    try:
        return abs(float(pred) - float(gt)) < 1e-6
    except ValueError:
        return pred.strip() == gt.strip()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_answer(
    model: Phase0Model,
    tokenizer,
    question: str,
    n_iter: int,
    max_new_tokens: int = 512,
    device: torch.device = None,
) -> str:
    """Generate model answer for a single GSM8K question."""
    orig_n_iter = model.cfg.n_iter
    model.cfg.n_iter = n_iter

    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    if device is not None:
        input_ids = input_ids.to(device)

    prompt_len = input_ids.shape[1]
    generated_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        out = model(generated_ids)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        # Move next_token to same device as generated_ids
        generated_ids = torch.cat([generated_ids, next_token.to(generated_ids.device)], dim=1)
        if next_token.item() == tokenizer.eos_token_id:
            break

    new_tokens = generated_ids[0, prompt_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    model.cfg.n_iter = orig_n_iter
    return response


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def eval_accuracy(
    model: Phase0Model,
    tokenizer,
    examples: list[dict],
    n_iter: int,
    device: torch.device,
    max_new_tokens: int = 512,
    desc: str = "",
) -> dict:
    correct = 0
    total   = 0
    parse_fail = 0
    results = []

    for ex in tqdm(examples, desc=desc or f"n_iter={n_iter}", ncols=80):
        gt  = extract_gt_answer(ex["answer"])
        gen = generate_answer(model, tokenizer, ex["question"], n_iter,
                              max_new_tokens=max_new_tokens, device=device)
        pred = extract_pred_answer(gen)
        ok   = answers_match(pred, gt)
        if pred is None:
            parse_fail += 1
        correct += int(ok)
        total   += 1
        results.append({
            "question": ex["question"][:120],
            "gt": gt,
            "pred": pred,
            "correct": ok,
        })

    acc = correct / total if total > 0 else 0.0
    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "parse_fail": parse_fail,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GSM8K reasoning benchmark")
    p.add_argument("--model",          default="Qwen/Qwen3-1.7B")
    p.add_argument("--checkpoint",     default=None,
                   help="Path to connect layer .pt checkpoint")
    p.add_argument("--connect_type",   default="residual",
                   choices=["residual", "mlp", "gated", "iter_aware"])
    p.add_argument("--loop_start",     type=int,   default=8)
    p.add_argument("--loop_end",       type=int,   default=20)
    p.add_argument("--n_iter",         type=int,   default=3,
                   help="n_iter for looped model (n_iter=0 = baseline)")
    p.add_argument("--eval_all_iters", action="store_true",
                   help="Eval n_iter=0..N, one pass per iter count")
    p.add_argument("--max_samples",    type=int,   default=None,
                   help="Limit to first N GSM8K test examples (default: all 1319)")
    p.add_argument("--max_new_tokens", type=int,   default=512)
    p.add_argument("--dtype",          default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--output_json",    default=None,
                   help="Save detailed results to JSON file")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]

    # Load GSM8K
    print("[gsm8k] Loading test set …")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    examples = [{"question": r["question"], "answer": r["answer"]} for r in ds]
    if args.max_samples:
        examples = examples[:args.max_samples]
    print(f"[gsm8k] {len(examples)} examples")

    # Load model
    cfg   = Phase0Config(
        model_name   = args.model,
        loop_start   = args.loop_start,
        loop_end     = args.loop_end,
        n_iter       = args.n_iter,
        connect_type = args.connect_type,
    )
    model = Phase0Model.from_pretrained(cfg, torch_dtype=dtype)
    device = next(model.connect.parameters()).device
    model.eval()

    if args.checkpoint:
        model.load_connect(args.checkpoint)
        print(f"[gsm8k] Checkpoint: {args.checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("=" * 65)
    print("  GSM8K Reasoning Benchmark")
    print("=" * 65)

    iters_to_eval = list(range(0, args.n_iter + 1)) if args.eval_all_iters else [0, args.n_iter]
    all_results = {}

    for n in iters_to_eval:
        label = "baseline" if n == 0 else f"looped n_iter={n}"
        res = eval_accuracy(
            model, tokenizer, examples, n_iter=n,
            device=device, max_new_tokens=args.max_new_tokens,
            desc=label,
        )
        all_results[n] = res
        print(f"\n  n_iter={n} ({label})")
        print(f"    Accuracy  : {res['accuracy']*100:.1f}%  ({res['correct']}/{res['total']})")
        print(f"    Parse fail: {res['parse_fail']}")

    # Summary
    print("\n" + "─" * 65)
    print("  Summary")
    print("─" * 65)
    baseline_acc = all_results[0]["accuracy"]
    print(f"  Baseline (n_iter=0): {baseline_acc*100:.1f}%")
    for n in iters_to_eval:
        if n == 0:
            continue
        acc = all_results[n]["accuracy"]
        delta = acc - baseline_acc
        sign = "✅" if delta > 0.005 else ("⚠️ " if abs(delta) <= 0.005 else "❌")
        print(f"  {sign} LoopedLM n_iter={n}: {acc*100:.1f}%  (Δ={delta*100:+.1f}%)")
    print("=" * 65)

    if args.output_json:
        # Strip per-example results for cleaner summary file
        summary = {
            str(n): {k: v for k, v in r.items() if k != "results"}
            for n, r in all_results.items()
        }
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[gsm8k] Results saved → {args.output_json}")


if __name__ == "__main__":
    main()
