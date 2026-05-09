"""
phase0/scripts/compare_generation.py

Compare baseline vs looped generation across a prompt set.

Example:
    python phase0/scripts/compare_generation.py \
      --checkpoint phase0/checkpoints/residual_seq512_3epoch/final_connect.pt \
      --connect_type residual \
      --n_iters 0,2,3,4 \
      --greedy --no_think \
      --max_new_tokens 256 \
      --output phase0/reports/generation_compare.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from transformers import AutoTokenizer

from phase0.src import Phase0Config, Phase0Model
from phase0.scripts.inference import build_prompt, generate, repetition_stats


DEFAULT_PROMPTS = [
    "Explain the history of artificial intelligence.",
    "What is deep learning? Give a concise explanation.",
    "Solve step by step: If a train travels 120 km in 2 hours, what is its average speed?",
    "Write a short Python function to reverse a string.",
    "Summarize why the sky appears blue.",
    "Compare supervised learning and reinforcement learning.",
    "Give three practical tips for debugging a machine learning training run.",
    "Translate to Vietnamese: Artificial intelligence is changing software development.",
    "Write a polite email asking for a project deadline extension.",
    "What are the main risks of overfitting in a neural network?",
]


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 — Compare baseline vs looped generation")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--connect_type", default="residual", choices=["residual", "mlp", "gated", "iter_aware"])
    p.add_argument("--loop_start", type=int, default=8)
    p.add_argument("--loop_end", type=int, default=20)
    p.add_argument("--n_iters", default="0,2,3,4",
                   help="Comma-separated loop depths to compare, e.g. 0,2,3,4")
    p.add_argument("--prompt_file", default=None,
                   help="Optional .txt/.jsonl prompt file. JSONL accepts fields: prompt, system.")
    p.add_argument("--output", default="phase0/reports/generation_compare.jsonl")
    p.add_argument("--system", default=None)
    p.add_argument("--max_prompts", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--no_think", action="store_true")
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])
    return p.parse_args()


def parse_n_iters(value: str) -> list[int]:
    n_iters = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        n = int(item)
        if n < 0:
            raise ValueError("n_iters must be >= 0")
        n_iters.append(n)
    if not n_iters:
        raise ValueError("n_iters must contain at least one value")
    return n_iters


def load_prompts(path: str | None, default_system: str | None) -> list[dict[str, str | None]]:
    if path is None:
        return [{"prompt": prompt, "system": default_system} for prompt in DEFAULT_PROMPTS]

    prompt_path = Path(path)
    prompts: list[dict[str, str | None]] = []
    if prompt_path.suffix.lower() == ".jsonl":
        with prompt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = row.get("prompt") or row.get("instruction") or row.get("question")
                if prompt:
                    prompts.append({"prompt": str(prompt), "system": row.get("system", default_system)})
    else:
        with prompt_path.open("r", encoding="utf-8") as f:
            for line in f:
                prompt = line.strip()
                if prompt:
                    prompts.append({"prompt": prompt, "system": default_system})

    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def no_think_format_stats(text: str) -> dict[str, bool]:
    has_open = "<think>" in text
    has_close = "</think>" in text
    close_after_open = has_open and has_close and text.find("</think>") > text.find("<think>")
    return {
        "has_think_open": has_open,
        "has_think_close": has_close,
        "think_closed_after_open": close_after_open,
        "format_valid": close_after_open,
    }


def main():
    args = parse_args()
    n_iters = parse_n_iters(args.n_iters)
    prompts = load_prompts(args.prompt_file, args.system)
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    cfg = Phase0Config(
        model_name=args.model,
        loop_start=args.loop_start,
        loop_end=args.loop_end,
        n_iter=max(n_iters),
        connect_type=args.connect_type,
    )

    print(f"[compare] prompts={len(prompts)} n_iters={n_iters}")
    print(f"[compare] checkpoint={args.checkpoint}")

    model = Phase0Model.from_pretrained(cfg, torch_dtype=dtype)
    device = next(model.connect.parameters()).device
    model.load_connect(args.checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[int, int] = defaultdict(int)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=not args.greedy,
        device=device,
    )

    with output_path.open("w", encoding="utf-8") as f:
        for prompt_idx, item in enumerate(prompts):
            prompt = str(item["prompt"])
            system = item.get("system")
            prompt_text = build_prompt(tokenizer, prompt, system, args.no_think)
            print(f"[compare] prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:80]}")

            for n_iter in n_iters:
                start = time.perf_counter()
                output = generate(model, tokenizer, prompt_text, n_iter=n_iter, **gen_kwargs)
                latency = time.perf_counter() - start

                rep = repetition_stats(output)
                fmt = no_think_format_stats(output) if args.no_think else {
                    "has_think_open": "<think>" in output,
                    "has_think_close": "</think>" in output,
                    "think_closed_after_open": True,
                    "format_valid": True,
                }
                tokens = rep["tokens"]
                row = {
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "system": system,
                    "n_iter": n_iter,
                    "latency_sec": latency,
                    "tokens_per_sec": tokens / max(latency, 1e-9),
                    "output": output,
                    **rep,
                    **fmt,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

                counts[n_iter] += 1
                for key in ("tokens", "distinct_ratio", "repeat_bigram_ratio",
                            "char_distinct_ratio", "repeat_trigram_ratio",
                            "latency_sec", "tokens_per_sec"):
                    summary[n_iter][key] += float(row[key])
                for key in ("has_think_open", "has_think_close",
                            "think_closed_after_open", "format_valid"):
                    summary[n_iter][key] += float(bool(row[key]))

    print("\nSummary")
    print("n_iter  count  fmt_ok  distinct  rep_bigram  rep_trigram  tok/s  latency")
    for n_iter in n_iters:
        count = max(counts[n_iter], 1)
        fmt_ok = summary[n_iter]["format_valid"] / count
        distinct = summary[n_iter]["distinct_ratio"] / count
        rep_bigram = summary[n_iter]["repeat_bigram_ratio"] / count
        rep_trigram = summary[n_iter]["repeat_trigram_ratio"] / count
        tok_s = summary[n_iter]["tokens_per_sec"] / count
        latency = summary[n_iter]["latency_sec"] / count
        print(
            f"{n_iter:>6}  {counts[n_iter]:>5}  {fmt_ok:>6.3f}  "
            f"{distinct:>8.3f}  {rep_bigram:>10.3f}  {rep_trigram:>11.3f}  "
            f"{tok_s:>5.1f}  {latency:>7.2f}"
        )

    print(f"\n[compare] wrote {output_path}")


if __name__ == "__main__":
    main()
