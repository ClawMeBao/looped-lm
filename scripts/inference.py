"""
inference.py — Interactive text generation với LoopedLM

Chạy sau khi train connect layer để test xem model generate ra gì.

Ví dụ:
    # Dùng connect layer đã train
    python scripts/inference.py \
        --model Qwen/Qwen3-1.7B \
        --checkpoint checkpoints/best_connect.pt \
        --connect_type mlp \
        --n_iter 3 \
        --prompt "The history of artificial intelligence" \
        --max_new_tokens 100

    # So sánh n_iter=0 (baseline) vs n_iter=3 (looped)
    python scripts/inference.py \
        --model Qwen/Qwen3-1.7B \
        --checkpoint checkpoints/best_connect.pt \
        --compare \
        --prompt "Once upon a time"
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer

from src import LoopedLM, LoopedLMConfig


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: LoopedLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    n_iter: int = 3,
    temperature: float = 1.0,
    top_k: int = 50,
    do_sample: bool = True,
    device: torch.device = None,
) -> str:
    """
    Autoregressive generation, token by token.
    Model.forward được gọi lại từ đầu mỗi bước (no KV cache — Phase 0).
    """
    original_n_iter = model.cfg.n_iter
    model.cfg.n_iter = n_iter
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    for _ in range(max_new_tokens):
        outputs = model(input_ids=input_ids)
        next_token_logits = outputs.logits[:, -1, :]  # [1, vocab]

        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        if top_k > 0:
            topk_vals, _ = torch.topk(next_token_logits, top_k)
            threshold = topk_vals[:, -1, None]
            next_token_logits = next_token_logits.masked_fill(
                next_token_logits < threshold, float("-inf")
            )

        probs = torch.softmax(next_token_logits, dim=-1)

        if do_sample:
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(probs, dim=-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_token], dim=-1)

        if next_token.item() == tokenizer.eos_token_id:
            break

    model.cfg.n_iter = original_n_iter
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           default="Qwen/Qwen3-1.7B")
    p.add_argument("--checkpoint",      default=None,
                   help="Path to connect layer .pt (optional)")
    p.add_argument("--connect_type",    default="mlp", choices=["mlp", "gated"])
    p.add_argument("--loop_start",      type=int, default=8)
    p.add_argument("--loop_end",        type=int, default=20)
    p.add_argument("--n_iter",          type=int, default=3)
    p.add_argument("--prompt",          default="The history of artificial intelligence")
    p.add_argument("--max_new_tokens",  type=int, default=100)
    p.add_argument("--temperature",     type=float, default=0.8)
    p.add_argument("--top_k",           type=int, default=50)
    p.add_argument("--do_sample",       action="store_true", default=True)
    p.add_argument("--greedy",          action="store_true",
                   help="Greedy decode (override do_sample)")
    p.add_argument("--compare",         action="store_true",
                   help="So sánh output baseline (n_iter=0) vs looped (n_iter=N)")
    p.add_argument("--dtype",           default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    return p.parse_args()


def main():
    args = parse_args()
    dtype_map = {
        "float32":  torch.float32,
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
    }
    dtype = dtype_map[args.dtype]
    do_sample = not args.greedy

    print("=" * 60)
    print("  LoopedLM — Inference")
    print("=" * 60)

    # --- Load model ---
    cfg = LoopedLMConfig(
        loop_start   = args.loop_start,
        loop_end     = args.loop_end,
        n_iter       = args.n_iter,
        connect_type = args.connect_type,
    )
    model = LoopedLM.from_pretrained(args.model, cfg=cfg, torch_dtype=dtype)
    device = next(model.connect.parameters()).device

    if args.checkpoint:
        print(f"\nLoading connect layer: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        model.connect.load_state_dict(state)
        print("✅ Checkpoint loaded")
    else:
        print("\n⚠️  Không có checkpoint — dùng random connect layer weights")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"\nPrompt: \"{args.prompt}\"")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}")
    print()

    if args.compare:
        # --- So sánh baseline vs looped ---
        print("─" * 60)
        print(f"  BASELINE (n_iter=0, no loop)")
        print("─" * 60)
        out_baseline = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            n_iter=0,
            temperature=args.temperature,
            top_k=args.top_k,
            do_sample=do_sample,
            device=device,
        )
        print(out_baseline)

        print()
        print("─" * 60)
        print(f"  LOOPED (n_iter={args.n_iter})")
        print("─" * 60)
        out_looped = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            n_iter=args.n_iter,
            temperature=args.temperature,
            top_k=args.top_k,
            do_sample=do_sample,
            device=device,
        )
        print(out_looped)
    else:
        # --- Single run ---
        print("─" * 60)
        print(f"  OUTPUT (n_iter={args.n_iter})")
        print("─" * 60)
        out = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens,
            n_iter=args.n_iter,
            temperature=args.temperature,
            top_k=args.top_k,
            do_sample=do_sample,
            device=device,
        )
        print(out)

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
