"""
phase0/scripts/inference.py — Text generation với Phase0Model

Chạy:
    python phase0/scripts/inference.py --checkpoint phase0/checkpoints/best_connect.pt
    python phase0/scripts/inference.py --checkpoint ... --compare --prompt "Once upon a time"
"""

import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from transformers import AutoTokenizer

from phase0.src import Phase0Model, Phase0Config


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 — Inference")
    p.add_argument("--model",          default="Qwen/Qwen3-1.7B")
    p.add_argument("--checkpoint",     default=None)
    p.add_argument("--connect_type",   default="gated", choices=["mlp", "gated"])
    p.add_argument("--loop_start",     type=int,   default=8)
    p.add_argument("--loop_end",       type=int,   default=20)
    p.add_argument("--n_iter",         type=int,   default=3)
    p.add_argument("--prompt",         default="The history of artificial intelligence")
    p.add_argument("--max_new_tokens", type=int,   default=100)
    p.add_argument("--temperature",    type=float, default=0.8)
    p.add_argument("--top_k",          type=int,   default=50)
    p.add_argument("--greedy",         action="store_true")
    p.add_argument("--compare",        action="store_true",
                   help="Side-by-side: baseline (n_iter=0) vs looped (n_iter=N)")
    p.add_argument("--dtype",          default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    return p.parse_args()


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens,
             n_iter, temperature, top_k, do_sample, device) -> str:
    orig = model.cfg.n_iter
    model.cfg.n_iter = n_iter
    model.eval()

    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    for _ in range(max_new_tokens):
        out    = model(input_ids=ids)
        logits = out.logits[:, -1, :]
        if temperature != 1.0:
            logits = logits / temperature
        if top_k > 0:
            thresh = torch.topk(logits, top_k).values[:, -1, None]
            logits = logits.masked_fill(logits < thresh, float("-inf"))
        probs      = torch.softmax(logits, dim=-1)
        next_tok   = torch.multinomial(probs, 1) if do_sample else logits.argmax(-1, keepdim=True)
        ids        = torch.cat([ids, next_tok], dim=-1)
        if next_tok.item() == tokenizer.eos_token_id:
            break

    model.cfg.n_iter = orig
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def main():
    args   = parse_args()
    dtype  = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[args.dtype]

    cfg    = Phase0Config(model_name=args.model, loop_start=args.loop_start,
                          loop_end=args.loop_end, n_iter=args.n_iter,
                          connect_type=args.connect_type)
    model  = Phase0Model.from_pretrained(cfg, torch_dtype=dtype)
    device = next(model.connect.parameters()).device

    if args.checkpoint:
        model.load_connect(args.checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    kwargs = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                  top_k=args.top_k, do_sample=not args.greedy, device=device)

    print(f'\nPrompt: "{args.prompt}"\n')

    if args.compare:
        print("── BASELINE (n_iter=0) " + "─" * 38)
        print(generate(model, tokenizer, args.prompt, n_iter=0, **kwargs))
        print(f"\n── LOOPED (n_iter={args.n_iter}) " + "─" * 38)
        print(generate(model, tokenizer, args.prompt, n_iter=args.n_iter, **kwargs))
    else:
        print(f"── OUTPUT (n_iter={args.n_iter}) " + "─" * 38)
        print(generate(model, tokenizer, args.prompt, n_iter=args.n_iter, **kwargs))


if __name__ == "__main__":
    main()
