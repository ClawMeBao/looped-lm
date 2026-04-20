"""
phase0/scripts/eval.py — Perplexity evaluation, n_iter=0..max_iter

Chạy:
    python phase0/scripts/eval.py
    python phase0/scripts/eval.py --checkpoint phase0/checkpoints/best_connect.pt
"""

import argparse, sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from phase0.src import Phase0Model, Phase0Config
from common.data_utils import load_text_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 — Perplexity Evaluation")
    p.add_argument("--model",         default="Qwen/Qwen3-1.7B")
    p.add_argument("--checkpoint",    default=None)
    p.add_argument("--connect_type",  default="gated", choices=["mlp", "gated"])
    p.add_argument("--loop_start",    type=int,   default=8)
    p.add_argument("--loop_end",      type=int,   default=20)
    p.add_argument("--max_iter",      type=int,   default=4)
    p.add_argument("--seq_len",       type=int,   default=256)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--max_samples",   type=int,   default=100)
    p.add_argument("--dtype",         default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    return p.parse_args()


@torch.no_grad()
def eval_perplexity(model, loader, device, n_iter, desc="") -> float:
    orig = model.cfg.n_iter
    model.cfg.n_iter = n_iter
    model.eval()
    total_loss, total_tok = 0.0, 0
    for batch in tqdm(loader, desc=desc, leave=False):
        ids = batch.to(device)
        out = model(input_ids=ids, labels=ids)
        n   = (ids.shape[1] - 1) * ids.shape[0]
        total_loss += out.loss.item() * n
        total_tok  += n
    model.cfg.n_iter = orig
    return math.exp(min(total_loss / max(total_tok, 1), 20))


def main():
    args  = parse_args()
    dtype = {"float32": torch.float32,
             "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]

    print("=" * 60)
    print("  Phase 0 — Perplexity Evaluation")
    print("=" * 60)

    cfg = Phase0Config(
        model_name   = args.model,
        loop_start   = args.loop_start,
        loop_end     = args.loop_end,
        connect_type = args.connect_type,
    )
    model  = Phase0Model.from_pretrained(cfg, torch_dtype=dtype)
    device = next(model.connect.parameters()).device

    if args.checkpoint:
        model.load_connect(args.checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_text_dataset(tokenizer, max_length=args.seq_len,
                                max_samples=args.max_samples)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    results = {}
    for n in range(args.max_iter + 1):
        ppl = eval_perplexity(model, loader, device, n_iter=n,
                              desc=f"n_iter={n}")
        results[n] = ppl
        print(f"  n_iter={n:2d}  PPL={ppl:.2f}")

    print("\n─" * 40)
    baseline = results[0]
    for n, ppl in results.items():
        suffix = "  ← baseline" if n == 0 else \
                 f"  ({'+' if ppl-baseline>=0 else ''}{ppl-baseline:.2f})"
        print(f"  n_iter={n}: PPL={ppl:.2f}{suffix}")

    best_n = min(results, key=results.get)
    print(f"\n  Best: n_iter={best_n}, PPL={results[best_n]:.2f}")
    if best_n > 0 and results[best_n] < baseline:
        print("  ✅ Loop IMPROVES perplexity!")
    print("=" * 60)


if __name__ == "__main__":
    main()
