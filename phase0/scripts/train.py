"""
phase0/scripts/train.py

Chạy:
    python phase0/scripts/train.py
    python phase0/scripts/train.py --connect_type gated --max_steps 1000 --curriculum
"""

import argparse, sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from phase0.src import Phase0Model, Phase0Config
from common.data_utils import load_text_dataset, load_instruction_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 — Train connect layer")
    p.add_argument("--model",          default="Qwen/Qwen3-1.7B")
    p.add_argument("--connect_type",   default="gated", choices=["mlp", "gated"])
    p.add_argument("--loop_start",     type=int,   default=8)
    p.add_argument("--loop_end",       type=int,   default=20)
    p.add_argument("--n_iter",         type=int,   default=3)
    p.add_argument("--max_steps",      type=int,   default=1000)
    p.add_argument("--eval_steps",     type=int,   default=200)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--seq_len",        type=int,   default=512)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--warmup_steps",   type=int,   default=50)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--k_bptt",         type=int,   default=2)
    p.add_argument("--curriculum",     action="store_true",
                   help="Tăng n_iter dần từ 1 → n_iter")
    p.add_argument("--max_samples",    type=int,   default=500)
    p.add_argument("--output_dir",     default="phase0/checkpoints")
    p.add_argument("--dtype",          default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--dataset",        default="Roman1111111/claude-opus-4.6-10000x",
                   help="HuggingFace dataset name (instruction) or 'wikitext' (text)")
    p.add_argument("--dataset_type",   default="instruction",
                   choices=["instruction", "text"],
                   help="'instruction' dùng chat template + label mask; 'text' dùng plain blocks")
    return p.parse_args()


def curriculum_n_iter(step: int, max_steps: int, max_n: int) -> int:
    """Divide training into max_n equal stages: 25% each at n=1,2,...,max_n."""
    if step / max_steps < 0.25:
        return 1
    prog = (step / max_steps - 0.25) / 0.75
    # Bug fix: was `1 + int(prog * (max_n-1))` which never reached max_n
    # (requires prog=1.0 exactly, but last step has prog<1.0).
    # Fix: start post-warmup at n=2, use same step size.
    return min(max_n, 2 + int(prog * (max_n - 1)))


def _unpack_batch(batch, device):
    """Return (input_ids, labels) from either a tensor or dict batch."""
    if isinstance(batch, dict):
        ids    = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
    else:
        ids    = batch.to(device)
        labels = ids
    return ids, labels


def eval_ppl(model, loader, device, n_iter) -> float:
    orig = model.cfg.n_iter
    model.cfg.n_iter = n_iter
    model.eval()
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            ids, labels = _unpack_batch(batch, device)
            out = model(input_ids=ids, labels=labels)
            # Count non-masked label tokens (shift by 1 like the model does)
            n = (labels[:, 1:] != -100).sum().item()
            if n == 0:
                continue
            total_loss += out.loss.item() * n
            total_tok  += n
    model.cfg.n_iter = orig
    if total_tok == 0:
        return float("inf")
    return math.exp(min(total_loss / total_tok, 20))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    dtype = {"float32": torch.float32,
             "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]

    print("=" * 60)
    print("  Phase 0 — Connect Layer Training")
    print("=" * 60)

    cfg = Phase0Config(
        model_name     = args.model,
        loop_start     = args.loop_start,
        loop_end       = args.loop_end,
        n_iter         = args.n_iter,
        connect_type   = args.connect_type,
        k_bptt         = args.k_bptt,
    )
    model  = Phase0Model.from_pretrained(cfg, torch_dtype=dtype)
    device = next(model.connect.parameters()).device
    model.print_summary()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dataset_type == "instruction":
        train_ds = load_instruction_dataset(
            tokenizer,
            dataset_name = args.dataset,
            split        = "train[:-500]",
            max_length   = args.seq_len,
            max_samples  = args.max_samples,
        )
        eval_ds = load_instruction_dataset(
            tokenizer,
            dataset_name = args.dataset,
            split        = "train[-500:]",
            max_length   = args.seq_len,
            max_samples  = 50,
        )
    else:
        train_ds = load_text_dataset(tokenizer, max_length=args.seq_len,
                                     split="train",
                                     max_samples=args.max_samples)
        eval_ds  = load_text_dataset(tokenizer, max_length=args.seq_len,
                                     split="validation",
                                     max_samples=50)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    eval_dl  = DataLoader(eval_ds,  batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.trainable_parameters(),
                                  lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)

    # True baseline: n_iter=0 sau Bug5 fix = all 28 layers = HF model gốc
    baseline_ppl = eval_ppl(model, eval_dl, device, n_iter=0)
    print(f"\nBaseline PPL (n_iter=0, all 28 layers): {baseline_ppl:.2f}")
    if args.dataset_type == "text":
        print(f"  (Expected ~8-15 for Qwen3-1.7B on WikiText-2)\n")
    else:
        print(f"  (Instruction dataset — PPL reflects chat-format perplexity)\n")

    best_ppl, step, log_loss = float("inf"), 0, 0.0
    data_iter = iter(train_dl)
    t0 = time.time()

    print(f"{'Step':>6} | {'n_iter':>6} | {'Loss':>8} | {'PPL':>8} | {'Time':>6}")
    print("-" * 50)

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_dl)
            batch = next(data_iter)

        cur_n = curriculum_n_iter(step, args.max_steps, args.n_iter) \
                if args.curriculum else args.n_iter
        model.cfg.n_iter = cur_n
        model.train()
        model.connect.train()

        ids, labels = _unpack_batch(batch, device)
        optimizer.zero_grad()
        out = model(input_ids=ids, labels=labels)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        step += 1
        log_loss += out.loss.item()

        if step % 10 == 0:
            avg = log_loss / 10
            print(f"{step:>6} | {cur_n:>6} | {avg:>8.4f} | "
                  f"{math.exp(min(avg,20)):>8.2f} | {time.time()-t0:>5.0f}s")
            log_loss = 0.0

        if step % args.eval_steps == 0:
            ppl = eval_ppl(model, eval_dl, device, n_iter=args.n_iter)
            sign = "+" if ppl - baseline_ppl >= 0 else ""
            print(f"\n  [Eval @ {step}] PPL={ppl:.2f}  "
                  f"({sign}{ppl-baseline_ppl:.2f} vs baseline={baseline_ppl:.2f})")
            if ppl < best_ppl:
                best_ppl = ppl
                ckpt = os.path.join(args.output_dir, "best_connect.pt")
                model.save_connect(ckpt)
                print(f"  💾 Best saved: {ckpt}")
            print()
            model.train(); model.connect.train()

    print("=" * 60)
    print(f"Done. Best eval PPL: {best_ppl:.2f}  (baseline: {baseline_ppl:.2f})")
    final = os.path.join(args.output_dir, "final_connect.pt")
    model.save_connect(final)


if __name__ == "__main__":
    main()
