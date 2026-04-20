"""
train_connect.py — Train connect layer trên text corpus

Mục tiêu Phase 0:
- Train chỉ connect layer (frozen base)
- Curriculum: bắt đầu n_iter=1, tăng dần lên 3-4
- Log loss + PPL mỗi eval_steps bước
- Save checkpoint tốt nhất

Chạy:
    # Quick smoke test (5 phút)
    python scripts/train_connect.py --max_steps 200 --eval_steps 50

    # Full Phase 0 training
    python scripts/train_connect.py --max_steps 2000 --eval_steps 200 \
        --curriculum --connect_type gated
"""

import argparse
import sys
import os
import math
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from src import LoopedLM, LoopedLMConfig
from eval_perplexity import load_wikitext2_val, evaluate_perplexity


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",         default="Qwen/Qwen3-1.7B")
    p.add_argument("--connect_type",  default="mlp", choices=["mlp", "gated"])
    p.add_argument("--loop_start",    type=int, default=8)
    p.add_argument("--loop_end",      type=int, default=20)
    p.add_argument("--n_iter",        type=int, default=3)

    # Training
    p.add_argument("--max_steps",     type=int, default=1000)
    p.add_argument("--eval_steps",    type=int, default=100)
    p.add_argument("--batch_size",    type=int, default=2)
    p.add_argument("--seq_len",       type=int, default=256)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--warmup_steps",  type=int, default=50)
    p.add_argument("--grad_clip",     type=float, default=1.0)
    p.add_argument("--k_bptt",        type=int, default=2,
                   help="Truncated BPTT depth (None = full)")

    # Curriculum: bắt đầu với n_iter=1, tăng dần
    p.add_argument("--curriculum",    action="store_true",
                   help="Tăng n_iter dần từ 1 → n_iter qua quá trình train")

    # Output
    p.add_argument("--output_dir",    default="checkpoints")
    p.add_argument("--dtype",         default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])

    # Dataset
    p.add_argument("--max_samples",   type=int, default=500)

    return p.parse_args()


def get_curriculum_n_iter(step: int, max_steps: int, max_n_iter: int) -> int:
    """
    Curriculum schedule: tăng n_iter từ 1 → max_n_iter tuyến tính.
    Phase đầu (25% steps): n_iter=1 (warm up connect layer)
    Phase cuối: tăng dần lên max_n_iter

    Ví dụ với max_steps=1000, max_n_iter=3:
        step    0–250 → n_iter=1
        step  250–500 → n_iter=2
        step 500–1000 → n_iter=3
    """
    warmup_frac = 0.25
    if step / max_steps < warmup_frac:
        return 1
    progress = (step / max_steps - warmup_frac) / (1.0 - warmup_frac)
    n = 1 + int(progress * (max_n_iter - 1))
    return min(n, max_n_iter)


def main():
    args = parse_args()
    dtype_map = {
        "float32":  torch.float32,
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
    }
    dtype = dtype_map[args.dtype]
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  LoopedLM — Phase 0 Connect Layer Training")
    print("=" * 60)

    # --- Load model ---
    cfg = LoopedLMConfig(
        loop_start   = args.loop_start,
        loop_end     = args.loop_end,
        n_iter       = args.n_iter,
        connect_type = args.connect_type,
        k_bptt       = args.k_bptt,
    )
    model = LoopedLM.from_pretrained(args.model, cfg=cfg, torch_dtype=dtype)
    device = next(model.connect.parameters()).device
    model.train()
    model.connect.train()

    print()
    model.print_param_summary()

    # --- Tokenizer + dataset ---
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset ...")
    train_dataset = load_wikitext2_val(
        tokenizer, max_length=args.seq_len, max_samples=args.max_samples
    )
    eval_dataset = load_wikitext2_val(
        tokenizer, max_length=args.seq_len, max_samples=50
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False
    )
    print(f"  Train: {len(train_dataset)} samples | Eval: {len(eval_dataset)} samples")

    # --- Optimizer + Scheduler ---
    optimizer = optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # --- Baseline PPL (trước khi train) ---
    print("\nComputing baseline PPL (n_iter=0, no loop) ...")
    baseline_ppl = evaluate_perplexity(
        model, eval_loader, device, n_iter=0, desc="baseline"
    )
    print(f"  Baseline PPL: {baseline_ppl:.2f}")

    best_ppl  = float("inf")
    best_step = 0

    # --- Training loop ---
    print(f"\nStarting training (max_steps={args.max_steps}) ...\n")
    print(f"{'Step':>6} | {'n_iter':>6} | {'Loss':>8} | {'PPL':>8} | {'Time/s':>7}")
    print("-" * 50)

    data_iter = iter(train_loader)
    step = 0
    t0   = time.time()
    log_loss = 0.0

    while step < args.max_steps:
        # Reload dataloader nếu hết
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        input_ids = batch.to(device)
        labels    = input_ids.clone()

        # Curriculum: điều chỉnh n_iter theo schedule
        if args.curriculum:
            current_n_iter = get_curriculum_n_iter(step, args.max_steps, args.n_iter)
            model.cfg.n_iter = current_n_iter
        else:
            current_n_iter = args.n_iter

        # Forward + backward
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss    = outputs.loss
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()
        step += 1
        log_loss += loss.item()

        # --- Logging ---
        if step % 10 == 0:
            avg_loss = log_loss / 10
            ppl      = math.exp(min(avg_loss, 20))   # cap để tránh inf
            elapsed  = time.time() - t0
            print(
                f"{step:>6} | {current_n_iter:>6} | {avg_loss:>8.4f} | "
                f"{ppl:>8.2f} | {elapsed:>7.1f}s"
            )
            log_loss = 0.0

        # --- Eval ---
        if step % args.eval_steps == 0:
            print(f"\n  [Eval @ step {step}]")
            eval_ppl = evaluate_perplexity(
                model, eval_loader, device,
                n_iter=args.n_iter,
                desc=f"  eval n_iter={args.n_iter}"
            )
            print(f"  Eval PPL (n_iter={args.n_iter}): {eval_ppl:.2f}")
            print(f"  Baseline PPL:                   {baseline_ppl:.2f}")
            delta = eval_ppl - baseline_ppl
            sign  = "+" if delta >= 0 else ""
            print(f"  Delta vs baseline:              {sign}{delta:.2f}")

            if eval_ppl < best_ppl:
                best_ppl  = eval_ppl
                best_step = step
                ckpt_path = os.path.join(args.output_dir, "best_connect.pt")
                torch.save(model.connect.state_dict(), ckpt_path)
                print(f"  💾 Saved best checkpoint: {ckpt_path}")

            model.train()
            model.connect.train()
            print()

    # --- Final summary ---
    print("=" * 60)
    print(f"Training completed!")
    print(f"  Best eval PPL:     {best_ppl:.2f} @ step {best_step}")
    print(f"  Baseline PPL:      {baseline_ppl:.2f}")
    delta = best_ppl - baseline_ppl
    sign  = "+" if delta >= 0 else ""
    print(f"  Delta:             {sign}{delta:.2f}")

    if best_ppl < baseline_ppl:
        print("  ✅ Connect layer IMPROVED perplexity vs baseline!")
    else:
        print("  ⚠️  Connect layer chưa improve PPL — thử train thêm hoặc tăng lr")

    # Save final checkpoint
    final_path = os.path.join(args.output_dir, "final_connect.pt")
    torch.save(model.connect.state_dict(), final_path)
    print(f"\n  Final checkpoint: {final_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
