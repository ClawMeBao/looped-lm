"""
phase0/scripts/test_forward.py — Smoke test

Chạy:
    python phase0/scripts/test_forward.py
    python phase0/scripts/test_forward.py --connect_type mlp
"""

import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from phase0.src import Phase0Model, Phase0Config


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 — Smoke Test")
    p.add_argument("--model",        default="Qwen/Qwen3-1.7B")
    p.add_argument("--connect_type", default="gated", choices=["mlp", "gated"])
    p.add_argument("--n_iter",       type=int, default=3)
    p.add_argument("--loop_start",   type=int, default=8)
    p.add_argument("--loop_end",     type=int, default=20)
    p.add_argument("--seq_len",      type=int, default=32)
    p.add_argument("--batch_size",   type=int, default=1)
    p.add_argument("--dtype",        default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    return p.parse_args()


def main():
    args  = parse_args()
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]

    print("=" * 60)
    print("  Phase 0 — Forward Pass Smoke Test")
    print("=" * 60)

    cfg = Phase0Config(
        model_name   = args.model,
        loop_start   = args.loop_start,
        loop_end     = args.loop_end,
        n_iter       = args.n_iter,
        connect_type = args.connect_type,
    )
    model  = Phase0Model.from_pretrained(cfg, torch_dtype=dtype)
    device = next(model.connect.parameters()).device
    model.train(); model.connect.train()

    print(); model.print_summary(); print()

    B, S      = args.batch_size, args.seq_len
    input_ids = torch.randint(0, 1000, (B, S), device=device)

    # --- Forward ---
    out = model(input_ids=input_ids, labels=input_ids)
    print(f"✅ Forward OK   loss={out.loss.item():.4f}  logits={list(out.logits.shape)}")

    # --- Backward ---
    out.loss.backward()
    print("✅ Backward OK")

    # --- Gradient check ---
    connect_ok = all(p.grad is not None for p in model.connect.parameters())
    print(f"✅ Connect grads: {'ALL present' if connect_ok else '❌ MISSING'}")

    frozen_with_grad = [n for n, p in model.bb.layers.named_parameters() if p.grad is not None]
    if frozen_with_grad:
        print(f"⚠️  {len(frozen_with_grad)} frozen params leaked gradient!")
    else:
        print("✅ Frozen backbone: no gradient (correct)")

    # Gate stats for gated connect
    if args.connect_type == "gated":
        with torch.no_grad():
            dummy = torch.randn(1, 8, cfg.loop_start * 2 or 2048,
                                device=device, dtype=dtype)
            # gate_stats needs matching d_model
            dummy = torch.randn(1, 8, model.bb.d_model, device=device, dtype=dtype)
            stats = model.connect.gate_stats(dummy)
        print(f"\nGate stats at init:")
        for k, v in stats.items():
            print(f"  {k}: {v:.4f}")
        print("  (Expected gate_mean ≈ 0.5)")

    print()
    print("=" * 60)
    print("  Smoke test PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
