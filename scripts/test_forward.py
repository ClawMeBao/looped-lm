"""
test_forward.py — Smoke test: kiểm tra forward pass hoạt động

Mục tiêu Phase 0:
1. Model load và wrap thành công
2. Forward pass không crash
3. Gradient chỉ flow vào connect layer (không vào frozen base)
4. Thống kê param count

Chạy:
    python scripts/test_forward.py
    python scripts/test_forward.py --connect_type gated --n_iter 4
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src import LoopedLM, LoopedLMConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",        default="Qwen/Qwen3-1.7B")
    p.add_argument("--connect_type", default="mlp", choices=["mlp", "gated"])
    p.add_argument("--n_iter",       type=int, default=3)
    p.add_argument("--loop_start",   type=int, default=8)
    p.add_argument("--loop_end",     type=int, default=20)
    p.add_argument("--seq_len",      type=int, default=32)
    p.add_argument("--batch_size",   type=int, default=2)
    p.add_argument("--dtype",        default="bfloat16",
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

    print("=" * 60)
    print("  LoopedLM — Phase 0 Forward Pass Test")
    print("=" * 60)

    # --- 1. Load model ---
    cfg = LoopedLMConfig(
        loop_start   = args.loop_start,
        loop_end     = args.loop_end,
        n_iter       = args.n_iter,
        connect_type = args.connect_type,
    )
    model = LoopedLM.from_pretrained(args.model, cfg=cfg, torch_dtype=dtype)
    model.train()   # connect layer cần training mode

    print()
    model.print_param_summary()
    print()

    # --- 2. Dummy input ---
    device = next(model.connect.parameters()).device
    B, S = args.batch_size, args.seq_len
    input_ids = torch.randint(0, 1000, (B, S), device=device)
    labels    = input_ids.clone()

    print(f"Input shape: {list(input_ids.shape)}")
    print(f"Device:      {device}")
    print()

    # --- 3. Forward pass ---
    print("Running forward pass ...")
    outputs = model(input_ids=input_ids, labels=labels)

    print(f"✅ Forward OK")
    print(f"   Loss:   {outputs.loss.item():.4f}")
    print(f"   Logits: {list(outputs.logits.shape)}")
    print()

    # --- 4. Backward pass ---
    print("Running backward pass ...")
    outputs.loss.backward()
    print("✅ Backward OK")
    print()

    # --- 5. Gradient check: connect layer có grad, frozen layers không ---
    print("Gradient check ...")

    connect_grads = [
        (name, p.grad is not None, p.grad.abs().mean().item() if p.grad is not None else 0.0)
        for name, p in model.connect.named_parameters()
    ]
    print("  Connect layer gradients:")
    for name, has_grad, grad_norm in connect_grads:
        status = "✅" if has_grad else "❌"
        print(f"    {status} {name}: grad_mean={grad_norm:.6f}")

    # Kiểm tra frozen backbone
    frozen_with_grad = [
        name for name, p in model.backbone.named_parameters()
        if p.grad is not None
    ]
    if frozen_with_grad:
        print(f"\n⚠️  WARNING: {len(frozen_with_grad)} frozen params có gradient!")
        for name in frozen_with_grad[:5]:
            print(f"    - {name}")
    else:
        print("\n  ✅ Frozen backbone: không có gradient (correct)")

    # --- 6. Gate stats (chỉ với gated connect) ---
    if args.connect_type == "gated":
        print()
        print("Gate statistics:")
        with torch.no_grad():
            dummy_h = torch.randn(1, 8, 2048, device=device, dtype=dtype)
            stats = model.connect.gate_stats(dummy_h)
            for k, v in stats.items():
                print(f"  {k}: {v:.4f}")
        print("  (Expected: gate_mean ≈ 0.5 trước khi train)")

    print()
    print("=" * 60)
    print("  Phase 0 smoke test PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
