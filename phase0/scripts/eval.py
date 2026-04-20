"""
phase0/scripts/eval.py — Perplexity evaluation

So sánh 3 trường hợp:
  1. TRUE BASELINE  — Qwen3-1.7B gốc, HF model chạy trực tiếp
  2. LoopedLM n_iter=0 — sau fix Bug5 phải khớp với true baseline (~same PPL)
  3. LoopedLM n_iter=1..N — sau train connect layer

Chạy:
    # Chưa train — đo pre-train state
    python phase0/scripts/eval.py

    # Sau khi train
    python phase0/scripts/eval.py --checkpoint phase0/checkpoints/best_connect.pt

    # Chỉ đo true baseline nhanh
    python phase0/scripts/eval.py --true_baseline_only
"""

import argparse, sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from phase0.src import Phase0Model, Phase0Config
from common.data_utils import load_text_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_ppl(logits_list, labels_list) -> float:
    """Tính PPL từ accumulated logits + labels."""
    total_loss, total_tok = 0.0, 0
    for logits, labels in zip(logits_list, labels_list):
        # shift: predict token i+1 from position i
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        n_tok = (shift_labels != -100).sum().item()
        total_loss += loss.item()
        total_tok  += n_tok
    return math.exp(min(total_loss / max(total_tok, 1), 20))


@torch.no_grad()
def eval_looped(model: Phase0Model, loader, device, n_iter: int, desc: str = "") -> float:
    """Eval LoopedLM với n_iter iterations."""
    orig = model.cfg.n_iter
    model.cfg.n_iter = n_iter
    model.eval()

    logits_list, labels_list = [], []
    for batch in tqdm(loader, desc=desc or f"  n_iter={n_iter}", leave=False, ncols=70):
        ids = batch.to(device)
        out = model(input_ids=ids, labels=ids)
        logits_list.append(out.logits.cpu())
        labels_list.append(ids.cpu())

    model.cfg.n_iter = orig
    return _compute_ppl(logits_list, labels_list)


@torch.no_grad()
def eval_true_baseline(model_name: str, loader, device, dtype, desc: str = "") -> float:
    """
    Eval Qwen3 gốc — load HF model trực tiếp, không qua LoopedLM wrapper.
    Đây là ground truth PPL để so sánh.
    """
    print(f"  Loading HF baseline model: {model_name} ...")
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    base.eval()

    logits_list, labels_list = [], []
    for batch in tqdm(loader, desc=desc or "  true baseline", leave=False, ncols=70):
        ids = batch.to(device)
        out = base(input_ids=ids, labels=ids)
        logits_list.append(out.logits.cpu())
        labels_list.append(ids.cpu())

    del base
    torch.cuda.empty_cache()
    return _compute_ppl(logits_list, labels_list)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 — Perplexity Evaluation")
    p.add_argument("--model",              default="Qwen/Qwen3-1.7B")
    p.add_argument("--checkpoint",         default=None,
                   help="Path to saved connect layer (.pt)")
    p.add_argument("--connect_type",       default="gated", choices=["mlp", "gated"])
    p.add_argument("--loop_start",         type=int, default=8)
    p.add_argument("--loop_end",           type=int, default=20)
    p.add_argument("--max_iter",           type=int, default=4)
    p.add_argument("--seq_len",            type=int, default=256)
    p.add_argument("--batch_size",         type=int, default=4)
    p.add_argument("--max_samples",        type=int, default=100)
    p.add_argument("--true_baseline_only", action="store_true",
                   help="Chỉ đo true baseline, bỏ qua LoopedLM eval")
    p.add_argument("--skip_true_baseline", action="store_true",
                   help="Bỏ qua true baseline eval (nếu đã biết rồi)")
    p.add_argument("--dtype",              default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args  = parse_args()
    dtype = {"float32": torch.float32,
             "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]

    print("=" * 65)
    print("  Phase 0 — Perplexity Evaluation")
    print("=" * 65)

    # Tokenizer + dataset (dùng chung cho tất cả eval)
    print(f"\nTokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_text_dataset(
        tokenizer,
        max_length  = args.seq_len,
        max_samples = args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    results = {}

    # ── 1. True baseline ──────────────────────────────────────────────
    if not args.skip_true_baseline:
        print("\n[1] TRUE BASELINE — Qwen3-1.7B original (no loop)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        true_ppl = eval_true_baseline(args.model, loader, device, dtype)
        results["true_baseline"] = true_ppl
        print(f"    PPL = {true_ppl:.2f}  ← expected ~8–15 for Qwen3-1.7B on WikiText-2")

    if args.true_baseline_only:
        print("\n" + "=" * 65)
        return

    # ── 2. LoopedLM n_iter=0 (should match true baseline after Bug5 fix) ──
    print("\n[2] LoopedLM n_iter=0 (should ≈ true baseline after Bug5 fix)")
    cfg   = Phase0Config(
        model_name   = args.model,
        loop_start   = args.loop_start,
        loop_end     = args.loop_end,
        connect_type = args.connect_type,
    )
    model  = Phase0Model.from_pretrained(cfg, torch_dtype=dtype)
    device = next(model.connect.parameters()).device

    if args.checkpoint:
        model.load_connect(args.checkpoint)
        print(f"    Checkpoint loaded: {args.checkpoint}")

    ppl_0 = eval_looped(model, loader, device, n_iter=0)
    results["looped_n0"] = ppl_0

    if "true_baseline" in results:
        delta = ppl_0 - results["true_baseline"]
        ok = "✅ OK" if abs(delta) < 1.0 else "⚠️  MISMATCH — check implementation"
        print(f"    PPL = {ppl_0:.2f}  (Δ={delta:+.2f} vs true baseline)  {ok}")
    else:
        print(f"    PPL = {ppl_0:.2f}")

    # ── 3. LoopedLM n_iter=1..max_iter ───────────────────────────────
    print(f"\n[3] LoopedLM n_iter=1..{args.max_iter} (connect layer)")
    for n in range(1, args.max_iter + 1):
        ppl = eval_looped(model, loader, device, n_iter=n)
        results[f"looped_n{n}"] = ppl
        print(f"    n_iter={n}  PPL={ppl:.2f}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  Summary")
    print("─" * 65)

    true_ppl = results.get("true_baseline", None)
    ref_label = "true baseline" if true_ppl else "n_iter=0"
    ref_ppl   = true_ppl if true_ppl else results["looped_n0"]

    if true_ppl:
        print(f"  True baseline (HF model)  : PPL = {true_ppl:.2f}")
    print(f"  LoopedLM n_iter=0         : PPL = {results['looped_n0']:.2f}"
          + (f"  (Δ={results['looped_n0']-true_ppl:+.2f})" if true_ppl else ""))
    print()
    for n in range(1, args.max_iter + 1):
        ppl   = results[f"looped_n{n}"]
        delta = ppl - ref_ppl
        better = "✅" if ppl < ref_ppl else "  "
        print(f"  {better} LoopedLM n_iter={n}         : PPL = {ppl:.2f}  "
              f"(Δ={delta:+.2f} vs {ref_label})")

    best_n = min(
        (n for n in range(1, args.max_iter + 1)),
        key=lambda n: results[f"looped_n{n}"]
    )
    best_ppl = results[f"looped_n{best_n}"]
    print(f"\n  Best LoopedLM: n_iter={best_n}, PPL={best_ppl:.2f}")

    if best_ppl < ref_ppl:
        print(f"  ✅ Loop BEATS {ref_label}!")
    elif best_ppl < ref_ppl * 1.1:
        print(f"  ⚠️  Loop is close but not better than {ref_label}. Train more.")
    else:
        print(f"  ❌ Loop still worse than {ref_label}. Connect layer needs more training.")

    print("=" * 65)


if __name__ == "__main__":
    main()
