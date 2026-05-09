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
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from phase0.src import Phase0Model, Phase0Config
from common.data_utils import load_text_dataset, load_instruction_dataset, load_glm_dataset


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


def _unpack_batch(batch, device):
    if isinstance(batch, dict):
        return batch["input_ids"].to(device), batch["labels"].to(device)
    ids = batch.to(device)
    return ids, ids


def split_dataset(dataset, ratios=(0.8, 0.1, 0.1), seed=42):
    n = len(dataset)
    n_train = int(ratios[0] * n)
    n_eval = int(ratios[1] * n)
    n_test = n - n_train - n_eval
    return random_split(
        dataset, [n_train, n_eval, n_test],
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def eval_looped_details(model: Phase0Model, loader, device, n_iter: int, desc: str = "") -> dict:
    """Eval LoopedLM với n_iter iterations; returns PPL + connect diagnostics."""
    orig = model.cfg.n_iter
    model.cfg.n_iter = n_iter
    model.eval()

    logits_list, labels_list = [], []
    connect_sums: dict[str, float] = {}
    connect_count = 0
    for batch in tqdm(loader, desc=desc or f"  n_iter={n_iter}", leave=False, ncols=70):
        ids, labels = _unpack_batch(batch, device)
        out = model(input_ids=ids, labels=labels)
        logits_list.append(out.logits.cpu())
        labels_list.append(labels.cpu())
        connect_metrics = model.last_connect_metrics()
        if connect_metrics:
            for key, value in connect_metrics.items():
                connect_sums[key] = connect_sums.get(key, 0.0) + float(value)
            connect_count += 1

    model.cfg.n_iter = orig
    metrics = {
        key: value / max(connect_count, 1)
        for key, value in connect_sums.items()
    }
    metrics["ppl"] = _compute_ppl(logits_list, labels_list)
    return metrics


@torch.no_grad()
def eval_looped(model: Phase0Model, loader, device, n_iter: int, desc: str = "") -> float:
    return eval_looped_details(model, loader, device, n_iter, desc)["ppl"]


@torch.no_grad()
def eval_logit_kl_vs_n0(model: Phase0Model, loader, device, n_iter: int) -> float:
    """
    Streaming KL(logits_n0 || logits_n). Optional: expensive but useful to see
    whether extra loop iterations materially change the distribution.
    """
    orig = model.cfg.n_iter
    model.eval()
    total_kl, total_tok = 0.0, 0
    for batch in tqdm(loader, desc=f"  KL n0→n{n_iter}", leave=False, ncols=70):
        ids, labels = _unpack_batch(batch, device)
        valid = labels[:, 1:] != -100
        n_tok = int(valid.sum().item())
        if n_tok == 0:
            continue

        model.cfg.n_iter = 0
        logits0 = model(input_ids=ids).logits[:, :-1, :].float()
        model.cfg.n_iter = n_iter
        logitsn = model(input_ids=ids).logits[:, :-1, :].float()

        logp0 = torch.log_softmax(logits0, dim=-1)
        logpn = torch.log_softmax(logitsn, dim=-1)
        p0 = logp0.exp()
        kl_tok = (p0 * (logp0 - logpn)).sum(dim=-1)
        total_kl += kl_tok[valid].sum().item()
        total_tok += n_tok

    model.cfg.n_iter = orig
    return total_kl / max(total_tok, 1)


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
        ids, labels = _unpack_batch(batch, device)
        out = base(input_ids=ids, labels=labels)
        logits_list.append(out.logits.cpu())
        labels_list.append(labels.cpu())

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
    p.add_argument("--connect_type",       default="residual", choices=["residual", "mlp", "gated", "iter_aware"])
    p.add_argument("--loop_start",         type=int, default=8)
    p.add_argument("--loop_end",           type=int, default=20)
    p.add_argument("--max_iter",           type=int, default=4)
    p.add_argument("--seq_len",            type=int, default=256)
    p.add_argument("--batch_size",         type=int, default=4)
    p.add_argument("--max_samples",        type=int, default=100)
    p.add_argument("--dataset",            default="Jackrong/GLM-5.1-Reasoning-1M-Cleaned")
    p.add_argument("--dataset_type",       default="text",
                   choices=["instruction", "text", "glm"])
    p.add_argument("--split",              default="validation")
    p.add_argument("--subset",             default="eval",
                   choices=["train", "eval", "test"],
                   help="Internal subset for single-split chat datasets")
    p.add_argument("--no_think",           action="store_true")
    p.add_argument("--local_dataset_dir",  default="data/glm_dataset")
    p.add_argument("--true_baseline_only", action="store_true",
                   help="Chỉ đo true baseline, bỏ qua LoopedLM eval")
    p.add_argument("--skip_true_baseline", action="store_true",
                   help="Bỏ qua true baseline eval (nếu đã biết rồi)")
    p.add_argument("--logit_kl",           action="store_true",
                   help="Also compute KL(logits_n0 || logits_n) for n>=2. Costs extra VRAM/time.")
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

    if args.dataset_type == "instruction":
        full_ds = load_instruction_dataset(
            tokenizer,
            dataset_name=args.dataset,
            split=args.split,
            max_length=args.seq_len,
            max_samples=args.max_samples,
            no_think=args.no_think,
        )
        train_ds, eval_ds, test_ds = split_dataset(full_ds)
        dataset = {"train": train_ds, "eval": eval_ds, "test": test_ds}[args.subset]
    elif args.dataset_type == "glm":
        full_ds = load_glm_dataset(
            tokenizer,
            dataset_name=args.dataset,
            split="train",
            max_length=args.seq_len,
            max_samples=args.max_samples,
            no_think=args.no_think,
            local_dir=args.local_dataset_dir,
        )
        train_ds, eval_ds, test_ds = split_dataset(full_ds)
        dataset = {"train": train_ds, "eval": eval_ds, "test": test_ds}[args.subset]
    else:
        dataset = load_text_dataset(
            tokenizer,
            split=args.split,
            max_length=args.seq_len,
            max_samples=args.max_samples,
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    results = {}

    # ── 1. True baseline ──────────────────────────────────────────────
    if not args.skip_true_baseline:
        print("\n[1] TRUE BASELINE — Qwen3-1.7B original (no loop)")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        true_ppl = eval_true_baseline(args.model, loader, device, dtype)
        results["true_baseline"] = true_ppl
        print(f"    PPL = {true_ppl:.2f}")

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
    connect_results = {}
    kl_results = {}
    for n in range(1, args.max_iter + 1):
        details = eval_looped_details(model, loader, device, n_iter=n)
        ppl = details["ppl"]
        results[f"looped_n{n}"] = ppl
        connect_results[n] = details
        extra = ""
        if "update_norm_ratio_mean" in details:
            extra = (
                f"  update_ratio={details['update_norm_ratio_mean']:.5f}"
                f"  cosine={details['output_cosine_to_prev_mean']:.5f}"
            )
        if args.logit_kl and n >= 2:
            kl = eval_logit_kl_vs_n0(model, loader, device, n_iter=n)
            kl_results[n] = kl
            extra += f"  KL(n0||n{n})={kl:.5f}"
        print(f"    n_iter={n}  PPL={ppl:.2f}{extra}")

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
        details = connect_results.get(n, {})
        extra = ""
        if "update_norm_ratio_mean" in details:
            extra = (
                f"  update={details['update_norm_ratio_mean']:.5f}"
                f"  cos={details['output_cosine_to_prev_mean']:.5f}"
            )
        if n in kl_results:
            extra += f"  KL={kl_results[n]:.5f}"
        print(f"  {better} LoopedLM n_iter={n}         : PPL = {ppl:.2f}  "
              f"(Δ={delta:+.2f} vs {ref_label}){extra}")

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
