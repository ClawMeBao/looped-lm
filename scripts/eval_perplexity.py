"""
eval_perplexity.py — Đánh giá perplexity của LoopedLM vs baseline

Mục tiêu Phase 0:
- So sánh PPL của: baseline (n_iter=0) vs loop (n_iter=1,2,3,4)
- Kiểm tra xem connect layer có học được gì hay không
- Chạy TRƯỚC khi train → PPL phải gần với baseline ban đầu
- Chạy SAU khi train một ít → PPL loop phải ≤ PPL baseline

Dataset: dùng WikiText-2 validation (nhỏ, nhanh đủ để test)

Chạy:
    python scripts/eval_perplexity.py --checkpoint path/to/connect.pt
    python scripts/eval_perplexity.py --baseline_only     # chỉ chạy baseline
"""

import argparse
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

from src import LoopedLM, LoopedLMConfig


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class TokenizedTextDataset(Dataset):
    """Tokenize và chunk text thành fixed-length blocks."""

    def __init__(self, texts: list[str], tokenizer, max_length: int = 512):
        all_ids = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id or 0)

        # Chunk thành blocks độ dài max_length
        self.blocks = [
            all_ids[i : i + max_length]
            for i in range(0, len(all_ids) - max_length, max_length)
        ]
        self.max_length = max_length

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        return torch.tensor(self.blocks[idx], dtype=torch.long)


def load_wikitext2_val(tokenizer, max_length: int = 512, max_samples: int = 200):
    """
    Load WikiText-2 validation set.
    Nếu datasets không có sẵn, dùng text mẫu.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [x["text"] for x in ds if len(x["text"].strip()) > 100]
        print(f"  Loaded WikiText-2 validation: {len(texts)} documents")
    except Exception as e:
        print(f"  ⚠️  Không load được WikiText-2 ({e})")
        print("  Dùng dummy text để test ...")
        texts = [
            "The history of artificial intelligence began in antiquity, " * 50,
            "Machine learning is a type of artificial intelligence that " * 50,
            "Deep learning models consist of multiple layers of neurons. " * 50,
        ]

    dataset = TokenizedTextDataset(texts, tokenizer, max_length=max_length)
    # Giới hạn số samples để chạy nhanh
    if len(dataset) > max_samples:
        indices = torch.randperm(len(dataset))[:max_samples].tolist()
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)
    return dataset


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_perplexity(
    model: LoopedLM,
    dataloader: DataLoader,
    device: torch.device,
    n_iter: int,
    desc: str = "",
) -> float:
    """
    Tính perplexity trên toàn dataloader với n_iter iterations.
    Temporarily overrides model.cfg.n_iter.
    """
    original_n_iter = model.cfg.n_iter
    model.cfg.n_iter = n_iter
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(dataloader, desc=desc, leave=False):
        input_ids = batch.to(device)
        labels    = input_ids.clone()

        outputs = model(input_ids=input_ids, labels=labels)
        loss    = outputs.loss

        # Đếm số tokens thực sự (trừ -100 nếu có)
        n_tokens = (labels != -100).sum().item() - labels.size(0)  # trừ first token
        total_loss   += loss.item() * n_tokens
        total_tokens += n_tokens

    model.cfg.n_iter = original_n_iter

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return math.exp(avg_loss)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",         default="Qwen/Qwen3-1.7B")
    p.add_argument("--checkpoint",    default=None,
                   help="Path to saved connect layer weights (.pt)")
    p.add_argument("--connect_type",  default="mlp", choices=["mlp", "gated"])
    p.add_argument("--loop_start",    type=int, default=8)
    p.add_argument("--loop_end",      type=int, default=20)
    p.add_argument("--max_iter",      type=int, default=4,
                   help="Test n_iter từ 0 đến max_iter")
    p.add_argument("--seq_len",       type=int, default=256)
    p.add_argument("--batch_size",    type=int, default=4)
    p.add_argument("--max_samples",   type=int, default=100)
    p.add_argument("--baseline_only", action="store_true",
                   help="Chỉ chạy baseline (n_iter=0)")
    p.add_argument("--dtype",         default="bfloat16",
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
    print("  LoopedLM — Perplexity Evaluation")
    print("=" * 60)

    # --- Load model ---
    cfg = LoopedLMConfig(
        loop_start   = args.loop_start,
        loop_end     = args.loop_end,
        n_iter       = 1,    # placeholder, sẽ override trong eval
        connect_type = args.connect_type,
    )
    model = LoopedLM.from_pretrained(args.model, cfg=cfg, torch_dtype=dtype)
    device = next(model.connect.parameters()).device

    # Load checkpoint nếu có
    if args.checkpoint:
        print(f"\nLoading connect layer from: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location=device)
        model.connect.load_state_dict(state)
        print("✅ Checkpoint loaded")

    # --- Load tokenizer và dataset ---
    print(f"\nLoading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading dataset ...")
    dataset = load_wikitext2_val(
        tokenizer,
        max_length  = args.seq_len,
        max_samples = args.max_samples,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"  Dataset size: {len(dataset)} samples × {args.seq_len} tokens\n")

    # --- Evaluate ---
    iter_range = [0] if args.baseline_only else range(args.max_iter + 1)

    results = {}
    for n in iter_range:
        label = f"baseline (n_iter=0, no connect)" if n == 0 else f"n_iter={n}"
        ppl = evaluate_perplexity(model, dataloader, device, n_iter=n, desc=label)
        results[n] = ppl
        print(f"  n_iter={n:2d}  PPL = {ppl:.2f}")

    # --- Summary ---
    print()
    print("─" * 40)
    print("Summary:")
    baseline_ppl = results.get(0, None)
    for n, ppl in results.items():
        if n == 0:
            print(f"  n_iter={n}: PPL={ppl:.2f}  ← baseline")
        else:
            delta = ppl - baseline_ppl if baseline_ppl else 0
            sign  = "+" if delta > 0 else ""
            print(f"  n_iter={n}: PPL={ppl:.2f}  ({sign}{delta:.2f} vs baseline)")

    if not args.baseline_only:
        best_n   = min(results, key=results.get)
        best_ppl = results[best_n]
        print(f"\n  Best: n_iter={best_n}, PPL={best_ppl:.2f}")

        if baseline_ppl and best_n > 0 and best_ppl < baseline_ppl:
            print("  ✅ Loop IMPROVES perplexity!")
        elif baseline_ppl and best_n > 0:
            print("  ⚠️  Loop không improve PPL — connect layer cần train thêm")

    print("=" * 60)


if __name__ == "__main__":
    main()
