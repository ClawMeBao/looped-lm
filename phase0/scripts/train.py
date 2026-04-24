"""
phase0/scripts/train.py

Chạy:
    python phase0/scripts/train.py
    python phase0/scripts/train.py --connect_type gated --epochs 5 --curriculum
    python phase0/scripts/train.py --dataset_type glm --no_think --epochs 3 --max_samples 10000

TensorBoard:
    tensorboard --logdir phase0/checkpoints/runs
"""

import argparse, sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from phase0.src import Phase0Model, Phase0Config
from common.data_utils import load_text_dataset, load_instruction_dataset, load_glm_dataset


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 — Train connect layer")
    p.add_argument("--model",          default="Qwen/Qwen3-1.7B")
    p.add_argument("--connect_type",   default="iter_aware", choices=["mlp", "gated", "iter_aware"])
    p.add_argument("--loop_start",     type=int,   default=8)
    p.add_argument("--loop_end",       type=int,   default=20)
    p.add_argument("--n_iter",         type=int,   default=4)
    p.add_argument("--epochs",         type=int,   default=3,
                   help="Number of training epochs")
    p.add_argument("--eval_every",     type=int,   default=1,
                   help="Run eval split every N epochs")
    p.add_argument("--log_steps",      type=int,   default=10,
                   help="Log to TensorBoard every N global steps")
    p.add_argument("--batch_size",     type=int,   default=2)
    p.add_argument("--seq_len",        type=int,   default=1024)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--warmup_ratio",   type=float, default=0.05,
                   help="Fraction of total steps for LR warmup")
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--k_bptt",         type=int,   default=2)
    p.add_argument("--aux_loss_weight",  type=float, default=0.3,
                   help="Weight of summed per-iteration auxiliary LM losses (0=disabled)")
    p.add_argument("--aux_loss_gamma",   type=float, default=0.5,
                   help="Geometric decay for earlier iterations in auxiliary loss")
    p.add_argument("--consistency_weight", type=float, default=0.05,
                   help="L2 convergence regularizer between loop iterations (0=disabled)")
    p.add_argument("--curriculum",     action="store_true",
                   help="Ramp n_iter from 1 -> n_iter across epochs")
    p.add_argument("--max_samples",    type=int,   default=None,
                   help="Limit total loaded samples (default: full dataset). "
                        "Recommended for GLM 1M-row dataset.")
    p.add_argument("--output_dir",     default="phase0/checkpoints")
    p.add_argument("--log_dir",        default=None,
                   help="TensorBoard log dir (default: output_dir/runs)")
    p.add_argument("--dtype",          default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--dataset",        default="Jackrong/GLM-5.1-Reasoning-1M-Cleaned",
                   help="HuggingFace dataset name or 'wikitext' for plain text")
    p.add_argument("--dataset_type",   default="glm",
                   choices=["instruction", "text", "glm"],
                   help="'instruction': Roman claude; 'glm': GLM reasoning; 'text': plain blocks")
    p.add_argument("--no_think",       action="store_true",
                   help="Strip <think> reasoning -- train on answer-only")
    p.add_argument("--local_dataset_dir", default="data/glm_dataset",
                   help="Local dir to cache GLM dataset (Arrow format). "
                        "Auto-download if not found, then save for reuse.")
    return p.parse_args()


def curriculum_n_iter(global_step: int, total_steps: int, max_n: int) -> int:
    """
    Step-based curriculum: ramp n_iter mượt từ 1 → max_n theo global step.

    Mịn hơn per-epoch khi số epoch nhỏ — không bao giờ skip n_iter value.
    Ví dụ (max_n=4, total_steps=100):
      step   0 →  24 : n=1
      step  25 →  49 : n=2
      step  50 →  74 : n=3
      step  75 → 100 : n=4
    """
    if total_steps <= 1:
        return max_n
    n = 1 + int(global_step / total_steps * max_n)
    return min(max_n, max(1, n))


def _unpack_batch(batch, device):
    """Return (input_ids, labels) from dict or tensor batch."""
    if isinstance(batch, dict):
        return batch["input_ids"].to(device), batch["labels"].to(device)
    ids = batch.to(device)
    return ids, ids


def split_dataset(dataset, ratios=(0.8, 0.1, 0.1), seed=42):
    """Split dataset into train/eval/test by ratios. Returns 3 Subsets."""
    n       = len(dataset)
    n_train = int(ratios[0] * n)
    n_eval  = int(ratios[1] * n)
    n_test  = n - n_train - n_eval
    return random_split(
        dataset, [n_train, n_eval, n_test],
        generator=torch.Generator().manual_seed(seed),
    )


@torch.no_grad()
def eval_ppl(model, loader, device, n_iter, desc="Eval") -> float:
    orig = model.cfg.n_iter
    model.cfg.n_iter = n_iter
    model.eval()
    total_loss, total_tok = 0.0, 0
    for batch in tqdm(loader, desc=desc, leave=False, unit="batch"):
        ids, labels = _unpack_batch(batch, device)
        out = model(input_ids=ids, labels=labels)
        n   = (labels[:, 1:] != -100).sum().item()
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
    log_dir = args.log_dir or os.path.join(args.output_dir, "runs")
    writer  = SummaryWriter(log_dir=log_dir)
    dtype   = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[args.dtype]

    print("=" * 60)
    print("  Phase 0 -- Connect Layer Training")
    print("=" * 60)

    cfg = Phase0Config(
        model_name   = args.model,
        loop_start   = args.loop_start,
        loop_end     = args.loop_end,
        n_iter       = args.n_iter,
        connect_type = args.connect_type,
        k_bptt       = args.k_bptt,
        aux_loss_weight    = args.aux_loss_weight,
        aux_loss_gamma     = args.aux_loss_gamma,
        consistency_weight = args.consistency_weight,
    )
    model  = Phase0Model.from_pretrained(cfg, torch_dtype=dtype)
    device = next(model.connect.parameters()).device
    model.print_summary()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- Load full dataset -------------------------------------------------
    if args.dataset_type == "instruction":
        full_ds = load_instruction_dataset(
            tokenizer,
            dataset_name = args.dataset,
            split        = "train",
            max_length   = args.seq_len,
            max_samples  = args.max_samples,
            no_think     = args.no_think,
        )
    elif args.dataset_type == "glm":
        glm_name = (args.dataset
                    if args.dataset != "Roman1111111/claude-opus-4.6-10000x"
                    else "Jackrong/GLM-5.1-Reasoning-1M-Cleaned")
        full_ds = load_glm_dataset(
            tokenizer,
            dataset_name     = glm_name,
            split            = "train",
            max_length       = args.seq_len,
            max_samples      = args.max_samples,
            no_think         = args.no_think,
            local_dir        = args.local_dataset_dir,
        )
    else:
        full_ds = load_text_dataset(
            tokenizer, max_length=args.seq_len,
            split="train", max_samples=args.max_samples,
        )

    # -- 80 / 10 / 10 split -----------------------------------------------
    train_ds, eval_ds, test_ds = split_dataset(full_ds)
    print(f"\n[data] Split -> train: {len(train_ds)}  "
          f"eval: {len(eval_ds)}  test: {len(test_ds)}\n")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    eval_dl  = DataLoader(eval_ds,  batch_size=args.batch_size, shuffle=False)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    steps_per_epoch = len(train_dl)
    total_steps     = args.epochs * steps_per_epoch
    warmup_steps    = max(1, int(total_steps * args.warmup_ratio))

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"Epochs: {args.epochs}  |  Steps/epoch: {steps_per_epoch}"
          f"  |  Total: {total_steps}  |  Warmup: {warmup_steps}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}\n")

    # -- Baseline ----------------------------------------------------------
    baseline_ppl = eval_ppl(model, eval_dl, device, n_iter=0, desc="Baseline")
    print(f"Baseline PPL (n_iter=0, all layers): {baseline_ppl:.2f}")
    if args.dataset_type != "text":
        mode = "no-think" if args.no_think else "thinking"
        print(f"  (instruction [{mode}] -- chat-format PPL)\n")
    else:
        print()
    writer.add_scalar("ppl/baseline", baseline_ppl, 0)

    best_ppl    = float("inf")
    global_step = 0

    # -- Training loop -----------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.connect.train()

        epoch_loss = 0.0
        log_loss   = 0.0
        log_cnt    = 0

        # cur_n shown in tqdm postfix; compute start value just for clarity
        pbar = tqdm(
            train_dl,
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="batch", dynamic_ncols=True,
        )
        for batch in pbar:
            global_step += 1
            ids, labels  = _unpack_batch(batch, device)

            # Update n_iter every step for smooth curriculum
            cur_n = curriculum_n_iter(global_step, total_steps, args.n_iter) \
                    if args.curriculum else args.n_iter
            model.cfg.n_iter = cur_n

            optimizer.zero_grad()
            out = model(input_ids=ids, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            loss_val    = out.loss.item()
            epoch_loss += loss_val
            log_loss   += loss_val
            log_cnt    += 1

            pbar.set_postfix(
                n=cur_n,
                loss=f"{loss_val:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            if global_step % args.log_steps == 0:
                avg = log_loss / log_cnt
                writer.add_scalar("train/loss",   avg,                        global_step)
                writer.add_scalar("train/ppl",    math.exp(min(avg, 20)),     global_step)
                writer.add_scalar("train/lr",     scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("train/n_iter", cur_n,                      global_step)
                log_loss, log_cnt = 0.0, 0

        pbar.close()

        avg_epoch = epoch_loss / steps_per_epoch
        writer.add_scalar("epoch/train_loss", avg_epoch, epoch)
        writer.add_scalar("epoch/train_ppl",  math.exp(min(avg_epoch, 20)), epoch)
        print(f"\n  Epoch {epoch}/{args.epochs} -- "
              f"avg loss: {avg_epoch:.4f}  "
              f"(train PPL~{math.exp(min(avg_epoch,20)):.2f})")

        # -- Eval on eval split --------------------------------------------
        if epoch % args.eval_every == 0:
            ppl  = eval_ppl(model, eval_dl, device, n_iter=args.n_iter,
                            desc=f"Eval [epoch {epoch}]")
            sign = "+" if ppl - baseline_ppl >= 0 else ""
            print(f"  [Eval]  PPL={ppl:.2f}  "
                  f"({sign}{ppl-baseline_ppl:.2f} vs baseline={baseline_ppl:.2f})")
            writer.add_scalar("eval/ppl", ppl, epoch)

            if ppl < best_ppl:
                best_ppl = ppl
                ckpt = os.path.join(args.output_dir, "best_connect.pt")
                model.save_connect(ckpt)
                print(f"  Best saved: {ckpt}")

        model.train()
        model.connect.train()

    # -- Final validation on held-out test split ---------------------------
    print("\n" + "=" * 60)
    val_ppl = eval_ppl(model, test_dl, device, n_iter=args.n_iter,
                       desc="Final Validate")
    sign = "+" if val_ppl - baseline_ppl >= 0 else ""
    print(f"Final Validate PPL: {val_ppl:.2f}  "
          f"({sign}{val_ppl-baseline_ppl:.2f} vs baseline={baseline_ppl:.2f})")
    writer.add_scalar("validate/ppl", val_ppl, args.epochs)

    final = os.path.join(args.output_dir, "final_connect.pt")
    model.save_connect(final)
    writer.close()

    print(f"\nDone. Best eval PPL: {best_ppl:.2f}  |  "
          f"Val PPL: {val_ppl:.2f}  |  Baseline: {baseline_ppl:.2f}")


if __name__ == "__main__":
    main()
