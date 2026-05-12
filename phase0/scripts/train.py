"""
phase0/scripts/train.py

Chạy:
    python phase0/scripts/train.py
    python phase0/scripts/train.py --connect_type residual --epochs 5 --curriculum
    python phase0/scripts/train.py --dataset_type glm --no_think --epochs 3 --max_samples 10000

TensorBoard:
    tensorboard --logdir phase0/checkpoints/runs
"""

import argparse, sys, os, math, json, time, itertools
from dataclasses import asdict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Disable HuggingFace tokenizer parallelism to avoid deadlocks with multi-worker DataLoader
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from phase0.src import Phase0Model, Phase0Config
from common.data_utils import load_text_dataset, load_instruction_dataset, load_glm_dataset

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            print("[warn] tensorboard not installed; scalar logging disabled")

        def add_scalar(self, *args, **kwargs):
            return None

        def close(self):
            return None


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 — Train connect layer")
    p.add_argument("--model",          default="Qwen/Qwen3-1.7B")
    p.add_argument("--connect_type",   default="residual", choices=["residual", "mlp", "gated", "iter_aware"])
    p.add_argument("--loop_start",     type=int,   default=8)
    p.add_argument("--loop_end",       type=int,   default=20)
    p.add_argument("--n_iter",         type=int,   default=4)
    p.add_argument("--epochs",         type=int,   default=3,
                   help="Number of training epochs")
    p.add_argument("--max_steps",      type=int,   default=None,
                   help="Stop after this many optimizer steps (overrides epoch budget)")
    p.add_argument("--eval_every",     type=int,   default=0,
                   help="Run eval split every N epochs (0=disabled; prefer --eval_steps)")
    p.add_argument("--eval_steps",     type=int,   default=250,
                   help="Run eval split every N optimizer steps (0=disabled)")
    p.add_argument("--eval_max_batches", type=int, default=None,
                   help="[Deprecated] Use --max_eval_batches instead.")
    p.add_argument("--max_eval_batches", type=int, default=500,
                   help="Cap eval DataLoader to this many batches. Default: 500. Set 0 for full eval.")
    p.add_argument("--no_eval_all_iters", action="store_true",
                   help="Only eval target n_iter instead of n_iter=0..N")
    p.add_argument("--log_steps",      type=int,   default=20,
                   help="Log to TensorBoard every N global steps")
    p.add_argument("--save_every",     type=int,   default=100,
                   help="Save a step checkpoint every N global steps (0=disabled)")
    p.add_argument("--resume",         default=None,
                   help="Path to a step checkpoint (.pt) to resume training from. "
                        "If omitted, auto-detects output_dir/resume.pt when it exists.")
    p.add_argument("--extra_steps",    type=int,   default=None,
                   help="Train exactly N more optimizer steps from the resume checkpoint "
                        "and then stop. Creates a fresh cosine LR schedule over extra_steps. "
                        "Ideal for Kaggle: resume where last session left off without "
                        "needing to know the current global_step. "
                        "Example: --extra_steps 500")
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--seq_len",        type=int,   default=512)
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--warmup_ratio",   type=float, default=0.05,
                   help="Fraction of total steps for LR warmup")
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--k_bptt",         type=int,   default=2)
    p.add_argument("--aux_loss_weight",  type=float, default=0.0,
                   help="Weight of summed per-iteration auxiliary LM losses (0=disabled)")
    p.add_argument("--aux_loss_gamma",   type=float, default=0.5,
                   help="Geometric decay for earlier iterations in auxiliary loss")
    p.add_argument("--consistency_weight", type=float, default=0.0,
                   help="Cosine-similarity convergence regularizer between loop iterations (0=disabled)")
    p.add_argument("--curriculum",     action="store_true",
                   help="Ramp n_iter from 2 -> n_iter across steps")
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
                   help="Strip reasoning content while preserving Qwen's empty <think></think> wrapper")
    p.add_argument("--filter_think_len", type=int, default=None,
                   help="Skip examples whose <think>…</think> block exceeds this many characters. "
                        "Auto-computed when --no_think is False and this is unset: "
                        "(seq_len - 200) * 3.5 (keeps ~58%% of GLM at seq_len=2048). "
                        "Set to 0 to disable filtering entirely.")
    p.add_argument("--num_workers",    type=int,   default=12,
                   help="DataLoader worker processes (default: 12)")
    p.add_argument("--local_dataset_dir", default="data/glm_dataset",
                   help="Local dir to cache GLM dataset (Arrow format). "
                        "Auto-download if not found, then save for reuse.")
    p.add_argument("--skip_baseline_eval", action="store_true",
                   help="Skip baseline PPL eval at training start (saves ~5-10 min). "
                        "Use --baseline_ppl to inject a known value for logging.")
    p.add_argument("--baseline_ppl",    type=float, default=None,
                   help="Known baseline PPL to use when --skip_baseline_eval is set. "
                        "Used only for logging/comparison display.")
    p.add_argument("--attn_impl",       default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"],
                   help="Attention backend. 'sdpa' (default): PyTorch built-in, no extra "
                        "package needed, uses flash attention kernel automatically. "
                        "'flash_attention_2': explicit FA2, requires flash-attn package "
                        "(only prebuilt for torch<=2.7; use sdpa for newer torch). "
                        "'eager': standard attention, no optimization.")
    p.add_argument("--flash_attn",      action="store_true",
                   help="[Deprecated] Alias for --attn_impl flash_attention_2.")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="Recompute activations during backward instead of storing them. "
                        "~2× slower backward but reduces activation VRAM by ~10-20×. "
                        "Enables seq_len=4096 with batch_size=2 on 16GB GPU.")
    p.add_argument("--loss_chunk_size",  type=int,   default=512,
                   help="Compute LM cross-entropy in chunks of this many tokens to avoid "
                        "materialising full [B, S, vocab] logits tensor (~2.4 GB at S=4096). "
                        "Set 0 to disable. Default: 512.")
    p.add_argument("--grad_accum",      type=int,   default=1,
                   help="Gradient accumulation steps. Effective batch = batch_size × grad_accum. "
                        "Use to increase effective batch without more VRAM. Default: 1 (disabled).")
    return p.parse_args()


def curriculum_n_iter(global_step: int, total_steps: int, max_n: int) -> int:
    """
    Step-based curriculum: ramp n_iter mượt từ 2 → max_n theo global step.

    Mịn hơn per-epoch khi số epoch nhỏ — không bao giờ skip n_iter value.
    Ví dụ (max_n=4, total_steps=100):
      step   0 →  33 : n=2
      step  34 →  66 : n=3
      step  67 → 100 : n=4
    """
    if max_n <= 1:
        return 1
    if total_steps <= 1:
        return max_n
    span = max_n - 1
    n = 2 + int(global_step / total_steps * span)
    return min(max_n, max(2, n))


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


def save_checkpoint_meta(path: str, cfg: Phase0Config, args, **extra) -> None:
    meta_path = path + ".meta.json"
    cfg_dict = asdict(cfg)
    cfg_dict["n_iter"] = args.n_iter
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cfg": cfg_dict,
                "args": vars(args),
                **extra,
            },
            f,
            indent=2,
            sort_keys=True,
        )


def dataset_diagnostics(dataset, name: str, max_items: int = 256) -> dict[str, float]:
    """Print cheap label/truncation diagnostics for tokenized datasets."""
    n = min(len(dataset), max_items)
    if n == 0:
        print(f"[data:{name}] empty dataset")
        return {}

    zero_grad = 0
    assistant_tokens = 0
    total_tokens = 0
    likely_truncated = 0
    end_think_labels = 0

    for i in range(n):
        item = dataset[i]
        if not isinstance(item, dict):
            continue
        labels = item["labels"]
        input_ids = item["input_ids"]
        valid = labels != -100
        valid_count = int(valid.sum().item())
        zero_grad += int(valid_count == 0)
        assistant_tokens += valid_count
        total_tokens += int(labels.numel())
        if input_ids.numel() > 0 and labels[-1].item() != -100:
            likely_truncated += 1
        end_think_labels += int((labels == 151668).any().item())

    if total_tokens == 0:
        return {}
    stats = {
        "checked": float(n),
        "assistant_token_ratio": assistant_tokens / total_tokens,
        "zero_grad_ratio": zero_grad / n,
        "likely_truncated_ratio": likely_truncated / n,
        "end_think_labeled_ratio": end_think_labels / n,
    }
    print(
        f"[data:{name}] checked={n}  "
        f"assistant_token_ratio={stats['assistant_token_ratio']:.3f}  "
        f"zero_grad={stats['zero_grad_ratio']:.3f}  "
        f"likely_truncated={stats['likely_truncated_ratio']:.3f}  "
        f"end_think_labeled={stats['end_think_labeled_ratio']:.3f}"
    )
    return stats


def warn_chat_format_risks(args, train_stats: dict[str, float], eval_stats: dict[str, float]) -> None:
    """Warn about label-format risks that make think/no_think training misleading."""
    if args.dataset_type == "text":
        return

    end_ratio = min(
        train_stats.get("end_think_labeled_ratio", 0.0),
        eval_stats.get("end_think_labeled_ratio", 0.0),
    )
    trunc_ratio = max(
        train_stats.get("likely_truncated_ratio", 0.0),
        eval_stats.get("likely_truncated_ratio", 0.0),
    )

    if args.no_think and end_ratio < 0.9:
        print(
            "[warn:data] no_think mode should label Qwen's empty </think> wrapper "
            f"almost always, but observed min ratio={end_ratio:.3f}. "
            "Check tokenizer/template compatibility before training."
        )
    if not args.no_think and end_ratio < 0.5:
        print(
            "[warn:data] thinking mode rarely reaches </think> "
            f"(min labeled ratio={end_ratio:.3f}). Reasoning traces are likely "
            f"truncated at seq_len={args.seq_len}; increase --seq_len, use "
            "--no_think, or filter long reasoning examples."
        )
    if trunc_ratio > 0.5:
        print(
            "[warn:data] many assistant spans reach the sequence boundary "
            f"(max likely_truncated={trunc_ratio:.3f}). PPL may be dominated by "
            "truncated completions; consider larger --seq_len or shorter samples."
        )


@torch.no_grad()
def eval_ppl(model, loader, device, n_iter, desc="Eval") -> tuple[float, float]:
    """Returns (avg_loss, ppl)."""
    orig = model.cfg.n_iter
    orig_aux = model.cfg.aux_loss_weight
    orig_consistency = model.cfg.consistency_weight
    model.cfg.n_iter = n_iter
    model.cfg.aux_loss_weight = 0.0
    model.cfg.consistency_weight = 0.0
    model.eval()
    total_loss, total_tok = 0.0, 0
    connect_sums: dict[str, float] = {}
    connect_count = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc=desc, leave=False, unit="batch")):
        if getattr(model, "_eval_max_batches", None) is not None and batch_idx >= model._eval_max_batches:
            break
        ids, labels = _unpack_batch(batch, device)
        out = model(input_ids=ids, labels=labels)
        n   = (labels[:, 1:] != -100).sum().item()
        if n == 0:
            continue
        total_loss += out.loss.item() * n
        total_tok  += n
        connect_metrics = model.last_connect_metrics()
        if connect_metrics:
            for key, value in connect_metrics.items():
                connect_sums[key] = connect_sums.get(key, 0.0) + float(value)
            connect_count += 1
    model.cfg.n_iter = orig
    model.cfg.aux_loss_weight = orig_aux
    model.cfg.consistency_weight = orig_consistency
    if total_tok == 0:
        model._last_eval_connect_metrics = {}
        return float("inf"), float("inf")
    model._last_eval_connect_metrics = {
        key: value / max(connect_count, 1)
        for key, value in connect_sums.items()
    }
    avg_loss = total_loss / total_tok
    return avg_loss, math.exp(min(avg_loss, 20))


def eval_suite(model, loader, device, args, baseline_ppl, writer, global_step: int) -> tuple[float, float]:
    """Eval target n_iter, optionally all depths, and log by global step."""
    if args.no_eval_all_iters:
        eval_iters = [args.n_iter]
    else:
        eval_iters = list(range(0, args.n_iter + 1))

    target_loss, target_ppl = float("inf"), float("inf")
    best_iter, best_ppl = None, float("inf")
    model._eval_max_batches = args.eval_max_batches

    for n_iter in eval_iters:
        loss, ppl = eval_ppl(model, loader, device, n_iter=n_iter,
                             desc=f"Eval n={n_iter} [step {global_step}]")
        connect_metrics = getattr(model, "_last_eval_connect_metrics", {})
        writer.add_scalar(f"eval/loss_n{n_iter}", loss, global_step)
        writer.add_scalar(f"eval/ppl_n{n_iter}", ppl, global_step)
        writer.add_scalar(f"eval/delta_vs_baseline_n{n_iter}",
                          ppl - baseline_ppl, global_step)
        for key, value in connect_metrics.items():
            writer.add_scalar(f"eval/connect/{key}_n{n_iter}", value, global_step)
        if n_iter == args.n_iter:
            target_loss, target_ppl = loss, ppl
            writer.add_scalar("eval/loss", loss, global_step)
            writer.add_scalar("eval/ppl", ppl, global_step)
            writer.add_scalar("eval/delta_vs_baseline", ppl - baseline_ppl, global_step)
            for key, value in connect_metrics.items():
                writer.add_scalar(f"eval/connect/{key}", value, global_step)
        if ppl < best_ppl:
            best_iter, best_ppl = n_iter, ppl

    if best_iter is not None:
        writer.add_scalar("eval/best_n_iter", best_iter, global_step)
        writer.add_scalar("eval/best_ppl", best_ppl, global_step)

    model._eval_max_batches = None
    return target_loss, target_ppl


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Merge deprecated --eval_max_batches into --max_eval_batches
    if args.eval_max_batches is not None and args.max_eval_batches == 500:
        args.max_eval_batches = args.eval_max_batches
    if args.max_eval_batches == 0:
        args.max_eval_batches = None   # 0 → full eval

    # Auto-detect resume checkpoint when --resume is not explicitly provided
    if not args.resume:  # None or empty string both suppress
        auto_resume = os.path.join(args.output_dir, "resume.pt")
        if os.path.exists(auto_resume) and args.resume is None:
            args.resume = auto_resume
            print(f"[resume] Auto-detected checkpoint: {auto_resume}")
            print(f"         Pass --resume '' to start fresh.\n")

    log_dir = args.log_dir or os.path.join(args.output_dir, "runs")
    writer  = SummaryWriter(log_dir=log_dir)
    dtype   = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[args.dtype]

    # -- Pre-flight validation: thinking mode requires large seq_len ----------
    if args.dataset_type != "text" and not args.no_think:
        if args.seq_len < 1024:
            raise ValueError(
                f"Thinking mode requires --seq_len >= 1024 (got {args.seq_len}). "
                "GLM thinking chains: p50≈1675 tokens, p90≈5514 tokens. "
                "Only 9.4% of chains fit at seq_len=512. "
                "Recommended: --seq_len 2048 (fits 58%) or --seq_len 4096 (fits 86%). "
                "To train without thinking, add --no_think."
            )
        if args.seq_len < 2048:
            print(
                f"[warn:think] seq_len={args.seq_len} is marginal for thinking mode. "
                "Only ~20% of GLM thinking chains fit. "
                "Recommend --seq_len 2048 (58% fit) or --seq_len 4096 (86% fit)."
            )

    # -- Auto-compute max_think_chars ----------------------------------------
    # When thinking mode and no explicit filter: auto-set to (seq_len - 200) * 3.5
    # so that examples fitting the budget pass; truncated-dominant ones are skipped.
    if args.dataset_type != "text" and not args.no_think:
        if args.filter_think_len is None:
            args.filter_think_len = int((args.seq_len - 200) * 3.5)
            print(f"[data] Auto filter_think_len={args.filter_think_len} chars "
                  f"(seq_len={args.seq_len}). Override with --filter_think_len N or "
                  "--filter_think_len 0 to disable.")
        elif args.filter_think_len == 0:
            args.filter_think_len = None  # 0 means disable

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
        aux_loss_weight       = args.aux_loss_weight,
        aux_loss_gamma        = args.aux_loss_gamma,
        consistency_weight    = args.consistency_weight,
        gradient_checkpointing = args.gradient_checkpointing,
        loss_chunk_size       = args.loss_chunk_size,
    )
    if args.n_iter < 2:
        raise ValueError("Training connect layer requires --n_iter >= 2; n_iter=1 is baseline pass-through.")
    model  = Phase0Model.from_pretrained(cfg, torch_dtype=dtype,
                                         attn_implementation="flash_attention_2" if args.flash_attn else args.attn_impl)
    device = next(model.connect.parameters()).device
    model.print_summary()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- Load full dataset -------------------------------------------------
    if args.dataset_type == "instruction":
        full_ds = load_instruction_dataset(
            tokenizer,
            dataset_name    = args.dataset,
            split           = "train",
            max_length      = args.seq_len,
            max_samples     = args.max_samples,
            no_think        = args.no_think,
            max_think_chars = args.filter_think_len,
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
            max_think_chars  = args.filter_think_len,
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
    train_diag = dataset_diagnostics(train_ds, "train")
    eval_diag = dataset_diagnostics(eval_ds, "eval")
    warn_chat_format_risks(args, train_diag, eval_diag)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    eval_dl  = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    # Cap eval to max_eval_batches steps to keep eval fast.
    # Overrides --eval_max_batches if set to a smaller value.
    _eval_cap = args.max_eval_batches
    if _eval_cap is not None and _eval_cap < len(eval_dl):
        from torch.utils.data import Subset
        eval_ds  = Subset(eval_ds, list(range(_eval_cap * args.batch_size)))
        eval_dl  = DataLoader(
            eval_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
            persistent_workers=(args.num_workers > 0),
        )
        print(f"[data] Eval capped to {_eval_cap} batches ({len(eval_ds)} samples)")
    test_dl  = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    steps_per_epoch = len(train_dl)
    total_steps     = args.max_steps or (args.epochs * steps_per_epoch)
    warmup_steps    = max(1, int(total_steps * args.warmup_ratio))

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01,
        fused=torch.cuda.is_available(),   # fused kernel: 1 launch vs 3 per param tensor
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    max_epochs = math.ceil(total_steps / steps_per_epoch)
    print(f"Epochs: {max_epochs}  |  Steps/epoch: {steps_per_epoch}"
          f"  |  Total: {total_steps}  |  Warmup: {warmup_steps}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}\n")

    # -- Baseline ----------------------------------------------------------
    model._eval_max_batches = args.eval_max_batches
    if args.skip_baseline_eval:
        if args.baseline_ppl is not None:
            baseline_ppl  = args.baseline_ppl
            baseline_loss = math.log(baseline_ppl)
            print(f"Baseline PPL (provided): {baseline_ppl:.2f}  [skipped eval]")
        else:
            baseline_ppl  = float("inf")
            baseline_loss = float("inf")
            print("Baseline PPL: skipped (use --baseline_ppl N to inject known value)")
    else:
        baseline_loss, baseline_ppl = eval_ppl(model, eval_dl, device, n_iter=0, desc="Baseline")
        print(f"Baseline PPL (n_iter=0, all layers): {baseline_ppl:.2f}  loss={baseline_loss:.4f}")
        if args.dataset_type != "text":
            mode = "no-think" if args.no_think else "thinking"
            print(f"  (instruction [{mode}] -- chat-format PPL)")
    model._eval_max_batches = None
    print()
    writer.add_scalar("ppl/baseline", baseline_ppl, 0)
    writer.add_scalar("eval/ppl_n0", baseline_ppl, 0)

    best_ppl       = float("inf")
    global_step    = 0
    start_epoch    = 1
    resume_batch   = 0   # micro-batches already processed in start_epoch

    # -- Resume from step checkpoint ---------------------------------------
    if args.resume:
        ckpt_data = torch.load(args.resume, map_location=device)
        model.connect.load_state_dict(ckpt_data["connect"])
        optimizer.load_state_dict(ckpt_data["optimizer"])
        scheduler.load_state_dict(ckpt_data["scheduler"])
        global_step  = ckpt_data.get("global_step", 0)
        start_epoch  = ckpt_data.get("epoch", 1)
        best_ppl     = ckpt_data.get("best_ppl", float("inf"))
        resume_batch = ckpt_data.get("batches_in_epoch", 0)
        print(f"[resume] Loaded checkpoint from '{args.resume}' "
              f"(step={global_step}, epoch={start_epoch}, best_ppl={best_ppl:.2f}, "
              f"skip_batches={resume_batch})")

    # -- Extra-steps: extend training by N more steps from resume point ----
    if args.extra_steps is not None:
        if global_step == 0 and not args.resume:
            # No checkpoint → treat extra_steps as plain max_steps
            print(f"[extra_steps] No checkpoint found; treating as --max_steps {args.extra_steps}")
            args.max_steps = args.extra_steps
        else:
            args.max_steps = global_step + args.extra_steps
            total_steps = args.max_steps           # update for curriculum ramp
            # Fresh cosine schedule scoped to the extra window
            new_warmup = max(1, int(args.extra_steps * args.warmup_ratio))
            scheduler  = get_cosine_schedule_with_warmup(optimizer, new_warmup, args.extra_steps)
            # Ensure epoch loop runs long enough to consume extra_steps
            max_epochs = start_epoch + math.ceil(args.extra_steps / steps_per_epoch) + 1
            print(
                f"[extra_steps] Resume step={global_step}  "
                f"extra_steps={args.extra_steps}  → stop at step {args.max_steps}\n"
                f"             Fresh cosine LR: warmup={new_warmup}, total={args.extra_steps}  "
                f"epochs budget={max_epochs - start_epoch}"
            )


    # -- Training loop -----------------------------------------------------
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        model.connect.train()

        epoch_loss = 0.0
        epoch_steps = 0
        log_loss_tokens = 0.0
        log_lm_loss_tokens = 0.0
        log_tokens = 0
        log_cnt = 0
        log_start = time.perf_counter()
        # Gradient accumulation state
        accum_micro  = 0    # micro-batches since last optimizer step
        _acc_loss    = 0.0  # accumulated loss sum (unscaled)
        _acc_lm_loss = 0.0
        _acc_tokens  = 0
        optimizer.zero_grad()

        # Micro-batches processed so far this epoch (for resume tracking)
        batches_in_epoch = 0

        def _do_optimizer_step(cur_n, epoch):
            """Clip + step + zero. Returns grad_norm."""
            nonlocal accum_micro, _acc_loss, _acc_lm_loss, _acc_tokens
            gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accum_micro = 0
            _acc_loss = 0.0; _acc_lm_loss = 0.0; _acc_tokens = 0
            return gn

        # Skip batches already processed in this epoch when resuming mid-epoch.
        # resume_batch is only non-zero on the first resumed epoch; clear after.
        skip_n = resume_batch if epoch == start_epoch else 0
        resume_batch = 0   # only skip once

        dl_iter = iter(train_dl)
        if skip_n > 0:
            print(f"  [resume] Skipping {skip_n} already-processed micro-batches in epoch {epoch}…")
            consumed = sum(1 for _ in itertools.islice(dl_iter, skip_n))
            batches_in_epoch = consumed
            print(f"  [resume] Skipped {consumed} batches, resuming from batch {consumed+1}")

        # cur_n shown in tqdm postfix; compute start value just for clarity
        pbar = tqdm(
            dl_iter,
            desc=f"Epoch {epoch}/{max_epochs}",
            unit="batch", dynamic_ncols=True,
            total=len(train_dl) - skip_n,
            initial=0,
        )
        for batch in pbar:
            batches_in_epoch += 1
            ids, labels  = _unpack_batch(batch, device)
            n_tokens = int((labels[:, 1:] != -100).sum().item())
            if n_tokens == 0:
                writer.add_scalar("data/skipped_zero_label_batches", 1, global_step)
                continue

            # Update n_iter every step for smooth curriculum (use pre-step global_step)
            cur_n = curriculum_n_iter(global_step, total_steps, args.n_iter) \
                    if args.curriculum else args.n_iter
            model.cfg.n_iter = cur_n

            out = model(input_ids=ids, labels=labels)
            # Scale loss so accumulated gradient == single-step gradient
            (out.loss / args.grad_accum).backward()

            loss_val = out.loss.item()
            metrics  = model.last_loss_metrics()
            accum_micro  += 1
            _acc_loss    += loss_val
            _acc_lm_loss += metrics.get("lm_loss", loss_val)
            _acc_tokens  += n_tokens

            # Mid-accumulation: update postfix but don't step
            if accum_micro % args.grad_accum != 0:
                pbar.set_postfix(
                    n=cur_n,
                    loss=f"{loss_val:.4f}",
                    acc=f"{accum_micro % args.grad_accum}/{args.grad_accum}",
                )
                continue

            # ── Optimizer step ────────────────────────────────────────────
            global_step += 1
            # Save accumulated stats before _do_optimizer_step resets them
            step_loss    = _acc_loss    / args.grad_accum
            step_lm_loss = _acc_lm_loss / args.grad_accum
            step_tokens  = _acc_tokens
            grad_norm    = _do_optimizer_step(cur_n, epoch)

            epoch_loss  += step_loss
            epoch_steps += 1
            log_loss_tokens    += step_loss    * step_tokens
            log_lm_loss_tokens += step_lm_loss * step_tokens
            log_tokens         += step_tokens
            log_cnt            += 1

            pbar.set_postfix(
                n=cur_n,
                loss=f"{step_loss:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            # Per-step loss (always logged for smooth curve)
            writer.add_scalar("train/loss_step", step_loss, global_step)
            writer.add_scalar("train/lm_loss_step", step_lm_loss, global_step)
            writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
            writer.add_scalar("train/tokens", step_tokens, global_step)
            for key, value in model.last_connect_metrics().items():
                writer.add_scalar(f"train/connect/{key}_step", value, global_step)
            for name in ("aux_loss", "consistency_loss"):
                if name in metrics:
                    writer.add_scalar(f"train/{name}_step", metrics[name], global_step)

            if global_step % args.log_steps == 0:
                denom = max(log_tokens, 1)
                avg = log_loss_tokens / denom
                lm_avg = log_lm_loss_tokens / denom
                elapsed = max(time.perf_counter() - log_start, 1e-9)
                writer.add_scalar("train/loss",   avg,                        global_step)
                writer.add_scalar("train/lm_loss", lm_avg,                    global_step)
                writer.add_scalar("train/ppl",    math.exp(min(lm_avg, 20)),  global_step)
                writer.add_scalar("train/lr",     scheduler.get_last_lr()[0], global_step)
                writer.add_scalar("train/n_iter", cur_n,                      global_step)
                writer.add_scalar("runtime/tokens_per_sec", log_tokens / elapsed, global_step)
                writer.add_scalar("runtime/steps_per_sec", log_cnt / elapsed, global_step)
                if torch.cuda.is_available():
                    writer.add_scalar("runtime/gpu_mem_allocated_gb",
                                      torch.cuda.max_memory_allocated() / 1e9,
                                      global_step)
                    torch.cuda.reset_peak_memory_stats()
                log_loss_tokens = 0.0
                log_lm_loss_tokens = 0.0
                log_tokens = 0
                log_cnt = 0
                log_start = time.perf_counter()

            # Save BEFORE eval — ensures checkpoint is written even if eval OOMs.
            if args.save_every > 0 and global_step % args.save_every == 0:
                resume_ckpt = os.path.join(args.output_dir, "resume.pt")
                tmp_ckpt    = resume_ckpt + ".tmp"
                torch.save(
                    {
                        "connect":          model.connect.state_dict(),
                        "optimizer":        optimizer.state_dict(),
                        "scheduler":        scheduler.state_dict(),
                        "global_step":      global_step,
                        "epoch":            epoch,
                        "best_ppl":         best_ppl,
                        "batches_in_epoch": batches_in_epoch,
                        "cfg":              {**asdict(cfg), "n_iter": args.n_iter},
                        "args":             vars(args),
                    },
                    tmp_ckpt,
                )
                os.replace(tmp_ckpt, resume_ckpt)  # atomic overwrite
                print(f"  [ckpt] Resume checkpoint → {resume_ckpt}  (step {global_step}, batch {batches_in_epoch})")

            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                try:
                    torch.cuda.empty_cache()
                    eval_loss, ppl = eval_suite(model, eval_dl, device, args,
                                                baseline_ppl, writer, global_step)
                    sign = "+" if ppl - baseline_ppl >= 0 else ""
                    print(f"  [Eval step {global_step}] loss={eval_loss:.4f}  PPL={ppl:.2f}  "
                          f"({sign}{ppl-baseline_ppl:.2f} vs baseline={baseline_ppl:.2f})")

                    if ppl < best_ppl:
                        best_ppl = ppl
                        ckpt = os.path.join(args.output_dir, "best_connect.pt")
                        model.save_connect(ckpt)
                        save_checkpoint_meta(ckpt, cfg, args, epoch=epoch,
                                             global_step=global_step, eval_ppl=ppl)
                        print(f"  Best saved: {ckpt}")
                except torch.cuda.OutOfMemoryError as e:
                    print(f"  [Eval step {global_step}] SKIPPED — OOM: {e}")
                    torch.cuda.empty_cache()
                finally:
                    model.train()
                    model.connect.train()

            if args.max_steps is not None and global_step >= args.max_steps:
                break

        # Flush any remaining accumulated gradients at epoch end
        if accum_micro % args.grad_accum != 0:
            global_step += 1
            _do_optimizer_step(cur_n, epoch)
            epoch_steps += 1

        pbar.close()

        avg_epoch = epoch_loss / max(epoch_steps, 1)
        writer.add_scalar("epoch/train_loss", avg_epoch, epoch)
        writer.add_scalar("epoch/train_ppl",  math.exp(min(avg_epoch, 20)), epoch)
        print(f"\n  Epoch {epoch}/{max_epochs} -- "
              f"avg loss: {avg_epoch:.4f}  "
              f"(train PPL~{math.exp(min(avg_epoch,20)):.2f})")

        # -- Eval on eval split --------------------------------------------
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            torch.cuda.empty_cache()
            eval_loss, ppl = eval_suite(model, eval_dl, device, args,
                                        baseline_ppl, writer, global_step)
            sign = "+" if ppl - baseline_ppl >= 0 else ""
            print(f"  [Eval]  loss={eval_loss:.4f}  PPL={ppl:.2f}  "
                  f"({sign}{ppl-baseline_ppl:.2f} vs baseline={baseline_ppl:.2f})")

            if ppl < best_ppl:
                best_ppl = ppl
                ckpt = os.path.join(args.output_dir, "best_connect.pt")
                model.save_connect(ckpt)
                save_checkpoint_meta(ckpt, cfg, args, epoch=epoch,
                                     global_step=global_step, eval_ppl=ppl)
                print(f"  Best saved: {ckpt}")

        model.train()
        model.connect.train()

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    # -- Final validation on held-out test split ---------------------------
    print("\n" + "=" * 60)
    model._eval_max_batches = args.eval_max_batches
    val_loss, val_ppl = eval_ppl(model, test_dl, device, n_iter=args.n_iter,
                                 desc="Final Validate")
    model._eval_max_batches = None
    sign = "+" if val_ppl - baseline_ppl >= 0 else ""
    print(f"Final Validate  loss={val_loss:.4f}  PPL={val_ppl:.2f}  "
          f"({sign}{val_ppl-baseline_ppl:.2f} vs baseline={baseline_ppl:.2f})")
    writer.add_scalar("validate/loss", val_loss, global_step)
    writer.add_scalar("validate/ppl",  val_ppl,  global_step)
    writer.add_scalar("validate/delta_vs_baseline", val_ppl - baseline_ppl, global_step)

    final = os.path.join(args.output_dir, "final_connect.pt")
    model.save_connect(final)
    save_checkpoint_meta(final, cfg, args, epoch=epoch,
                         global_step=global_step, validate_ppl=val_ppl)
    writer.close()

    print(f"\nDone. Best eval PPL: {best_ppl:.2f}  |  "
          f"Val loss: {val_loss:.4f}  PPL: {val_ppl:.2f}  |  Baseline: {baseline_ppl:.2f}")


if __name__ == "__main__":
    main()
