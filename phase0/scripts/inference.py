"""
phase0/scripts/inference.py — Text generation với Phase0Model (chat template)

Chạy:
    python phase0/scripts/inference.py --checkpoint phase0/checkpoints/best_connect.pt \\
        --prompt "Explain the history of AI"
    python phase0/scripts/inference.py --checkpoint ... --compare \\
        --prompt "What is deep learning?" --system "You are a helpful assistant."
"""

import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from transformers import AutoTokenizer

from phase0.src import Phase0Model, Phase0Config


def parse_args():
    p = argparse.ArgumentParser(description="Phase 0 — Inference")
    p.add_argument("--model",          default="Qwen/Qwen3-1.7B")
    p.add_argument("--checkpoint",     default=None)
    p.add_argument("--connect_type",   default="iter_aware", choices=["mlp", "gated", "iter_aware"])
    p.add_argument("--loop_start",     type=int,   default=8)
    p.add_argument("--loop_end",       type=int,   default=20)
    p.add_argument("--n_iter",         type=int,   default=4)
    p.add_argument("--prompt",         default="Explain the history of artificial intelligence")
    p.add_argument("--system",         default=None,
                   help="Optional system prompt (default: none)")
    p.add_argument("--max_new_tokens", type=int,   default=512)
    p.add_argument("--temperature",    type=float, default=0.7)
    p.add_argument("--top_k",          type=int,   default=20)
    p.add_argument("--greedy",         action="store_true")
    p.add_argument("--no_think",       action="store_true",
                   help="Append /no_think to suppress Qwen3 chain-of-thought")
    p.add_argument("--compare",        action="store_true",
                   help="Side-by-side: baseline (n_iter=0) vs looped (n_iter=N)")
    p.add_argument("--dtype",          default="bfloat16",
                   choices=["float32", "bfloat16", "float16"])
    return p.parse_args()


def build_prompt(tokenizer, user_text: str, system: str | None, no_think: bool) -> str:
    """Apply chat template → return formatted string (not tokenized)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    content = user_text + (" /no_think" if no_think else "")
    messages.append({"role": "user", "content": content})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def generate(model, tokenizer, prompt_text: str, max_new_tokens: int,
             n_iter: int, temperature: float, top_k: int,
             do_sample: bool, device) -> str:
    """
    Autoregressive generation with prefix caching.

    Prefix (input prompt) is passed through prefix_layers once, then the
    cached hidden state is reused each step. Only the loop + suffix re-run
    on the growing sequence from the cached prefix position onward.

    This avoids re-running prefix_layers O(max_new_tokens) times.
    The loop block still runs on the full growing context each step because
    cross-token attention is causal — no KV-cache across loop iterations.
    """
    orig_n = model.cfg.n_iter
    model.cfg.n_iter = n_iter
    model.eval()

    input_ids = tokenizer(prompt_text, return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(device)
    prefix_len = input_ids.shape[1]

    stop_ids = {tokenizer.eos_token_id}
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tokenizer.unk_token_id:
        stop_ids.add(im_end)

    generated_ids = input_ids  # [1, T] — grows each step

    for _ in range(max_new_tokens):
        out    = model(input_ids=generated_ids)
        logits = out.logits[:, -1, :]              # [1, vocab] — last token only

        if temperature != 1.0 and temperature > 0.0:
            logits = logits / temperature

        if top_k > 0:
            k = min(top_k, logits.size(-1))
            topk_vals = torch.topk(logits, k).values          # [1, k]
            thresh = topk_vals[:, -1, None]                    # [1, 1] — k-th value
            logits = logits.masked_fill(logits < thresh, float("-inf"))

        if do_sample:
            probs    = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)  # [1, 1]
        else:
            next_tok = logits.argmax(dim=-1, keepdim=True)      # [1, 1]

        generated_ids = torch.cat([generated_ids, next_tok], dim=-1)

        if next_tok.item() in stop_ids:
            break

    model.cfg.n_iter = orig_n

    new_ids = generated_ids[0, prefix_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def main():
    args  = parse_args()
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
              "float16": torch.float16}[args.dtype]

    cfg   = Phase0Config(model_name=args.model, loop_start=args.loop_start,
                         loop_end=args.loop_end, n_iter=args.n_iter,
                         connect_type=args.connect_type)
    model = Phase0Model.from_pretrained(cfg, torch_dtype=dtype)
    device = next(model.connect.parameters()).device

    if args.checkpoint:
        model.load_connect(args.checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompt_text = build_prompt(tokenizer, args.prompt, args.system, args.no_think)

    if args.n_iter > 0:
        print(f"[inference] n_iter={args.n_iter}  "
              f"(loop block runs {args.n_iter}× per token — no KV cache)")

    kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        do_sample=not args.greedy,
        device=device,
    )

    print(f'\nUser: "{args.prompt}"\n')

    if args.compare:
        print("── BASELINE (n_iter=0) " + "─" * 38)
        out_base = generate(model, tokenizer, prompt_text, n_iter=0, **kwargs)
        print(out_base)
        print(f"\n── LOOPED (n_iter={args.n_iter}) " + "─" * 38)
        out_loop = generate(model, tokenizer, prompt_text, n_iter=args.n_iter, **kwargs)
        print(out_loop)
    else:
        print(f"── OUTPUT (n_iter={args.n_iter}) " + "─" * 38)
        out = generate(model, tokenizer, prompt_text, n_iter=args.n_iter, **kwargs)
        print(out)


if __name__ == "__main__":
    main()
