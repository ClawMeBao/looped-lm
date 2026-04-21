# LoopedLM — Copilot Instructions

## Environment

```bash
# Activate virtualenv before running anything
source /home/tieubaoca/AI/learn-ai/deep-learning/.venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

All scripts must be run from the **repo root** — they insert `../..` into `sys.path` at startup.

---

## Commands

```bash
# Smoke test (forward + backward + grad check, no GPU required for small seq_len)
python phase0/scripts/test_forward.py
python phase0/scripts/test_forward.py --connect_type mlp --seq_len 32 --batch_size 1

# Train connect layer
python phase0/scripts/train.py --max_steps 300 --curriculum
python phase0/scripts/train.py --connect_type gated --max_steps 1000 --n_iter 3

# Evaluate perplexity (pre-train state)
python phase0/scripts/eval.py
# After training
python phase0/scripts/eval.py --checkpoint phase0/checkpoints/best_connect.pt

# Inference / generation
python phase0/scripts/inference.py \
    --checkpoint phase0/checkpoints/best_connect.pt \
    --compare --prompt "The history of AI"

# True baseline only (fast)
python phase0/scripts/eval.py --true_baseline_only
```

Checkpoints are saved to `phase0/checkpoints/` (default) as `.pt` files containing only the connect layer state dict.

---

## Architecture

The project turns a frozen pretrained LLM (`Qwen/Qwen3-1.7B`, 28 layers) into a **Looped Language Model** by splitting it into three frozen segments and inserting a small trainable connect layer between loop iterations:

```
embed → PREFIX (L0–7, frozen)
              ↓ h_loop_in
         ╔══════════════════╗
         ║  LOOP (L8–19)    ║ ← frozen, repeated n_iter times
         ╚══════╦═══════════╝
                ║ h_out
                ▼
         [ ConnectLayer ]    ← ONLY trainable component (~6M params)
                │
         [ExitGate?]         ← Phase 1+
                ↓
         SUFFIX (L20–27, frozen) → norm → lm_head
```

**`n_iter=0` is the true baseline** — runs all 28 layers unchanged (identical to HF model). Any `n_iter≥1` routes through the loop block repeatedly.

### Code layers

| Path | Responsibility |
|---|---|
| `common/backbone.py` | `load_backbone()` — loads any Qwen3/Qwen2 model, freezes all params, returns `BackboneComponents` |
| `common/connect_layer.py` | `MLPConnectLayer` (Phase 0 option B) and `GatedResidualConnectLayer` (Phase 0+ default, option C) |
| `common/data_utils.py` | `load_text_dataset()` — wraps WikiText-2; falls back to dummy text if `datasets` unavailable |
| `phase0/src/model.py` | `Phase0Model` + `Phase0Config` — fixed `n_iter`, no gate |
| `phase1/src/exit_gate.py` | `ExitGate` — soft (training) / hard (inference) gate with ponder loss |
| `phase1/src/model.py` | `Phase1Model` skeleton — **not yet implemented** |
| `phase2/src/model.py` | Raises `NotImplementedError` — depends on Phase 1 |

### Phase status

| Phase | Status | Key feature |
|---|---|---|
| 0 | ✅ Done | Fixed `n_iter`, trainable `GatedResidualConnectLayer` |
| 1 | 🚧 In progress | Dynamic `ExitGate` (soft→hard), ponder + consistency loss |
| 2 | ⬜ Planned | Iteration-aware connect, linear-attn state carry, LoRA unfreeze |

---

## Key Conventions

### Backbone is always frozen
`load_backbone()` freezes all params before returning. `Phase0Model.train()` is overridden to keep all backbone segments in `eval()` mode — only `self.connect` follows the requested mode. Never call `model.backbone.train()`.

### Only save/load connect layer weights
`model.save_connect(path)` / `model.load_connect(path)` — checkpoints contain only `connect.state_dict()` (~6M params), not the full model.

### `model.trainable_parameters()` — not `model.parameters()`
Pass `model.trainable_parameters()` to the optimizer. `model.parameters()` includes frozen backbone params (1.7B).

### Backbone registration pattern
Backbone layers must be wrapped in `nn.ModuleList` when assigned as attributes so that `model.to(device)`, `model.state_dict()`, and `model.parameters()` work correctly. Assigning a bare list silently skips them.

### Transformers version compatibility
Layer outputs are handled as:
```python
hidden_states = out[0] if isinstance(out, tuple) else out
```
Transformers ≥5.x returns a `Tensor`; <5.x returns a tuple. All layer calls use `use_cache=False`.

### Rotary embedding API detection
`BackboneComponents.rotary_uses_position_ids` is detected at load time via `inspect.signature`. Use it to decide between `rotary_emb(h, position_ids)` vs `rotary_emb(h, seq_len=S)`.

### Truncated BPTT
In the training loop, gradient is detached for iterations `< n_iter - k_bptt` to avoid vanishing gradients across deep unrolls. Default `k_bptt=2`.

### GatedResidualConnectLayer signature
Takes two tensors: `connect(h_out, h_prev)`. `MLPConnectLayer` takes one: `connect(h_out)`. Always check with `isinstance(self.connect, GatedResidualConnectLayer)`.

### Phase 1 warm-start
`Phase1Config.phase0_checkpoint` — path to a Phase 0 `best_connect.pt`. `Phase1Model.from_pretrained()` loads it automatically into the connect layer.

### Loss recipe
- Phase 0: standard LM cross-entropy only
- Phase 1 (planned): `L = L_lm + β*L_ponder + λ*L_consistency`  
  Default: `β=0.01`, `λ=0.1`

### Memory budget
`n_iter=3`, `seq_len=256`, `batch_size=2`, bf16 → ~8GB VRAM. Reduce `seq_len` or `batch_size` for OOM.
