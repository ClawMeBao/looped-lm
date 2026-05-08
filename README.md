# LoopedLM

> Biến pretrained LLM thành **Looped Language Model** — thêm iterative reasoning bằng cách inject loop block + connect layer vào frozen backbone.

Base model: `Qwen/Qwen3-1.7B` | Loop block: layers [8, 20) | Trainable: connect layer only

---

## Cấu trúc project

```
looped-lm/
│
├── common/                     ← Shared code cho tất cả phases
│   ├── backbone.py             ← load_backbone(): unpack + freeze pretrained model
│   ├── connect_layer.py        ← MLPConnectLayer (B) + GatedResidualConnectLayer (C)
│   └── data_utils.py           ← TokenizedTextDataset, load_text_dataset()
│
├── phase0/                     ← ✅ DONE — Fixed n_iter, no exit gate
│   ├── src/
│   │   └── model.py            ← Phase0Model, Phase0Config
│   ├── scripts/
│   │   ├── train.py            ← Train connect layer (curriculum)
│   │   ├── eval.py             ← Perplexity comparison n_iter=0..N
│   │   ├── inference.py        ← Text generation (baseline vs looped)
│   │   └── test_forward.py     ← Smoke test
│   └── checkpoints/            ← Saved .pt files
│
├── phase1/                     ← 🚧 IN PROGRESS — Dynamic exit gate
│   ├── src/
│   │   ├── exit_gate.py        ← ExitGate (soft/hard, ponder loss)
│   │   └── model.py            ← Phase1Model skeleton
│   ├── scripts/
│   │   └── train.py            ← 🚧 Skeleton
│   └── checkpoints/
│
├── phase2/                     ← ⬜ PLANNED — Full power
│   ├── src/
│   │   └── model.py            ← 🚧 Skeleton
│   └── checkpoints/
│
└── docs/
    ├── design.md               ← Architecture design + phương án choices
    └── PHASE0_REPORT.md        ← Phase 0 experiment results
```

---

## Setup

```bash
git clone https://github.com/ClawMeBao/looped-lm.git
cd looped-lm
pip install -r requirements.txt
```

---

## Phase 0 — ✅ Completed

**Goal:** Verify connect layer có thể học được cách remap distribution gap giữa loop block output và input.

**Result (from PHASE0_REPORT.md):**

| Connect type | Best PPL | Baseline PPL | n_iter stability |
|---|---|---|---|
| MLP | **59.0** (n_iter=1) | ~9,972 | ❌ Degrades 59→9860 |
| **GatedResidual** | **98.4** (n_iter=2) | ~10,249 | ✅ Flat ~98 for n_iter 1–4 |

→ **GatedResidual** là default cho Phase 1+ vì depth-invariant.

### Quick start

```bash
# 1. Smoke test
python phase0/scripts/test_forward.py

# 2. Train (300 steps ~36s trên GPU)
python phase0/scripts/train.py --max_steps 300 --curriculum

# 3. Eval
python phase0/scripts/eval.py --checkpoint phase0/checkpoints/best_connect.pt

# 4. Generate
python phase0/scripts/inference.py \
    --checkpoint phase0/checkpoints/best_connect.pt \
    --compare --prompt "The history of AI"
```

---

## Phase 1 — 🚧 In Progress

**Goal:** Thêm exit gate để quyết định dynamic số iterations.

**Design (từ docs/design.md):**
- Signals: `h_end` + `delta norm` + `iter_embedding`
- Training: soft gate + ponder loss + consistency loss
- Inference: hard threshold → true computational skip
- Warm-start connect layer từ Phase 0 checkpoint

**Next steps:**
1. Implement `Phase1Model.forward()` với soft gate loop
2. Training recipe: `L = L_lm + β*L_ponder + λ*L_consistency`
3. Validate gate không collapse (luôn exit sớm hoặc không bao giờ exit)

---

## Phase 2 — ⬜ Planned

**Goal:** Full power — iteration-aware connect, linear attention state carry, LoRA unfreeze.

Depends on Phase 1 completion.

---

## Architecture overview

```
input → embed → PREFIX (L0–7, frozen)
                    ↓
             ╔══════════════════╗
             ║  LOOP (L8–19)    ║ ← frozen, n_iter times
             ║  3 groups of     ║
             ║  [lin,lin,lin,   ║
             ║   full_attn]     ║
             ╚══════╦═══════════╝
                    ║ h_out
                    ▼
             ┌──────────────┐
             │ ConnectLayer │ ← trainable (~6M params)
             │ GatedResidual│   gate * transform(h_out)
             │              │   + (1-gate) * h_prev
             └──────┬───────┘
                    │
             [ExitGate?]  ← Phase 1+
                    │
             SUFFIX (L20–27, frozen) → norm → lm_head
```

---

## Key numbers

| | Value |
|---|---|
| Base model | Qwen3-1.7B (1.7B params) |
| d_model | 2048 |
| Loop block depth | 12 layers (43% of model) |
| Connect layer params | ~6M (0.3%) |
| Exit gate params | ~1.5M (Phase 1) |
| Training dataset | WikiText-2 |
| Training time (300 steps) | ~36s on GPU |
