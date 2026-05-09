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
│   ├── connect_layer.py        ← Residual + MLP/Gated/IterAware ablations
│   └── data_utils.py           ← Text/chat/GLM dataset utilities
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

## Phase 0 — ✅ Structurally usable

**Goal:** Verify connect layer có thể học được cách remap distribution gap giữa loop block output và input.

**Current result (from PHASE0_REPORT.md):**

| Item | Current status |
|---|---|
| Default connect | `residual` |
| Loop topology | `prefix -> loop -> (connect -> loop)* -> suffix` |
| GLM 3-epoch final PPL | 3.3054 on current validation split |
| Connect update ratio | ~2.2-2.4% hidden norm |
| Repetition-collapse bug | Fixed in current residual path |
| Final proof of model quality | Pending held-out generation/eval sweep |

→ Phase 0 hạ tầng đã ổn để đánh giá nghiêm túc hơn và thử cải tiến kiến trúc. PPL thấp không được xem là bằng chứng đủ nếu chưa kiểm tra generation/repetition.

### Quick start

```bash
# 1. Smoke test
python phase0/scripts/test_forward.py

# 2. Train GLM local subset
python phase0/scripts/train.py \
  --connect_type residual \
  --n_iter 4 \
  --curriculum \
  --epochs 3 \
  --dataset_type glm \
  --local_dataset_dir data/glm_dataset/train_10k \
  --no_think \
  --seq_len 512 \
  --batch_size 1 \
  --lr 5e-5

# 3. Eval
python phase0/scripts/eval.py \
  --checkpoint phase0/checkpoints/residual_seq512_3epoch/final_connect.pt \
  --connect_type residual \
  --dataset_type glm \
  --local_dataset_dir data/glm_dataset/train_10k \
  --no_think \
  --seq_len 512 \
  --max_iter 4 \
  --skip_true_baseline

# 4. Generate
python phase0/scripts/inference.py \
  --checkpoint phase0/checkpoints/residual_seq512_3epoch/final_connect.pt \
  --connect_type residual \
  --compare \
  --greedy \
  --no_think \
  --prompt "The history of AI"
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
             │ ConnectLayer │ ← trainable (~2.1M params)
             │ Residual     │   h_prev + scale * delta(h_out)
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
| Connect layer params | ~2.1M for residual |
| Exit gate params | ~1.5M (Phase 1) |
| Training dataset | GLM local subset (`data/glm_dataset/train_10k`) |
| Current status | Structurally usable; held-out generation/eval pending |
