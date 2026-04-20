# LoopedLM — Phase 0 Prototype

Biến một pretrained LM thành **Looped Language Model** bằng cách thêm loop block + connect layer.

---

## Cấu trúc

```
looped-lm/
├── src/
│   ├── __init__.py
│   ├── connect_layer.py   ← MLPConnectLayer (B) + GatedResidualConnectLayer (C)
│   └── looped_lm.py       ← LoopedLM wrapper + LoopedLMConfig
├── scripts/
│   ├── test_forward.py    ← Smoke test: forward/backward + grad check
│   ├── eval_perplexity.py ← So sánh PPL n_iter=0,1,2,3,4
│   └── train_connect.py   ← Train connect layer (curriculum support)
├── checkpoints/           ← Saved .pt files
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Phase 0 — Workflow

### Bước 1: Smoke test (không cần GPU mạnh, ~5 phút)

Kiểm tra forward pass, gradient flow, param count:

```bash
python scripts/test_forward.py \
    --model Qwen/Qwen3-1.7B \
    --connect_type mlp \
    --n_iter 3 \
    --seq_len 32 \
    --batch_size 1
```

Output mong đợi:
```
✅ Forward OK  Loss: 3.xxxx
✅ Backward OK
✅ Connect layer: tất cả params có gradient
✅ Frozen backbone: không có gradient
```

**Kiểm tra với Gated connect:**
```bash
python scripts/test_forward.py --connect_type gated
```

---

### Bước 2: Baseline PPL (trước khi train)

Xác nhận PPL với n_iter=0,1,2,3,4 khi connect layer chưa được train:

```bash
python scripts/eval_perplexity.py \
    --model Qwen/Qwen3-1.7B \
    --max_iter 4 \
    --baseline_only    # chỉ chạy n_iter=0 để nhanh
```

**Quan sát kỳ vọng (chưa train):**
- `n_iter=0`: PPL ≈ baseline của model gốc
- `n_iter=1,2,3`: PPL có thể cao hơn đôi chút (connect layer random weights)
- Nếu PPL với loop **không crash** và không quá khác baseline → loop block selection đúng

---

### Bước 3: Quick training test (~30 phút, 1 GPU)

```bash
python scripts/train_connect.py \
    --model Qwen/Qwen3-1.7B \
    --connect_type mlp \
    --max_steps 500 \
    --eval_steps 100 \
    --batch_size 2 \
    --seq_len 256 \
    --lr 3e-4
```

**Với curriculum (recommended):**
```bash
python scripts/train_connect.py \
    --connect_type mlp \
    --max_steps 1000 \
    --curriculum \
    --n_iter 3
```

---

### Bước 4: Evaluate sau train

```bash
python scripts/eval_perplexity.py \
    --checkpoint checkpoints/best_connect.pt \
    --connect_type mlp \
    --max_iter 4
```

**Pass criteria cho Phase 0:**
- PPL với n_iter=best ≤ PPL baseline (n_iter=0)
- Training loss giảm ổn định (không diverge)
- Không có gradient vào frozen layers

---

## Key config params

| Param | Default | Ý nghĩa |
|-------|---------|---------|
| `loop_start` | 8 | Layer bắt đầu loop block |
| `loop_end` | 20 | Layer kết thúc loop block |
| `n_iter` | 3 | Số vòng lặp cố định |
| `connect_type` | `mlp` | `mlp` = Phương án B, `gated` = Phương án C |
| `k_bptt` | 2 | Truncated BPTT depth |
| `bottleneck_ratio` | 0.25 | Bottleneck size = d_model × ratio |

---

## Lưu ý kỹ thuật

**Về model tương thích:**
- `Qwen/Qwen3-1.7B`: ✅ Pure text, 28 layers, d_model=2048 — **recommended cho Phase 0**
- `Qwen/Qwen3-2B` (nếu tồn tại): ✅ Similar
- `Qwen/Qwen3.5-2B`: ⚠️ Multimodal (VLM), cần adapter riêng — Phase 1+

**Về KV cache:**
Phase 0 không dùng KV cache (`use_cache=False`). Loop block chạy full attention
mỗi iteration → đúng về mặt semantics, chậm hơn cached inference.
KV cache + loop sẽ được handle ở Phase 1.

**Về memory:**
- `n_iter=3`, `seq_len=256`, `batch_size=2`, bf16 → ~8GB VRAM
- Giảm `seq_len` hoặc `batch_size` nếu OOM

---

## Roadmap

```
Phase 0 (current) : Connect layer, fixed n_iter, no gate
Phase 1           : Exit gate (soft/hard), ponder loss
Phase 2           : Iteration-aware connect + full training recipe
Phase 3           : Linear attention state carry, LoRA unfreeze
```
