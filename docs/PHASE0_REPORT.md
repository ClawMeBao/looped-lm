# LoopedLM — Phase 0 Report

> **Objective:** Validate that a trainable connect layer inserted into a frozen Qwen3-1.7B backbone
> can reduce perplexity degradation when the loop block is executed iteratively, while confirming
> the `n_iter=0` path reproduces the true HF model baseline exactly.

---

## 1. Environment

| Item | Value |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.9.1+cu128 |
| Transformers | 5.3.0 |
| Accelerate | 1.12.0 |
| Datasets | 4.8.4 |
| CUDA | 12.8 |
| Base model | `Qwen/Qwen3-1.7B` (bfloat16) |
| Virtualenv | `/home/tieubaoca/AI/learn-ai/deep-learning/.venv` |

---

## 2. Architecture

| Component | Detail |
|---|---|
| Total params | 2,033,841,153 |
| Trainable params | **2,101,249** (0.103 %) — GatedResidual connect layer only |
| Frozen params | 2,031,739,904 (backbone) |
| Connect type | `GatedResidualConnectLayer` (default) |
| Loop block | Transformer layers **[8, 20)** (12 layers) |
| Prefix block | Layers [0, 8) — run once before loop |
| Suffix block | Layers [20, 28) — run once after loop |
| `n_iter=0` | Runs all 28 layers unchanged — true HF model baseline |
| `n_iter≥1` | prefix → (loop → connect) × n_iter → suffix |
| KV cache | Disabled (`use_cache=False`) — full recompute per iteration |
| Truncated BPTT | `k_bptt=2` — only last 2 iterations backpropagate |

---

## 3. Bugs Found and Fixed

Six bugs were identified and fixed across development iterations:

### Bug 1 — Decoder layer return-type mismatch
- **Root cause:** `transformers ≥ 5.x` changed `Qwen3DecoderLayer.forward` to return a plain
  `torch.Tensor` instead of a tuple. The old code did `layer_out[0]` unconditionally, dropping
  batch dimension after the prefix block.
- **Fix:** `hidden_states = layer_out[0] if isinstance(layer_out, tuple) else layer_out`

### Bug 2 — Rotary embedding API mismatch
- **Root cause:** New HF API is `rotary_emb(x, position_ids)`. Old code called
  `rotary_emb(x, seq_len=N)`, causing a shape crash.
- **Fix:** Detect API at load time via `inspect.signature`; branch in `forward` accordingly.

### Bug 3 — `torch.no_grad()` blocking the gradient path
- **Root cause:** Loop and suffix blocks were wrapped in `torch.no_grad()`, cutting autograd
  graph to connect layer. `.backward()` raised *"element 0 of tensors does not require grad"*.
- **Fix:** Remove `torch.no_grad()` from loop and suffix. Frozen params have
  `requires_grad=False` so they never accumulate gradients regardless.

### Bug 4 — Causal mask dtype mismatch
- **Root cause:** `_make_causal_mask()` produced float32 mask; SDPA requires dtype to match
  query dtype (bfloat16).
- **Fix:** Pass `dtype=hidden_states.dtype` to `torch.full(...)`.

### Bug 5 — `n_iter=0` skipped loop block (critical correctness bug)
- **Root cause:** Previous code branched early and ran only prefix + suffix for `n_iter=0`,
  producing a mutilated 16-layer forward pass with PPL ~10,000 instead of the true ~30.
  This made all "improvement" comparisons in earlier report runs meaningless (comparing
  looped output against a broken baseline).
- **Fix:** `n_iter=0` now runs all 28 layers (prefix + loop + suffix) with `torch.no_grad()`,
  exactly matching the HF model forward pass.

### Bug 6 — Backbone not registered as `nn.Module`
- **Root cause:** Backbone layer lists were stored as plain Python lists, so `model.to(device)`,
  `model.state_dict()`, and `model.parameters()` silently skipped them.
- **Fix:** Wrap all backbone segments in `nn.ModuleList` before assigning as attributes.

---

## 4. Step 1 — Smoke Test

**Command:**
```bash
python phase0/scripts/test_forward.py --seq_len 32 --batch_size 1
```

**Result:** ✅ **PASSED**

```
✅ Forward OK   loss=15.3750  logits=[1, 32, 151936]
✅ Backward OK
✅ Connect grads: ALL present
✅ Frozen backbone: no gradient (correct)

Gate stats at init:
  gate_mean: 0.5000  gate_std: 0.0000
  gate_min:  0.5000  gate_max: 0.5000
  (Expected gate_mean ≈ 0.5)
```

- Forward pass, backward pass, and gradient isolation all confirmed.
- Gate initialises at 0.5 (neutral mixing) as designed.

---

## 5. Step 2 — Pre-Training Perplexity (Random Connect Weights)

Dataset: WikiText-2 validation, 50 samples × 256 tokens

| n_iter | PPL | Δ vs baseline |
|---|---|---|
| 0 (baseline — all 28 layers) | **30.23** | — |
| 1 | 1,120,641 | +1,120,611 |
| 2 | 1,120,641 | +1,120,611 |
| 3 | 1,120,641 | +1,120,611 |
| 4 | 1,120,641 | +1,120,611 |

**Note:** All n_iter≥1 give identical PPL because the gate is initialised at 0.5 and the
transform MLP is initialised to zero output — the layer produces a fixed 50/50 blend with
no learned semantics. The ~37,000× PPL explosion confirms the connect layer is actively
injected into the computation path and requires training.

> **Corrected vs previous report:** Prior runs showed baseline PPL ~10,000 due to Bug 5
> (n_iter=0 skipped the loop block, running only 16 layers). The true Qwen3-1.7B baseline
> on WikiText-2 is **~30–31 PPL**.

---

## 6. Step 3 — Training Connect Layer (GatedResidual, 300 steps)

**Command:**
```bash
python phase0/scripts/train.py \
    --connect_type gated \
    --max_steps 300 \
    --curriculum \
    --n_iter 3 \
    --batch_size 2 \
    --seq_len 256 \
    --lr 3e-4 \
    --output_dir phase0/checkpoints
```

**Curriculum schedule:** n_iter=1 for first 25% of steps, then increases linearly to n_iter=3.

**True baseline PPL (n_iter=0) before training:** 30.72

**Training loss curve (sampled):**

| Step | n_iter | Loss | PPL |
|---|---|---|---|
| 10 | 1 | 11.36 | 85,498 |
| 30 | 1 | 8.06 | 3,163 |
| 60 | 1 | 7.34 | 1,547 |
| 100 | 1 | 6.08 | 439 |
| 150 | 1 | 5.21 | 183 |
| 190 | 2 | 5.20 | 181 |
| 200 | 2 | 4.66 | 106 |
| 260 | 2 | 4.38 | 80 |
| 300 | 2 | 4.53 | 93 |

**Eval PPL on validation set (n_iter=3):**

| Eval @ step | PPL | Δ vs baseline (30.72) |
|---|---|---|
| 200 | **132.87** | +102.15 |

Best checkpoint saved: `phase0/checkpoints/best_connect.pt` (step 200)

**Training time:** ~35 seconds total (GPU CUDA 12.8)

---

## 7. Step 4 — Post-Training Perplexity

**Command:**
```bash
python phase0/scripts/eval.py \
    --checkpoint phase0/checkpoints/best_connect.pt \
    --max_iter 4 \
    --max_samples 100
```

Dataset: WikiText-2 validation, 100 samples × 256 tokens

| | PPL |
|---|---|
| True baseline (HF Qwen3-1.7B direct) | **31.16** |
| LoopedLM n_iter=0 (Bug5 fix verify) | **31.20** (Δ=+0.04 ✅) |

| n_iter | Pre-train PPL | Post-train PPL | Δ vs true baseline |
|---|---|---|---|
| 1 | 1,120,641 | **125.86** | +94.70 |
| 2 | 1,120,641 | **125.38** | +94.23 |
| 3 | 1,120,641 | **125.38** | +94.23 |
| 4 | 1,120,641 | **126.01** | +94.86 |

**Key result:** n_iter=0 matches true HF baseline to within 0.04 PPL, confirming Bug 5 fix
is correct. The connect layer reduced PPL from ~1,120,641 → ~125 (9,000× improvement over
untrained), but is still ~4× worse than the true baseline (31 PPL). **300 steps is
insufficient** to match baseline quality with a correct 31 PPL reference.

---

## 8. Step 5 — Inference Comparison

**Command:**
```bash
python phase0/scripts/inference.py \
    --checkpoint phase0/checkpoints/best_connect.pt \
    --compare --prompt "The history of artificial intelligence"
```

**Baseline (n_iter=0):**
> *"The history of artificial intelligence (AI) is a fascinating journey that has evolved from
> the early days of theoretical exploration to the sophisticated technologies we see today.
> One of the pivotal moments in this history was the creation of the first artificial neural
> network (ANN)..."*

**Looped (n_iter=3):**
> *"The history of artificial intelligence in the business of business business is of the
> business of the global world to the world of the business region..."*

**Observation:** At 300 training steps, the connect layer has not yet learned to reconstruct
a coherent hidden-state mapping. The baseline generates fluent, on-topic text while the looped
model produces repetitive, incoherent output.

---

## 9. Experiment C — GatedResidual 2000-Step Run

### Setup
| Parameter | Value |
|---|---|
| `--connect_type` | gated |
| `--max_steps` | 2000 |
| `--curriculum` | enabled |
| `--max_samples` | 1000 |
| `--lr` | 3e-4 |
| Baseline PPL (eval set) | 30.99 |
| Training time | ~224s (~3.7 min) |

### Eval PPL during training (n_iter=3 eval)
| Step | Eval PPL | Note |
|---|---|---|
| 400 | 7,295 | n_iter=1 training, n_iter=3 eval — depth mismatch |
| 800 | 9,604 | still n_iter=1 training |
| 1200 | 47,334 | curriculum transition to n_iter=2 destabilised eval |
| 1600 | 46.12 | **breakthrough** — depth-invariance kicked in |
| 2000 | **45.43** | best checkpoint |

**Key observation:** Eval PPL remained in the thousands until step ~1600, when the curriculum
reached n_iter=2 and gated residual's depth-invariance allowed the trained weights to
generalise to n_iter=3 evaluation. PPL collapsed ~1000× in one eval window.

### Post-train PPL per n_iter (best_connect.pt)
| n_iter | PPL | Δ vs baseline |
|---|---|---|
| 0 (pass-through) | 30.81 | +0.04 ✅ |
| 1 | 46.76 | +15.77 |
| 2 | 46.64 | +15.65 |
| 3 | **46.61** | +15.62 |
| 4 | 46.73 | +15.74 |

**Depth-invariant plateau confirmed:** n_iter=1..4 all land within 0.15 PPL of each other.
GatedResidual learned a near-identity transform that is composable across arbitrary loop depth.

### Gap to baseline
45.43 vs 30.99 → **1.47× worse** than true baseline. Significant improvement over 300-step
result (132.87 PPL), but the connect layer has not yet matched frozen model quality.

### Inference quality (2000 steps)
**Baseline (n_iter=0):**
> *"The history of artificial intelligence (AI) is a rich tapestry woven with the threads of
> academic curiosity, technological innovation, and societal impact..."*

**Looped (n_iter=3):**
> *"The history of artificial intelligence is the largest most important and the most most
> least one to take the position as the first of the group and the twoth of the group ."*

**Observation:** Less repetitive than 300-step output (no pure word-loop), but still semantically
incoherent. The connect layer has learned basic token-level statistics but not higher-level
semantic structure.

---

## 10. Phase 0 Pass/Fail Assessment

| Criterion | Result |
|---|---|
| Smoke test passes (forward + backward + grad check) | ✅ PASS |
| `n_iter=0` matches true HF baseline (Bug 5 fix verified) | ✅ PASS (Δ=+0.04) |
| Pre-train PPL >> baseline (random weights corrupt hidden states) | ✅ PASS (~37,000×) |
| Training loss decreases per curriculum stage | ✅ PASS |
| Connect layer reduces PPL vs untrained (9,000× improvement) | ✅ PASS |
| Gradients flow only to connect layer (backbone frozen) | ✅ PASS |
| Checkpoint saved and loadable | ✅ PASS |
| Post-train PPL beats true baseline (31 PPL) after 300 steps | ❌ FAIL (125 vs 31) |
| Post-train PPL after 2000 steps (best checkpoint) | ❌ FAIL (45.43 vs 31) |
| Depth-invariant plateau confirmed (n_iter=1..4 within 0.15 PPL) | ✅ PASS |

**Overall Phase 0 verdict: ⚠️ PARTIAL — infrastructure validated, depth-invariance confirmed, baseline not yet matched**

---

## 10. Discussion

### What changed vs previous report
The most significant change is that **Bug 5 correction completely invalidates the prior
perplexity comparisons**. The old baseline of ~10,000 PPL was an artefact of running only
16 out of 28 layers, not the full model. The true Qwen3-1.7B baseline is **~31 PPL** on
WikiText-2, which is a far more competitive target.

### What the 300-step run actually proves
- The loop pipeline is wired correctly end-to-end (forward, backward, checkpoint I/O).
- The GatedResidual connect layer learns signal from essentially random weights (PPL 1.1M → 125).
- The `n_iter=0` branch is a lossless pass-through of the original model (Δ=0.04 PPL).

### What the 2000-step run proves
- Depth-invariance is real: gated residual generalises across n_iter=1..4 (plateau ±0.15 PPL).
- The breakthrough in PPL is tied directly to curriculum depth — the collapse happened when
  training reached n_iter=2, confirming the curriculum is the right training signal.
- PPL improves substantially with more steps (125 → 45.43), but the gap to baseline (~15 PPL)
  requires longer training.

### Why the looped model hasn't beaten baseline yet
1. **Training budget:** 300 steps on WikiText-2 (500 samples) is a proof-of-concept run.
   With a true baseline of 31 PPL, the connect layer must do substantially more work.
2. **Curriculum depth:** Training only reaches n_iter=2; the model hasn't been exposed to
   the full 3-iteration regime for long enough.
3. **Data coverage:** 500 training samples × 256 tokens is a very small dataset.

### Next steps for Phase 0 completion
1. **Longer training:** Run 5,000+ steps. Trend suggests PPL still decreasing at step 2000.
2. **Deeper curriculum:** Extend to n_iter=4 gradually.
3. **Eval metric:** Add downstream task eval (BoolQ, HellaSwag) once PPL gap closes.
4. **Hyperparameter search:** Try `lr=1e-3` with warmup; current 3e-4 may be too conservative.
5. **Larger dataset:** `--max_samples 2000+` to reduce overfitting risk.

### Recommendation for Phase 1
Do not start Phase 1 (exit gate) until the GatedResidual connect layer can match or beat
the ~31 PPL baseline. Phase 1 depends on a stable connect layer as its warm-start.

---

*Report updated after Bug 5 & Bug 6 fixes on Qwen/Qwen3-1.7B.*
*Experiment C (2000-step GatedResidual) added.*
*Previous results with ~10,000 PPL baseline are superseded by this report.*

---

## Experiment D — 5000-step GatedResidual (n_iter=4, max_samples=1500)

**Command:**
```bash
python phase0/scripts/train.py \
    --connect_type gated \
    --max_steps 5000 \
    --curriculum --max_samples 1500 --n_iter 4
```

### Training — Eval PPL curve (n_iter=4)

| Step | PPL | Δ vs baseline (32.30) |
|---|---|---|
| 200 | 6,063 | +6,031 |
| 600 | 6,618 | +6,586 |
| 1000 | 27,310 | +27,278 |
| 1600 | 47,930 | +47,898 |
| 2000 | 64,699 | +64,667 |
| 2400 | 127,708 | +127,676 |
| **2600** | **11.65** | **−20.65** 🔻 |
| 2800 | 10.47 | −21.83 |
| 3000 | 9.33 | −22.97 |
| 3400 | 8.27 | −24.03 |
| 3800 | 7.63 | −24.67 |
| 4200 | 7.36 | −24.94 |
| 4600–5000 | **7.36** | plateau |

**Best checkpoint:** `phase0/checkpoints/best_connect.pt` — PPL **7.36** vs baseline **32.30**

### Inference test

```bash
python phase0/scripts/inference.py \
    --checkpoint phase0/checkpoints/best_connect.pt \
    --greedy --prompt "Explain about history of AI"
```

**Output (n_iter=3):**
> *"Explain about history of AI . The first two times and its early developments . The 20th century became a time which helped direct the path of international alliance . But the main burden of the cost of the war was not to the land , but by the amount of military logistics ..."*

PPL rất thấp (7.36) nhưng **output inference hoàn toàn sai chủ đề** — model trả lời câu hỏi về AI bằng nội dung về chiến tranh/quân sự.

---

## Phân tích: Tại sao PPL thấp nhưng inference tệ?

### Nguyên nhân 1 — Train và eval dùng chung split (data leakage)

`load_text_dataset` mặc định `split="validation"`. Script gọi nó 2 lần mà **không override split**:

```python
train_ds = load_text_dataset(tokenizer, max_length=args.seq_len, max_samples=args.max_samples)
eval_ds  = load_text_dataset(tokenizer, max_length=args.seq_len, max_samples=50)
```

→ **Cả train và eval đều lấy từ WikiText-2 validation.** Connect layer optimize thẳng trên eval data → PPL eval giảm mạnh, không phản ánh generalization thật.

### Nguyên nhân 2 — Overfit vào WikiText-2 distribution

5000 steps × 1500 samples WikiText-2 = connect layer học cách shift hidden states để predict Wikipedia text tốt hơn. Khi inference với prompt `"Explain about history of AI"`:
- Connect layer kéo hidden states về phân phối Wikipedia
- Wikipedia có nhiều bài về chiến tranh (WWI, WWII) → model generate text về chiến tranh
- Không liên quan prompt

### Nguyên nhân 3 — Không có instruction-following signal

Training chỉ dùng **next-token prediction trên Wikipedia plain text**. Model không bao giờ thấy dạng `"Explain X → trả lời về X"`. PPL thấp chứng minh connect layer giỏi predict Wikipedia, **không phải** giỏi trả lời câu hỏi.

### Tóm tắt

| Hiện tượng | Nguyên nhân |
|---|---|
| PPL 7.36 < baseline 32.30 | Train/eval cùng split → data leakage |
| PPL giảm đột ngột tại step 2600 | Gated layer "lock vào" Wikipedia distribution sau khi overfit |
| Inference lạc chủ đề | Connect layer shift hidden states → WikiText distribution, không phải instruction space |
| Output về chiến tranh dù hỏi AI | WikiText-2 validation dominated by war/history articles |

### Fix đề xuất

1. **Tách split**: `train_ds` dùng `split="train"`, `eval_ds` dùng `split="validation"`
2. **Early stopping**: Dừng khi PPL eval dưới threshold (ví dụ: `< baseline * 0.9`)
3. **Instruction-tuning data**: Thêm instruction-following dataset (Alpaca, FLAN) vào training mix
4. **Monitor train vs eval gap**: PPL eval < train PPL là dấu hiệu overfit/leakage

```python
# Fix in phase0/scripts/train.py
train_ds = load_text_dataset(tokenizer, max_length=args.seq_len,
                             split="train",       # ← thêm
                             max_samples=args.max_samples)
eval_ds  = load_text_dataset(tokenizer, max_length=args.seq_len,
                             split="validation",  # ← explicit
                             max_samples=50)
```

---

*Experiment D added. Data leakage bug documented.*

---

## Experiment E — IterationAwareConnectLayer + Multi-step Aux Loss (Phase 0 Optimization)

### Motivation

Root-cause analysis of Phase 0 poor training (comparing against Ouro 1.4B paper, arxiv 2510.25741):

| Root Cause | Old Behavior | Fix |
|---|---|---|
| Scalar gate `[B,S,1]` | Too coarse — same blend ratio for all d_model dims | Channel-wise gate `[B,S,d]` |
| No iteration awareness | Connect doesn't know which loop step it's on | `iter_emb = nn.Embedding(8, d_model)` |
| No input normalization | Hidden states arrive at arbitrary scale between iterations | Pre-norm on both `h_j` and `h_prev` |
| Single-iter loss | Only final iteration contributes → thin gradient to connect | Multi-step aux loss at every iteration |
| LR 3e-4 too high | Recurrent architectures need lower LR (Ouro paper explicitly warns) | Default `lr = 1e-4` |

---

### Architecture — `IterationAwareConnectLayer` (Option D, ~12M params)

```python
h_j_n    = pre_norm_j(h_j) + iter_emb[step_idx]      # normalize + inject iter context
h_prev_n = pre_norm_prev(h_prev)                       # normalize previous hidden state
gate     = sigmoid(gate_proj(cat([h_j_n, h_prev_n]))) # [B,S,d] — per-dim channel-wise gate
blended  = gate * transform(h_j_n) + (1 - gate) * h_prev_n
output   = post_norm(blended)
```

**Init:** `transform` near-zero (safe warm-start), `gate` bias → 0.5 (neutral start), `iter_emb` small random.

---

### Multi-step Auxiliary Loss

```
L_total = L_final + aux_loss_weight × Σ_i [ γ^(n-1-i) × CE(lm_head(norm(h_out_i)), labels) ]
```

- Geometric decay `γ = 0.5` → earlier iterations contribute less (natural curriculum)
- `lm_head` + `norm` frozen but gradients flow back to connect layer
- Based on Ouro Section 3.1 — expected task loss across all exit steps
- Default: `aux_loss_weight = 0.3`, off by setting to `0.0`

---

### New Phase0Config Defaults

| Field | Old | New |
|---|---|---|
| `n_iter` | 3 | 4 |
| `connect_type` | `gated` | `iter_aware` |
| `lr` (train.py) | 3e-4 | 1e-4 |
| `aux_loss_weight` | — | 0.3 |
| `aux_loss_gamma` | — | 0.5 |
| `consistency_weight` | — | 0.0 (opt-in) |

---

### Smoke Test Result

```
python phase0/scripts/test_forward.py
```

```
IterationAwareConnectLayer summary:
  d_model : 2048
  d_inner : 512
  max_iter: 8
  Trainable params: 10,510,337

✅ Forward OK   loss=15.5781  logits=[1, 32, 151936]
✅ Backward OK
✅ Connect grads: ALL present
✅ Frozen backbone: no gradient (correct)
```

**Status: ✅ PASS** — 10.5M trainable params, all gradients flowing correctly.

---

### Recommended Training Command

```bash
python phase0/scripts/train.py \
    --connect_type iter_aware \
    --n_iter 4 \
    --lr 1e-4 \
    --max_steps 3000 \
    --curriculum \
    --aux_loss_weight 0.3 \
    --aux_loss_gamma 0.5 \
    --max_samples 1500 \
    --batch_size 2 \
    --seq_len 256
```

Note: use `split="train"` in `load_text_dataset` (see Experiment D data leakage fix) before running.

---

*Experiment E: IterationAwareConnectLayer + multi-step aux loss. Smoke test ✅. Training results pending.*
