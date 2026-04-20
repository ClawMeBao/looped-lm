# Looped Language Model — Tổng hợp thiết kế

> Dựa trên Qwen3.5-2B. Mục tiêu: biến một pretrained LM thành mô hình có khả năng iterative reasoning bằng cách thêm loop block + connect layer + exit gate, với frozen base model.

---

## 1. Kiến trúc tổng quan

```
Input Tokens
    │
    ▼
┌─────────────────────────┐
│   PREFIX (frozen)       │  Layers 0–7
│   Early features        │  syntax, local context
└────────────┬────────────┘
             │ h_8^(0)
             ▼
┌────────────────────────────────────────────┐
│           LOOP BLOCK (frozen)              │  ◄── lặp n lần
│   Layers 8–15  (Groups 2–3)               │
│   L8,9,10 (linear) + L11 (full)           │
│   L12,13,14 (linear) + L15 (full)         │
└────────────┬───────────────────────────────┘
             │ h_16^(n)
             ▼
┌────────────────────────────────────────────┐
│   CONNECT LAYER (trainable)                │  remap h_16 → h_8 space
└────────────┬───────────────────────────────┘
             │
        ┌────┴─────┐
        ▼          ▼
   EXIT GATE   continue loop
   (trainable)
        │ exit
        ▼
┌─────────────────────────┐
│   SUFFIX (frozen)       │  Layers 16–23
│   Late features         │  output refinement
└────────────┬────────────┘
             ▼
         LM Head
```

**Parameters cần train:**
- Connect Layer: ~5–10M params
- Exit Gate: ~1.5M params
- **Tổng: < 0.5% tổng params của model (2B)**

---

## 2. Phân tích kiến trúc Qwen3.5-2B

```
num_hidden_layers : 24
hidden_size       : 2048
intermediate_size : 6144
num_attn_heads    : 8   (head_dim = 256)
num_kv_heads      : 2   (GQA 4:1)
layer_types       : [linear, linear, linear, full] × 6  (full_attention_interval = 4)
attn_output_gate  : true
partial_rotary_factor: 0.25
mtp_num_hidden_layers: 1  (Multi-Token Prediction head)
```

**6 natural groups:**

| Group | Layers | Zone | Vai trò |
|-------|--------|------|---------|
| 0 | 0–3 | Early | Local syntax, n-gram patterns |
| 1 | 4–7 | Early | Shallow semantic, local context |
| **2** | **8–11** | **Middle** | **Semantic composition** |
| **3** | **12–15** | **Middle** | **Reasoning, inference** |
| 4 | 16–19 | Late | Output refinement |
| 5 | 20–23 | Late | Vocabulary bias, LM head preparation |

**Loop Block = Layers 8–15** vì:
- Middle zone có distribution gap nhỏ nhất (CKA studies)
- Chứa 2 full_attention cycles → 2 global context integrations per iteration
- Balanced 1/3 split: prefix (0–7) / loop (8–15) / suffix (16–23)
- Ranh giới tự nhiên đúng group boundary

---

## 3. Connect Layer — Các phương án

### Vấn đề cốt lõi

`h_16` sống trong **semantic space của layer 16** (high abstraction).
`h_8` phải sống trong **semantic space của layer 8** (lower abstraction).

Đây là **approximate inverse mapping** — không tồn tại closed-form solution.

---

### 3.1 Phương án A — Linear + Norm (Baseline)

```python
C(h) = LayerNorm(W @ h + b)
# W ∈ R^(d × d), d = 2048
# Params: ~4M
```

| Tiêu chí | Đánh giá |
|----------|----------|
| Expressivity | ⭐⭐ Thấp — linear không đủ cho semantic remapping |
| Gradient flow | ✅ Ổn định |
| Params | 4M |
| Training ease | ✅ Rất dễ |
| Use case | Baseline benchmark only |

---

### 3.2 Phương án B — MLP Bottleneck

```python
C(h) = LayerNorm(
    Linear(SiLU(Linear(h, d_inner)), d_model)
)
# d_inner = d_model // 4 = 512
# Params: ~2M
```

| Tiêu chí | Đánh giá |
|----------|----------|
| Expressivity | ⭐⭐⭐ Trung bình |
| Gradient flow | ✅ Ổn định |
| Params | ~2M |
| Training ease | ✅ Dễ |
| Thiếu | Không có iteration awareness, không có memory |

---

### 3.3 Phương án C — Gated Residual *(Khuyến nghị làm trước)*

```python
gate      = sigmoid(W_g @ h_j)           # [B, S, 1] — learned per token
transform = MLP(h_j)                     # remap semantic space
output    = gate * transform + (1 - gate) * h_i_prev
# h_i_prev: hidden state của vòng trước tại loop input
# Params: ~6M
```

| Tiêu chí | Đánh giá |
|----------|----------|
| Expressivity | ⭐⭐⭐⭐ Cao |
| Gradient flow | ✅✅ Rất tốt (residual path) |
| Params | ~6M |
| Memory overhead | +1 tensor [B, S, 2048] |
| Đặc điểm | Tự học "pace of update" — gate → 0 là conservative |

---

### 3.4 Phương án D — Iteration-Aware MLP

```python
iter_emb = nn.Embedding(max_iter, d_model)   # 8 × 2048 = 16K params
C(h, n)  = LayerNorm(MLP(h + iter_emb[n]))
```

| Tiêu chí | Đánh giá |
|----------|----------|
| Expressivity | ⭐⭐⭐⭐ Cao |
| Iteration awareness | ✅ Biết đang ở iteration nào |
| Params | ~6M + 16K |
| Analogy | Như positional encoding nhưng cho "reasoning depth" |

---

### 3.5 Phương án E — LoRA-style Adapter

```python
C(h) = h + alpha * (B @ (A @ h))
# A ∈ R^(r × d), B ∈ R^(d × r), r = 64
# Params: 2 × 2048 × 64 = 262K
```

| Tiêu chí | Đánh giá |
|----------|----------|
| Expressivity | ⭐⭐ Thấp–Trung bình |
| Params | 262K — cực nhỏ |
| Training ease | ✅ Proven (LoRA literature) |
| Thiếu | Không có gating, expressivity thấp |
| Use case | Khi muốn minimal params hoặc ablation |

---

### 3.6 So sánh Connect Layer

| Phương án | Expressivity | Gradient | Params | Khuyến nghị |
|-----------|-------------|----------|--------|-------------|
| A: Linear+Norm | ⭐⭐ | ✅ | 4M | Baseline only |
| B: MLP Bottleneck | ⭐⭐⭐ | ✅ | 2M | Phase 1 start |
| **C: Gated Residual** | **⭐⭐⭐⭐** | **✅✅** | **6M** | **✅ Ưu tiên** |
| D: Iteration-Aware | ⭐⭐⭐⭐ | ✅✅ | 6M+ | ✅ Phase 2 |
| E: LoRA | ⭐⭐ | ✅ | 262K | Minimal/ablation |

**Best combo (Phase 2):** C + D = Gated Residual + Iteration Embedding

---

## 4. Linear Attention State — Quyết định riêng cho Qwen3.5

Qwen3.5 có SSM-style linear attention state:
```
state^(t) = state^(t-1) * decay + k^(t) ⊗ v^(t)
out^(t)   = state^(t) @ q^(t)
```

Khi bước vào vòng loop mới, có 2 lựa chọn:

| Lựa chọn | Cách làm | Pros | Cons |
|----------|----------|------|------|
| **Reset** | Zero state mỗi iteration | Safe, deterministic, dễ debug | Mất continuity giữa loops |
| **Carry** | Giữ state qua iterations | State = "compressed loop memory" tự nhiên | Khó train, state drift |

**Khuyến nghị:** Bắt đầu với Reset, thử Carry sau khi baseline ổn định.

---

## 5. Exit Gate — Các tín hiệu (Signals)

### 5.1 Signal S1 — Representation Convergence (delta norm)

```python
delta = h_8^(n) - h_8^(n-1)
signal = ||delta||_2 / ||h_8^(n-1)||_2   # normalized
```

| | |
|---|---|
| Ý nghĩa | Hidden state thay đổi bao nhiêu so với vòng trước |
| Pros | Không cần thêm params; unsupervised |
| Cons | L2 norm ≠ semantic quality; threshold không có giá trị tự nhiên |

---

### 5.2 Signal S2 — Output Confidence (shallow probe)

```python
logit      = W_probe @ h_15^(n)    # W_probe ∈ R^(vocab × d) — lightweight
confidence = max(softmax(logit))
```

| | |
|---|---|
| Ý nghĩa | Model "chắc" bao nhiêu về output tiếp theo |
| Pros | Có semantic meaning rõ ràng |
| Cons | Model có thể overconfident sai; cần train probe riêng |

---

### 5.3 Signal S3 — Full Attention Entropy

```python
attn_weights = softmax(Q @ K.T / sqrt(d))       # layer 11 hoặc 15
entropy = -sum(attn_weights * log(attn_weights + eps), dim=-1)
signal = mean(entropy)   # per sequence
```

| | |
|---|---|
| Ý nghĩa | Entropy cao → attention "tìm kiếm" → chưa converged |
| Pros | Nội sinh từ architecture; đặc biệt mạnh với `attn_output_gate=true` |
| Cons | Cần lưu attention matrix → memory overhead |

---

### 5.4 Signal S4 — Linear Attention State Saturation

```python
rate = ||state^(n)||_F / ||state^(n-1)||_F
signal = abs(rate - 1.0)   # gần 0 → state không còn cập nhật
```

| | |
|---|---|
| Ý nghĩa | SSM memory đã "đầy" chưa |
| Pros | Free signal từ Qwen3.5 linear attention architecture |
| Cons | Decay factor có thể làm norm ổn định giả tạo |

---

### 5.5 Signal S5 — Iteration Index (positional prior)

```python
iter_emb = nn.Embedding(max_iter, d_gate_input)
```

| | |
|---|---|
| Ý nghĩa | Gate biết mình đang ở iteration nào |
| Pros | Cực rẻ (16K params); học prior "thường exit ở iteration nào" |
| Cons | Không adaptive — cùng iteration có thể cần exit hoặc không |

---

## 6. Exit Gate — Cấu trúc và differentiability

### 6.1 Hard Gate

```python
exit = (gate_logit > threshold)   # discrete, True/False
```
- ✅ True skip — không tốn compute khi exit
- ❌ Không differentiable → không train được trực tiếp

### 6.2 Soft Gate (training-time)

```python
exit_prob = sigmoid(gate_logit)
output = exit_prob * h_via_suffix + (1 - exit_prob) * h_continue_loop
```
- ✅ Fully differentiable
- ❌ Vẫn chạy full loop — không tiết kiệm compute lúc train

### 6.3 ACT — Adaptive Computation Time (Graves 2016)

```python
halt_prob = sigmoid(gate_mlp(h^(n)))
cumulative = sum(halt_prob over iterations)
exit khi cumulative > 1 - epsilon
```
- ✅ Differentiable với ponder loss; provably efficient
- ❌ Phức tạp để implement; training unstable nếu tune λ sai

### 6.4 Straight-Through Estimator (STE)

```python
# Forward: hard threshold
exit = (gate_prob > 0.5).float()
# Backward: treat as if soft
exit = gate_prob   # gradient đi qua
```
- ✅ True skip lúc forward; gradient approximate
- ❌ Gradient bias; unstable trong một số trường hợp

### 6.5 Khuyến nghị: Soft (train) → Hard (infer)

```python
if training:
    output = gate_prob * suffix_out + (1 - gate_prob) * loop_continue
else:
    if gate_prob > threshold: break
```
- ✅ Differentiable trong training
- ✅ True skip trong inference
- ✅ Đơn giản nhất để implement đúng

---

## 7. Exit Gate — Training signal

### 7.1 Ponder Loss (Graves ACT style)

```python
L_ponder = β * sum_over_batch(mean_iterations_used)
L_total  = L_lm + β * L_ponder
```
- Không cần oracle labels
- β điều chỉnh trade-off accuracy/speed

### 7.2 Consistency Loss

```python
L_consist = KL(output^(n_exit) || output^(MAX_ITER).detach())
L_total   = L_lm + λ * L_consist
```
- Ý nghĩa: output khi exit sớm phải nhất quán với output khi chạy đủ vòng
- Dạy gate exit khi đã converge về semantic

### 7.3 Oracle Teacher

```python
# Offline: chạy MAX_ITER, tìm iteration n* đầu tiên mà task loss < ε
# Online: supervise gate bằng n*
L_gate = CrossEntropy(gate_logit, oracle_label)
```
- Cần precompute oracle → tốn compute offline
- Mạnh nhất về signal quality, nhưng không scalable

### 7.4 Khuyến nghị: Ponder + Consistency

```
L_total = L_lm + β * L_ponder + λ * L_consistency
```
Không cần oracle, end-to-end trainable.

---

## 8. Gradient flow qua loop

Backprop qua N iterations của frozen block → `N × 8 layers` gradient steps → **vanishing/exploding risk**.

**Giải pháp: Truncated BPTT**

```python
K = 3  # chỉ backprop qua K iterations gần nhất

for n in range(N):
    if n < N - K:
        h = h.detach()   # stop gradient
    h_out = loop_block(connect_layer(h))
```

- K = 2–3 đủ trong practice
- Giảm memory và gradient instability đáng kể
- Trade-off: gradient không flow qua early iterations

---

## 9. Các Option triển khai — Đánh giá tổng thể

### Option 1 — "Safe Start" (Minimal risk, maximum learnability)

**Thành phần:**
| Component | Lựa chọn |
|-----------|----------|
| Connect Layer | MLP Bottleneck (Phương án B) |
| Linear Attn State | Reset mỗi iteration |
| Exit Gate signals | delta norm + iter_emb (S1 + S5) |
| Gate differentiability | Soft (train) / Hard (infer) |
| Gate granularity | Per-sequence |
| Training signal | Ponder Loss |
| Gradient | Truncated BPTT, K=2 |

**Params trainable:** ~3.5M
**Ưu điểm:**
- Dễ debug nhất
- Ít hyperparameters cần tune
- Gradient ổn định
- Có thể chạy trên single GPU nhỏ

**Nhược điểm:**
- Expressivity thấp ở connect layer
- Tín hiệu exit đơn giản, có thể kém accuracy
- Không tận dụng được linear attention state

**Phù hợp khi:** Muốn verify ý tưởng trước, chạy ablation, hardware hạn chế.

---

### Option 2 — "Balanced" *(Khuyến nghị)*

**Thành phần:**
| Component | Lựa chọn |
|-----------|----------|
| Connect Layer | Gated Residual + MLP (Phương án C) |
| Linear Attn State | Reset (an toàn) |
| Exit Gate signals | h_15 + delta + iter_emb (S1 + S3 + S5) |
| Gate differentiability | Soft (train) / Hard (infer) |
| Gate granularity | Per-sequence |
| Training signal | Ponder + Consistency |
| Gradient | Truncated BPTT, K=3 |

**Params trainable:** ~8M

**Ưu điểm:**
- Gated Residual đảm bảo gradient flow ổn định và tránh catastrophic forgetting iteration trước
- Combo (delta + entropy + iter_emb) cho gate đủ signal mà không overfit
- Soft/Hard split: train clean, inference nhanh
- Ponder + Consistency là cặp loss bổ sung tốt nhau

**Nhược điểm:**
- Phức tạp hơn Option 1
- Cần tune 2 loss weights (β, λ)
- Memory overhead thêm 1 tensor [B, S, 2048] cho gated residual

**Risk level:** Trung bình — từng component đã có tiền lệ trong literature

**Phù hợp khi:** Muốn kết quả tốt mà vẫn implement được trong 1–2 tuần.

---

### Option 3 — "Full Power" (Maximum capability)

**Thành phần:**
| Component | Lựa chọn |
|-----------|----------|
| Connect Layer | Gated Residual + Iteration-Aware MLP (C + D) |
| Linear Attn State | **Carry state** (SSM memory qua iterations) |
| Exit Gate signals | h_15 + delta + attn_entropy + state_norm + iter_emb (S1+S2+S3+S4+S5) |
| Gate differentiability | ACT với Ponder Loss |
| Gate granularity | Per-token soft → per-sequence hard |
| Training signal | Ponder + Consistency + Oracle (hybrid) |
| Gradient | Truncated BPTT K=3 + LoRA unfreeze 1% loop block |

**Params trainable:** ~15M + 1% loop block (LoRA)

**Ưu điểm:**
- Tận dụng tối đa kiến trúc Qwen3.5 (carry state = free memory)
- Gate cực kỳ informative, decision quality cao nhất
- Per-token routing tiềm năng mạnh hơn per-sequence
- LoRA unfreeze nhỏ cho phép loop block adapt với loop context

**Nhược điểm:**
- Implementation phức tạp đáng kể
- Carry state rất khó debug và tránh gradient issues
- ACT cần tune cẩn thận, dễ collapse (gate luôn exit sớm hoặc không bao giờ exit)
- Cần nhiều GPU memory hơn
- Nhiều hyperparameters: β_ponder, λ_consist, λ_oracle, LoRA rank, decay rate

**Risk level:** Cao — nhiều moving parts, dễ có unexpected interaction

**Phù hợp khi:** Có infrastructure tốt, muốn push SOTA, có time để experiment kỹ.

---

### Option 4 — "Fixed-depth, No Gate" (Simplified variant)

**Thành phần:**
| Component | Lựa chọn |
|-----------|----------|
| Connect Layer | MLP Bottleneck (B) |
| Linear Attn State | Reset |
| Exit Gate | **Không có** — fixed N iterations |
| N | Curriculum: train với N=1, 2, 3, 4 dần dần |
| Gradient | Truncated BPTT K=2 |

**Params trainable:** ~2M

**Ưu điểm:**
- Đơn giản nhất — chỉ cần train connect layer
- Không có gate collapse risk
- Dễ reproduce
- Chứng minh loop concept trước khi thêm gate

**Nhược điểm:**
- Không adaptive — mọi input đều chạy N iterations
- Inference cost cố định, không efficient
- Thiếu một phần ý tưởng gốc

**Phù hợp khi:** Proof-of-concept nhanh nhất, verify loop block selection trước khi đầu tư vào gate.

---

## 10. So sánh tổng thể 4 Options

| | Option 1 | Option 2 | Option 3 | Option 4 |
|---|---|---|---|---|
| **Tên** | Safe Start | Balanced | Full Power | No-Gate PoC |
| **Params** | ~3.5M | ~8M | ~15M+ | ~2M |
| **Implementation effort** | Thấp | Trung bình | Cao | Rất thấp |
| **Training stability** | ✅✅ | ✅✅ | ⚠️ | ✅✅ |
| **Expected performance** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Debug difficulty** | Dễ | Trung bình | Khó | Dễ nhất |
| **Inference efficiency** | ✅ | ✅✅ | ✅✅ | ❌ |
| **Tận dụng Qwen3.5 đặc thù** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ |

---

## 11. Lộ trình đề xuất

```
Phase 0 (1 tuần): Option 4 — No-Gate PoC
    └── Mục tiêu: Xác nhận loop block L8–L15 hoạt động
    └── Metric: perplexity với N=1,2,3,4 so với baseline

Phase 1 (2 tuần): Option 1 — Safe Start
    └── Mục tiêu: Thêm exit gate, verify training signal
    └── Metric: iterations/sample, accuracy vs speed trade-off

Phase 2 (3 tuần): Option 2 — Balanced
    └── Mục tiêu: Full pipeline production-quality
    └── Metric: benchmark trên reasoning tasks (GSM8K, ARC...)

Phase 3 (optional): Option 3 — Full Power
    └── Nếu Phase 2 cho kết quả tốt
    └── Cần thêm GPU resource và experiment budget
```

---

## 12. Open questions cần giải quyết khi implementation

1. **Loop block depth:** 8 layers (L8–L15) hay thử 4 layers (1 group) trước?
2. **Max iterations N:** 4 hay 8? Nên bắt đầu thấp để kiểm soát inference cost
3. **Training data:** SFT hay pretrain-style continual? Với task cụ thể nào?
4. **Evaluation protocol:** Làm thế nào đo "iteration contributed to answer quality"?
5. **KV cache:** Với causal LM, KV cache của loop block tích lũy qua iterations như thế nào?
6. **Qwen3.5 MTP head:** Multi-Token Prediction head có cần điều chỉnh không?

---

*File: `looped_lm/looped_lm_design.md`*
*Ngày: 2026-04-20*
