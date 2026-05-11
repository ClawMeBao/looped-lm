#!/usr/bin/env bash
# =============================================================================
#  Kaggle 2xT4 — LoopedLM Phase 0 training script
#
#  Architecture: device_map="auto" splits frozen Qwen3-1.7B across 2 GPUs.
#                Only ~6M connect-layer params are trained.
#
#  T4 notes:
#    - SM 7.5 → no native BF16 hardware → use --dtype float16
#    - 16 GB VRAM × 2 = 32 GB total; model ~3.5 GB leaves ~12 GB/card for acts
#    - Session limit: ~9 hours (GPU T4 x2 quota)
#
#  Usage
#  -----
#  Session 1 (fresh start):
#    bash scripts/kaggle_train.sh
#
#  Session 2+ (auto-resumes from output_dir/resume.pt):
#    bash scripts/kaggle_train.sh
#
#  Override step budget per session:
#    EXTRA_STEPS=1500 bash scripts/kaggle_train.sh
#
#  Override baseline PPL (skip eval if already known):
#    BASELINE_PPL=3.56 bash scripts/kaggle_train.sh
#
#  Dry-run / smoke test (no GPU needed):
#    SMOKE=1 bash scripts/kaggle_train.sh
# =============================================================================

set -euo pipefail

# ── Configurable via env vars ─────────────────────────────────────────────────
REPO_ROOT="${REPO_ROOT:-/kaggle/working/looped-lm}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/phase0/checkpoints/kaggle_run}"
DATASET_CACHE="${DATASET_CACHE:-${REPO_ROOT}/data/glm_dataset}"

# Per-session step budget. Tune to fit within Kaggle's 9-hour window.
# At seq_len=2048, batch=2 on T4×2: ~1.5–2 s/step → 1000 steps ≈ 30–40 min.
EXTRA_STEPS="${EXTRA_STEPS:-2000}"

# Known baseline PPL from a previous run; set to skip re-evaluation.
# Leave empty ("") to run baseline eval (adds ~10 min per session).
BASELINE_PPL="${BASELINE_PPL:-}"

# Smoke test: tiny run to verify everything wires up
SMOKE="${SMOKE:-0}"

# ── Derived ──────────────────────────────────────────────────────────────────
cd "${REPO_ROOT}"

echo "============================================================"
echo "  LoopedLM — Kaggle 2×T4 Training"
echo "  REPO:        ${REPO_ROOT}"
echo "  OUTPUT:      ${OUTPUT_DIR}"
echo "  EXTRA_STEPS: ${EXTRA_STEPS}"
echo "  BASELINE_PPL:${BASELINE_PPL:-<will eval>}"
echo "============================================================"

# ── 1. Install / verify dependencies ─────────────────────────────────────────
echo "[setup] Installing dependencies..."
pip install -q -r requirements.txt

# sdpa is PyTorch built-in — no extra packages needed.
# flash_attention_2 is optional; only has prebuilt wheels for torch <= 2.7.
# Default: use sdpa (--attn_impl sdpa in CMD below).

# ── 2. GPU sanity check ───────────────────────────────────────────────────────
python - <<'PYCHECK'
import torch, sys
n = torch.cuda.device_count()
if n == 0:
    print("[warn] No CUDA devices found — running on CPU (smoke only)")
else:
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  {p.total_memory/1e9:.1f} GB  sm_{p.major}{p.minor}")
PYCHECK

# ── 3. Build train command ────────────────────────────────────────────────────
CMD=(
    python phase0/scripts/train.py

    # ── Model ────────────────────────────────────────────────────────────────
    --model          "Qwen/Qwen3-1.7B"
    --connect_type   residual
    --n_iter         3
    --loop_start     8
    --loop_end       20

    # ── Data ─────────────────────────────────────────────────────────────────
    --dataset_type   glm
    --max_samples    50000          # stream only 50k rows; ~30k train examples
    --seq_len        2048           # fits ~58% of GLM thinking chains
    --local_dataset_dir "${DATASET_CACHE}"

    # ── Training ─────────────────────────────────────────────────────────────
    --dtype          float16        # T4 (SM 7.5) has no native BF16 hardware
    --batch_size     2              # 2 × grad_accum=4 = effective batch 8
    --grad_accum     4              # accumulate 4 micro-batches per optimizer step
    --lr             5e-5
    --warmup_ratio   0.05
    --grad_clip      1.0
    --k_bptt         2
    --curriculum                    # ramp n_iter 2→3 over training
    --attn_impl      sdpa           # PyTorch built-in SDPA (no extra package)

    # ── Session management (Kaggle-friendly) ─────────────────────────────────
    --save_every     50             # write resume.pt every 50 steps
    --extra_steps    "${EXTRA_STEPS}"

    # ── Eval ─────────────────────────────────────────────────────────────────
    --eval_steps     200
    --eval_max_batches 50           # fast mid-run eval (~50 batches)
    --skip_baseline_eval         # skip baseline eval if PPL already known (saves ~10 min/session)
    # ── Logging ──────────────────────────────────────────────────────────────
    --log_steps      10
    --num_workers    2              # Kaggle CPU is limited
    --output_dir     "${OUTPUT_DIR}"
)

# Inject known baseline PPL to skip the ~10-min eval at session start
if [[ -n "${BASELINE_PPL}" ]]; then
    CMD+=(--skip_baseline_eval --baseline_ppl "${BASELINE_PPL}")
fi

# Remove --flash_attn block (no longer used; we now use --attn_impl sdpa)

# Smoke test overrides: tiny data, few steps, no dataset download
if [[ "${SMOKE}" == "1" ]]; then
    echo "[smoke] Overriding to smoke-test settings..."
    CMD+=(
        --dataset_type   text        # no download needed
        --seq_len        64
        --batch_size     1
        --extra_steps    5
        --eval_steps     5
        --save_every     5
        --num_workers    0
    )
    # Remove GLM-specific args (smoke uses text dataset)
    FILTERED=()
    SKIP_NEXT=0
    for arg in "${CMD[@]}"; do
        if [[ "${SKIP_NEXT}" == "1" ]]; then SKIP_NEXT=0; continue; fi
        case "${arg}" in
            --max_samples|--local_dataset_dir) SKIP_NEXT=1; continue ;;
        esac
        FILTERED+=("${arg}")
    done
    CMD=("${FILTERED[@]}")
fi

echo ""
echo "[train] Command:"
echo "  ${CMD[*]}"
echo ""

# ── 4. Run ────────────────────────────────────────────────────────────────────
"${CMD[@]}"

echo ""
echo "============================================================"
echo "  Session complete. Checkpoint: ${OUTPUT_DIR}/resume.pt"
echo "  Best model:       ${OUTPUT_DIR}/best_connect.pt"
echo ""
echo "  To continue next session, re-run this script."
echo "  (resume.pt is auto-detected on next run)"
echo "============================================================"
