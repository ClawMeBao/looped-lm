#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

source /home/tieubaoca/ai_env/bin/activate

python3 phase0/scripts/train.py \
    --output_dir       phase0/checkpoints/residual_think_100k \
    --dataset          Jackrong/GLM-5.1-Reasoning-1M-Cleaned \
    --dataset_type     glm \
    --max_samples      100000 \
    --filter_think_len 13636 \
    --n_iter           3 \
    --seq_len          4096 \
    --batch_size       2 \
    --grad_accum       8 \
    --epochs           3 \
    --lr               5e-5 \
    --curriculum \
    --connect_type     residual \
    --attn_impl        sdpa \
    --gradient_checkpointing \
    --loss_chunk_size  512 \
    --dtype            bfloat16 \
    --save_every       50  \
    --eval_steps       200 \
    --skip_baseline_eval \
    --baseline_ppl     3.56 \
    --log_steps        10
