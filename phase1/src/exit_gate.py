"""
phase1/src/exit_gate.py — Exit Gate cho Phase 1

Quyết định khi nào loop nên dừng lại.

Signals được dùng (từ thiết kế trong looped_lm_design.md):
  S1 — delta norm:       ||h_loop_in^n - h_loop_in^(n-1)||
  S3 — attention entropy (full attention layer cuối loop block)
  S5 — iteration embedding (biết mình đang ở iteration nào)

Differentiability:
  Training → soft gate (differentiable, loss = L_task + β*L_ponder)
  Inference → hard threshold (true skip)

Granularity: per-sequence (aggregate mean over tokens)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ExitGateConfig:
    d_model:      int   = 2048
    max_iter:     int   = 8
    # Training mode: "soft" | "hard" (inference-only)
    mode:         str   = "soft"
    # Threshold cho hard mode (inference)
    threshold:    float = 0.5
    # Ponder loss coefficient
    ponder_beta:  float = 0.01


class ExitGate(nn.Module):
    """
    Exit gate nhận 3 signals:
      h_end:   [B, S, d] — hidden state cuối loop block (layer loop_end-1)
      delta:   [B, S, d] — h_in^n - h_in^(n-1)
      iter_idx: int      — iteration index hiện tại

    Output:
      Training: scalar probability [B] (soft, differentiable)
      Inference: bool [B] với threshold

    TODO (Phase 1 implementation):
      - Thêm attention entropy signal từ full attention layer 15
      - Tune threshold trên validation set sau khi train
    """

    def __init__(self, cfg: ExitGateConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        # S5 — iteration embedding
        self.iter_emb = nn.Embedding(cfg.max_iter, d)

        # Gate MLP: 3*d → d//4 → 1
        self.gate_mlp = nn.Sequential(
            nn.Linear(3 * d, d // 4, bias=False),
            nn.SiLU(),
            nn.Linear(d // 4, 1, bias=True),
        )

        # Init: gate bắt đầu neutral (~0.5 sigmoid)
        nn.init.zeros_(self.gate_mlp[-1].bias)

    def forward(
        self,
        h_end:    torch.Tensor,      # [B, S, d]
        delta:    torch.Tensor,      # [B, S, d]  = h_in^n - h_in^(n-1)
        iter_idx: int,
    ) -> torch.Tensor:
        """
        Returns:
            Training (soft): gate probability [B]  ∈ (0, 1)
            Inference (hard): exit decision  [B]  ∈ {True, False}
        """
        B, S, d = h_end.shape

        # Expand iter embedding to [B, S, d]
        iter_e = self.iter_emb(
            torch.tensor(iter_idx, device=h_end.device)
        ).view(1, 1, d).expand(B, S, -1)

        gate_input = torch.cat([h_end, delta, iter_e], dim=-1)   # [B, S, 3d]
        logit      = self.gate_mlp(gate_input)                    # [B, S, 1]
        prob       = torch.sigmoid(logit).mean(dim=1).squeeze(-1) # [B]

        if self.cfg.mode == "hard":
            return prob > self.cfg.threshold    # [B] bool

        return prob  # [B] soft

    def ponder_loss(self, probs: list[torch.Tensor]) -> torch.Tensor:
        """
        Ponder loss (Graves ACT style): penalize mỗi iteration thêm.
        probs: list of gate probabilities tại mỗi iteration [B]
        """
        # Mean iterations used ≈ weighted sum
        total = torch.zeros_like(probs[0])
        for i, p in enumerate(probs):
            total = total + (i + 1) * p
        return total.mean()
