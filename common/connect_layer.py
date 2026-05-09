"""
Connect Layer — dùng chung cho tất cả phases.
Implements: conservative residual + MLP + gated + iteration-aware variants.

Mục tiêu: remap hidden state từ loop block output (layer j space)
về loop block input (layer i space).

Input/Output shape: [batch, seq_len, d_model]  (d_model = 2048 với Qwen3-1.7B)
"""

import torch
import torch.nn as nn


def _build_norm(d_model: int) -> nn.Module:
    """RMSNorm (PyTorch ≥2.4) với fallback sang LayerNorm."""
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(d_model, eps=1e-6)
    # Fallback cho PyTorch < 2.4
    return nn.LayerNorm(d_model, eps=1e-6, elementwise_affine=True)


# ---------------------------------------------------------------------------
# Phương án A — Conservative Residual
# ---------------------------------------------------------------------------

class ResidualConnectLayer(nn.Module):
    """
    Conservative remap: h_j -> delta, then add a small learned delta to h_prev.

    Warm-start behavior is exactly stable for a looped backbone:
        output = h_prev

    This keeps the hidden state on the original layer-i manifold at step 0.
    The trainable path can then learn only the correction needed to reuse the
    loop block, instead of freely rewriting/renormalizing the whole hidden.
    """

    def __init__(self, d_model: int = 2048, bottleneck_ratio: float = 0.25):
        super().__init__()
        d_inner = max(64, int(d_model * bottleneck_ratio))

        self.pre_norm = _build_norm(d_model)
        self.down_proj = nn.Linear(d_model, d_inner, bias=False)
        self.up_proj = nn.Linear(d_inner, d_model, bias=False)
        self.act = nn.SiLU()

        # Start from no-op. The scalar is also learned, but initialized small.
        self.residual_scale = nn.Parameter(torch.tensor(0.01))

        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, h_j: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        delta = self.up_proj(self.act(self.down_proj(self.pre_norm(h_j))))
        return h_prev + self.residual_scale.to(dtype=h_prev.dtype) * delta


# ---------------------------------------------------------------------------
# Phương án B — MLP Bottleneck
# ---------------------------------------------------------------------------

class MLPConnectLayer(nn.Module):
    """
    Non-linear projection với bottleneck:
        h_j → Linear(d, d_inner) → SiLU → Linear(d_inner, d)
        output = h_prev + delta

    Params (d=2048, d_inner=512): ~2.1M
    Kept for ablation. Warm-start is no-op when h_prev is provided.
    """

    def __init__(self, d_model: int = 2048, bottleneck_ratio: float = 0.25):
        super().__init__()
        d_inner = max(64, int(d_model * bottleneck_ratio))

        self.down_proj = nn.Linear(d_model, d_inner, bias=False)
        self.up_proj   = nn.Linear(d_inner, d_model, bias=False)
        self.act       = nn.SiLU()

        # Khởi tạo no-op để tránh training collapse ban đầu.
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, h_j: torch.Tensor, h_prev: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            h_j: [B, S, d_model] — output của loop block (layer j hidden state)
            h_prev: [B, S, d_model] — input của loop block iteration hiện tại
        Returns:
            h_i_hat: [B, S, d_model] — remapped để feed vào loop block input
        """
        projected = self.up_proj(self.act(self.down_proj(h_j)))
        if h_prev is None:
            return projected
        return h_prev + projected


# ---------------------------------------------------------------------------
# Phương án C — Gated Residual
# ---------------------------------------------------------------------------

class GatedResidualConnectLayer(nn.Module):
    """
    Gated residual update từ transformed h_j lên h_i từ iteration trước:
        gate      = sigmoid(W_g @ h_j)
        transform = MLP(h_j)
        output    = h_i_prev + gate * transform

    Đặc điểm:
    - gate → 1: lấy nhiều từ new iteration (aggressive update)
    - gate → 0: giữ nguyên hidden state cũ (conservative, stable)
    - Gradient flow ổn định hơn B vì có residual path sang h_i_prev

    Params (d=2048, d_inner=512): ~6M
    """

    def __init__(self, d_model: int = 2048, bottleneck_ratio: float = 0.25):
        super().__init__()
        d_inner = max(64, int(d_model * bottleneck_ratio))

        # Transform MLP (remap semantic space)
        self.transform = nn.Sequential(
            nn.Linear(d_model, d_inner, bias=False),
            nn.SiLU(),
            nn.Linear(d_inner, d_model, bias=False),
        )
        # Gate network (scalar per token)
        self.gate_proj = nn.Linear(d_model, 1, bias=True)

        # Init: transform gần zero ban đầu, nên output ban đầu đúng bằng h_i_prev.
        nn.init.normal_(self.transform[0].weight, std=0.02)
        nn.init.zeros_(self.transform[2].weight)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(
        self,
        h_j: torch.Tensor,
        h_i_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_j:      [B, S, d_model] — output của loop block iteration n
            h_i_prev: [B, S, d_model] — loop block input của iteration n-1
                      (iteration 0: dùng h_i^(0) ban đầu)
        Returns:
            h_i_next: [B, S, d_model] — input cho loop block iteration n+1
        """
        gate = torch.sigmoid(self.gate_proj(h_j))   # [B, S, 1]
        transformed = self.transform(h_j)            # [B, S, d_model]

        return h_i_prev + gate * transformed

    def gate_stats(self, h_j: torch.Tensor) -> dict:
        """Utility: trả về gate statistics để debug training."""
        with torch.no_grad():
            gate = torch.sigmoid(self.gate_proj(h_j))
            return {
                "gate_mean": gate.mean().item(),
                "gate_std":  gate.std().item(),
                "gate_min":  gate.min().item(),
                "gate_max":  gate.max().item(),
            }


# ---------------------------------------------------------------------------
# Phương án D — Iteration-Aware Connect (Phase 0 optimized)
# ---------------------------------------------------------------------------

class IterationAwareConnectLayer(nn.Module):
    """
    Upgraded connect layer dựa trên insights từ Ouro paper:

    Cải tiến so với GatedResidualConnectLayer:
    1. Pre-norm cả hai inputs (h_j và h_prev) → stable activation scale
    2. Iteration embedding — biết đang ở loop step nào → context-aware blending
    3. Channel-wise gate [B, S, d_model] thay vì scalar [B, S, 1] → per-dim control
    4. Gate dùng cả h_j và h_prev (concatenated) → richer gating signal
    5. No-op warm-start: output starts exactly at h_prev

    Forward:
        h_j_n    = pre_norm_j(h_j) + iter_emb[step_idx]
        h_prev_n = pre_norm_prev(h_prev)
        gate     = sigmoid(gate_proj(cat([h_j_n, h_prev_n])))   # [B, S, d]
        transform = MLP(h_j_n)
        return h_prev + gate * transform

    Params (d=2048, d_inner=512, max_iter=8): ~12M
    """

    def __init__(
        self,
        d_model:          int   = 2048,
        bottleneck_ratio: float = 0.25,
        max_iter:         int   = 8,
    ):
        super().__init__()
        d_inner = max(64, int(d_model * bottleneck_ratio))

        self.pre_norm_j    = _build_norm(d_model)
        self.pre_norm_prev = _build_norm(d_model)

        # Iteration index embedding — informs connect which loop step we're at
        self.iter_emb = nn.Embedding(max_iter, d_model)

        # Transform MLP (remap semantic space)
        self.transform = nn.Sequential(
            nn.Linear(d_model, d_inner, bias=False),
            nn.SiLU(),
            nn.Linear(d_inner, d_model, bias=False),
        )

        # Channel-wise gate: uses BOTH h_j and h_prev → richer signal
        self.gate_proj = nn.Linear(d_model * 2, d_model, bias=True)

        # Init: transform near zero → output ≈ h_prev at start (safe warm-start)
        nn.init.normal_(self.transform[0].weight, std=0.02)
        nn.init.zeros_(self.transform[2].weight)
        # Gate init: uniform blend (gate ≈ 0.5) at start
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        # Iter emb: small random
        nn.init.normal_(self.iter_emb.weight, std=0.01)

    def forward(
        self,
        h_j:       torch.Tensor,
        h_prev:    torch.Tensor,
        step_idx:  int = 0,
    ) -> torch.Tensor:
        """
        Args:
            h_j:      [B, S, d_model] — output của loop block iteration n
            h_prev:   [B, S, d_model] — loop block input của iteration n
            step_idx: int — 0-indexed loop iteration number
        Returns:
            h_next:   [B, S, d_model] — input cho loop block iteration n+1
        """
        device = h_j.device
        iter_idx = torch.tensor(step_idx, device=device)

        h_j_n    = self.pre_norm_j(h_j) + self.iter_emb(iter_idx)   # [B, S, d]
        h_prev_n = self.pre_norm_prev(h_prev)                         # [B, S, d]

        combined = torch.cat([h_j_n, h_prev_n], dim=-1)              # [B, S, 2d]
        gate     = torch.sigmoid(self.gate_proj(combined))            # [B, S, d]

        transformed = self.transform(h_j_n)                           # [B, S, d]
        return h_prev + gate * transformed

    def gate_stats(self, h_j: torch.Tensor, h_prev: torch.Tensor, step_idx: int = 0) -> dict:
        """Utility: gate statistics để debug training."""
        with torch.no_grad():
            device   = h_j.device
            iter_idx = torch.tensor(step_idx, device=device)
            h_j_n    = self.pre_norm_j(h_j) + self.iter_emb(iter_idx)
            h_prev_n = self.pre_norm_prev(h_prev)
            gate     = torch.sigmoid(self.gate_proj(torch.cat([h_j_n, h_prev_n], dim=-1)))
            return {
                "gate_mean": gate.mean().item(),
                "gate_std":  gate.std().item(),
                "gate_min":  gate.min().item(),
                "gate_max":  gate.max().item(),
            }
