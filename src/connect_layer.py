"""
Connect Layer — Phase 0 prototype
Implements: MLPConnectLayer (Phương án B) + GatedResidualConnectLayer (Phương án C)

Mục tiêu: remap hidden state từ loop block output (layer j space)
về loop block input (layer i space).

Input/Output shape: [batch, seq_len, d_model]  (d_model = 2048 với Qwen3-1.7B)
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Phương án B — MLP Bottleneck
# ---------------------------------------------------------------------------

class MLPConnectLayer(nn.Module):
    """
    Non-linear projection với bottleneck:
        h_j → Linear(d, d_inner) → SiLU → Linear(d_inner, d) → RMSNorm

    Params (d=2048, d_inner=512): ~2.1M
    Dùng cho Phase 0 vì đơn giản, dễ debug, gradient ổn định.
    """

    def __init__(self, d_model: int = 2048, bottleneck_ratio: float = 0.25):
        super().__init__()
        d_inner = max(64, int(d_model * bottleneck_ratio))

        self.down_proj = nn.Linear(d_model, d_inner, bias=False)
        self.up_proj   = nn.Linear(d_inner, d_model, bias=False)
        self.act       = nn.SiLU()
        self.norm      = nn.RMSNorm(d_model, eps=1e-6)

        # Khởi tạo gần identity để tránh training collapse ban đầu
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.up_proj.weight)   # out = norm(x + 0) ≈ norm(x) lúc đầu

    def forward(self, h_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_j: [B, S, d_model] — output của loop block (layer j hidden state)
        Returns:
            h_i_hat: [B, S, d_model] — remapped để feed vào loop block input
        """
        projected = self.up_proj(self.act(self.down_proj(h_j)))
        return self.norm(h_j + projected)   # residual để giữ thông tin gốc


# ---------------------------------------------------------------------------
# Phương án C — Gated Residual
# ---------------------------------------------------------------------------

class GatedResidualConnectLayer(nn.Module):
    """
    Gated combination giữa transformed h_j và h_i từ iteration trước:
        gate      = sigmoid(W_g @ h_j)
        transform = MLP(h_j)
        output    = gate * transform + (1 - gate) * h_i_prev

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

        self.norm = nn.RMSNorm(d_model, eps=1e-6)

        # Init: transform gần zero ban đầu, gate gần 0.5 (neutral)
        nn.init.normal_(self.transform[0].weight, std=0.02)
        nn.init.zeros_(self.transform[2].weight)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)   # sigmoid(0) = 0.5

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

        blended = gate * transformed + (1.0 - gate) * h_i_prev
        return self.norm(blended)

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
