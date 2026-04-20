"""
phase1/src/model.py — Phase 1: Dynamic loop với exit gate

Builds on Phase 0:
  - Dùng GatedResidualConnectLayer (per Phase 0 report recommendation)
  - Thêm ExitGate để quyết định dynamic số iterations
  - Training: soft gate + ponder loss
  - Inference: hard gate (true skip)

STATUS: 🚧 SKELETON — chờ Phase 0 checkpoint để khởi tạo connect weights

TODO:
  [ ] Implement soft-gate training loop với L_task + β*L_ponder + λ*L_consistency
  [ ] Load Phase 0 connect weights làm warm start
  [ ] Attention entropy signal từ full attention layer 15
  [ ] Inference mode: hard gate với true skip
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from common.backbone import BackboneComponents, load_backbone
from common.connect_layer import GatedResidualConnectLayer
from .exit_gate import ExitGate, ExitGateConfig


@dataclass
class Phase1Config:
    """Config cho Phase 1 — dynamic exit gate."""

    # Base model
    model_name:   str   = "Qwen/Qwen3-1.7B"

    # Loop block (kế thừa từ Phase 0)
    loop_start:   int   = 8
    loop_end:     int   = 20

    # Max iterations (upper bound khi gate không exit)
    max_iter:     int   = 8

    # Connect layer — Phase 1 dùng gated (per Phase 0 report)
    bottleneck_ratio: float = 0.25

    # Phase 0 checkpoint để warm-start connect layer
    phase0_checkpoint: Optional[str] = None

    # Exit gate
    gate_threshold:    float = 0.5
    ponder_beta:       float = 0.01   # weight của ponder loss
    consistency_lambda: float = 0.1  # weight của consistency loss

    # Truncated BPTT
    k_bptt: Optional[int] = 2

    def validate(self, num_layers: int):
        assert 0 <= self.loop_start < self.loop_end <= num_layers


class Phase1Model(nn.Module):
    """
    Phase 1: Frozen backbone + trainable connect (GatedResidual) + trainable exit gate.

    Training loss:
        L = L_lm + β * L_ponder + λ * L_consistency

    Inference:
        Loop stops when exit_gate(h_end, delta, iter_idx) > threshold
        (hard gate — true computational skip)

    🚧 SKELETON — core loop logic not yet implemented.
    """

    def __init__(self, bb: BackboneComponents, cfg: Phase1Config):
        super().__init__()
        cfg.validate(bb.num_layers)
        self.cfg = cfg
        self.bb  = bb

        # Connect layer (trainable) — warm-started từ Phase 0 nếu có checkpoint
        self.connect = GatedResidualConnectLayer(bb.d_model, cfg.bottleneck_ratio)

        # Exit gate (trainable)
        gate_cfg = ExitGateConfig(
            d_model     = bb.d_model,
            max_iter    = cfg.max_iter,
            mode        = "soft",         # training mode
            threshold   = cfg.gate_threshold,
            ponder_beta = cfg.ponder_beta,
        )
        self.exit_gate = ExitGate(gate_cfg)

        # Layer segments
        self._prefix_layers = bb.layers[: cfg.loop_start]
        self._loop_layers   = bb.layers[cfg.loop_start : cfg.loop_end]
        self._suffix_layers = bb.layers[cfg.loop_end :]

    def trainable_parameters(self):
        return list(self.connect.parameters()) + list(self.exit_gate.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def print_summary(self):
        total = sum(p.numel() for p in self.parameters())
        train = self.trainable_param_count()
        connect_p = sum(p.numel() for p in self.connect.parameters())
        gate_p    = sum(p.numel() for p in self.exit_gate.parameters())
        print(f"  Total params      : {total:,}")
        print(f"  Trainable params  : {train:,}  ({100*train/total:.3f}%)")
        print(f"    ├─ connect layer : {connect_p:,}")
        print(f"    └─ exit gate     : {gate_p:,}")
        print(f"  Loop block        : layers [{self.cfg.loop_start}, {self.cfg.loop_end})")
        print(f"  Max iterations    : {self.cfg.max_iter}")

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        """
        🚧 SKELETON — to be implemented in Phase 1.

        Planned forward flow:
          1. embed + RoPE + causal mask  (same as Phase 0)
          2. Prefix (frozen)
          3. Loop:
             for i in range(max_iter):
                 h_out = loop_block(h_in)
                 h_in  = connect(h_out, h_prev)
                 delta = h_in - h_prev
                 gate_prob = exit_gate(h_out, delta, i)
                 gate_probs.append(gate_prob)
                 if inference and gate_prob > threshold: break
                 if training: output += gate_prob * lm_head(suffix(h_in))
          4. Suffix + norm + lm_head
          5. L = L_lm + β * ponder_loss(gate_probs) + λ * consistency_loss(...)
        """
        raise NotImplementedError(
            "Phase1Model.forward() is a skeleton. "
            "Implement after Phase 0 is fully validated."
        )

    @classmethod
    def from_pretrained(
        cls,
        cfg: Phase1Config,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ) -> "Phase1Model":
        bb       = load_backbone(cfg.model_name, torch_dtype, device_map)
        instance = cls(bb, cfg)
        device   = next(bb.lm_head.parameters()).device
        instance.connect    = instance.connect.to(device=device, dtype=torch_dtype)
        instance.exit_gate  = instance.exit_gate.to(device=device, dtype=torch_dtype)

        # Warm-start connect từ Phase 0 checkpoint
        if cfg.phase0_checkpoint:
            instance.connect.load_state_dict(
                torch.load(cfg.phase0_checkpoint, map_location=device)
            )
            print(f"[phase1] Connect warm-started from: {cfg.phase0_checkpoint}")

        return instance
