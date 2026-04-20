"""
phase0/src/model.py — LoopedLM Phase 0

Fixed n_iter, không có exit gate.
Loop block frozen, chỉ connect layer được train.

Bugs fixed từ report của Bảo (2026-04-20):
  - Bug1: layer_out isinstance check (transformers 5.x returns Tensor not tuple)
  - Bug2: rotary embedding API detection via inspect.signature
  - Bug3: removed torch.no_grad() từ suffix để gradient reach connect layer
  - Bug4: causal mask dtype match với hidden_states.dtype
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from common.backbone import BackboneComponents, load_backbone
from common.connect_layer import MLPConnectLayer, GatedResidualConnectLayer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Phase0Config:
    """Config cho Phase 0 — fixed-iteration loop, no exit gate."""

    # Base model
    model_name: str = "Qwen/Qwen3-1.7B"

    # Loop block boundaries [loop_start, loop_end)
    loop_start: int = 8
    loop_end:   int = 20

    # Fixed iterations (Phase 0: không có dynamic exit)
    n_iter: int = 3

    # Connect layer: "mlp" | "gated"
    connect_type: str = "gated"   # default gated per Phase 0 report

    # Bottleneck: d_inner = d_model * bottleneck_ratio
    bottleneck_ratio: float = 0.25

    # Truncated BPTT depth (None = full unroll)
    k_bptt: Optional[int] = 2

    def validate(self, num_layers: int):
        assert 0 <= self.loop_start < self.loop_end <= num_layers, (
            f"Invalid loop range [{self.loop_start}, {self.loop_end}) "
            f"for model with {num_layers} layers"
        )


# ---------------------------------------------------------------------------
# Phase0Model
# ---------------------------------------------------------------------------

class Phase0Model(nn.Module):
    """
    Phase 0: Frozen backbone + trainable connect layer.

    Forward:
        input_ids → embed → prefix → [loop block × n_iter ↔ connect] → suffix → lm_head → logits

    Trainable params: connect layer only (~2M for MLP, ~6M for Gated)
    """

    def __init__(self, bb: BackboneComponents, cfg: Phase0Config):
        super().__init__()
        cfg.validate(bb.num_layers)
        self.cfg = cfg
        self.bb  = bb

        # Connect layer (trainable)
        if cfg.connect_type == "gated":
            self.connect = GatedResidualConnectLayer(bb.d_model, cfg.bottleneck_ratio)
        else:
            self.connect = MLPConnectLayer(bb.d_model, cfg.bottleneck_ratio)

        # Layer segments (references, không register lại params frozen)
        self._prefix_layers = bb.layers[: cfg.loop_start]
        self._loop_layers   = bb.layers[cfg.loop_start : cfg.loop_end]
        self._suffix_layers = bb.layers[cfg.loop_end :]

    # ------------------------------------------------------------------
    # Layer runner
    # ------------------------------------------------------------------

    def _run_layers(
        self,
        hidden_states: torch.Tensor,
        layers,
        position_embeddings,
        causal_mask: torch.Tensor,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        for layer in layers:
            out = layer(
                hidden_states,
                attention_mask     = causal_mask,
                position_ids       = position_ids,
                past_key_value     = None,
                output_attentions  = False,
                use_cache          = False,
                cache_position     = cache_position,
                position_embeddings= position_embeddings,
            )
            # transformers 5.x: returns Tensor; <5.x: returns tuple
            hidden_states = out[0] if isinstance(out, tuple) else out
        return hidden_states

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        device = input_ids.device
        B, S   = input_ids.shape
        bb     = self.bb

        # 1. Embed
        h = bb.embed_tokens(input_ids)                              # [B, S, d]

        # 2. Position ids + RoPE
        position_ids     = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        cache_position   = torch.arange(S, device=device)

        if bb.rotary_uses_position_ids:
            pos_emb = bb.rotary_emb(h, position_ids)
        else:
            pos_emb = bb.rotary_emb(h, seq_len=S)

        # 3. Causal mask [B, 1, S, S] — dtype matches hidden_states
        causal_mask = self._make_causal_mask(B, S, device, h.dtype)
        if attention_mask is not None:
            pad = (1.0 - attention_mask[:, None, None, :].to(h.dtype)) \
                  * torch.finfo(h.dtype).min
            causal_mask = causal_mask + pad

        # 4. Prefix (frozen — no grad needed here)
        with torch.no_grad():
            h = self._run_layers(h, self._prefix_layers,
                                 pos_emb, causal_mask, position_ids, cache_position)

        # 5. Loop block × n_iter  +  connect layer
        h_prev = h
        for i in range(self.cfg.n_iter):
            # Truncated BPTT: detach early iterations
            if (self.cfg.k_bptt is not None
                    and i < self.cfg.n_iter - self.cfg.k_bptt):
                h = h.detach()

            # Loop block — frozen weights, activations in autograd graph
            # (no torch.no_grad() here — gradient must reach connect layer)
            h_out = self._run_layers(h, self._loop_layers,
                                     pos_emb, causal_mask, position_ids, cache_position)

            # Connect layer (trainable)
            if isinstance(self.connect, GatedResidualConnectLayer):
                h = self.connect(h_out, h_prev)
            else:
                h = self.connect(h_out)

            h_prev = h

        # 6. Suffix — no torch.no_grad() (gradient path back to connect)
        h = self._run_layers(h, self._suffix_layers,
                             pos_emb, causal_mask, position_ids, cache_position)

        # 7. Final norm + LM head (frozen weights, activations flow grad)
        h      = bb.norm(h)
        logits = bb.lm_head(h)                                      # [B, S, vocab]

        # 8. Loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _make_causal_mask(B, S, device, dtype) -> torch.Tensor:
        mask = torch.full((S, S), torch.finfo(dtype).min, device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None].expand(B, 1, S, S)

    def trainable_parameters(self):
        return list(self.connect.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def print_summary(self):
        total = self.total_param_count()
        train = self.trainable_param_count()
        print(f"  Total params    : {total:,}")
        print(f"  Trainable params: {train:,}  ({100*train/total:.3f}%)")
        print(f"  Frozen params   : {total - train:,}")
        print(f"  Connect type    : {self.cfg.connect_type}")
        print(f"  Loop block      : layers [{self.cfg.loop_start}, {self.cfg.loop_end})")
        print(f"  n_iter          : {self.cfg.n_iter}")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        cfg: Phase0Config,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ) -> "Phase0Model":
        bb       = load_backbone(cfg.model_name, torch_dtype, device_map)
        instance = cls(bb, cfg)
        device   = next(bb.lm_head.parameters()).device
        instance.connect = instance.connect.to(device=device, dtype=torch_dtype)
        return instance

    def save_connect(self, path: str):
        torch.save(self.connect.state_dict(), path)
        print(f"[phase0] Connect layer saved → {path}")

    def load_connect(self, path: str):
        device = next(self.connect.parameters()).device
        self.connect.load_state_dict(torch.load(path, map_location=device))
        print(f"[phase0] Connect layer loaded ← {path}")
