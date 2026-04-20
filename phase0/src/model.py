"""
phase0/src/model.py — LoopedLM Phase 0

Fixed n_iter, không có exit gate.
Loop block frozen, chỉ connect layer được train.

Bugs fixed từ report của Bảo (2026-04-20):
  - Bug1: layer_out isinstance check (transformers 5.x returns Tensor not tuple)
  - Bug2: rotary embedding API detection via inspect.signature
  - Bug3: removed torch.no_grad() từ suffix để gradient reach connect layer
  - Bug4: causal mask dtype match với hidden_states.dtype

Bugs fixed (2026-04-21):
  - Bug5: n_iter=0 skipped loop block (layers 8-19), giving PPL ~10k on a
          mutilated 16-layer model instead of the full 28-layer baseline.
          Fix: n_iter=0 now runs all layers unchanged — true model baseline.
  - Bug6: backbone components not registered as nn.Modules, so total_param_count()
          only returned ~2M (connect only) instead of ~2B, and model.to(device)
          silently skipped backbone. Fix: register layers via nn.ModuleList.
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

    model_name:       str            = "Qwen/Qwen3-1.7B"
    loop_start:       int            = 8
    loop_end:         int            = 20
    n_iter:           int            = 3
    connect_type:     str            = "gated"   # "mlp" | "gated"
    bottleneck_ratio: float          = 0.25
    k_bptt:           Optional[int]  = 2         # None = full unroll

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

    n_iter=0  → true baseline: run all 28 layers normally, no connect.
    n_iter>=1 → looped: prefix → (loop block → connect) × n_iter → suffix.

    Trainable params: connect layer only (~2M MLP / ~6M Gated).
    """

    def __init__(self, bb: BackboneComponents, cfg: Phase0Config):
        super().__init__()
        cfg.validate(bb.num_layers)
        self.cfg = cfg

        # ── Bug6 fix: register backbone as nn.ModuleList so that ────────────
        # model.parameters(), model.to(device), model.state_dict() all work.
        # layers are frozen (requires_grad=False); registering them here does
        # NOT make them trainable — it just makes them visible to PyTorch.
        self.prefix_layers = nn.ModuleList(list(bb.layers[: cfg.loop_start]))
        self.loop_layers   = nn.ModuleList(list(bb.layers[cfg.loop_start : cfg.loop_end]))
        self.suffix_layers = nn.ModuleList(list(bb.layers[cfg.loop_end :]))

        # Remaining backbone components
        self.embed_tokens = bb.embed_tokens
        self.rotary_emb   = bb.rotary_emb
        self.norm         = bb.norm
        self.lm_head      = bb.lm_head

        # Stash metadata
        self._d_model                 = bb.d_model
        self._rotary_uses_position_ids = bb.rotary_uses_position_ids
        self._backbone_config         = bb.config

        # Connect layer (ONLY trainable component)
        if cfg.connect_type == "gated":
            self.connect = GatedResidualConnectLayer(bb.d_model, cfg.bottleneck_ratio)
        else:
            self.connect = MLPConnectLayer(bb.d_model, cfg.bottleneck_ratio)

    # -----------------------------------------------------------------------
    # Layer runner
    # -----------------------------------------------------------------------

    def _run_layers(
        self,
        hidden_states:       torch.Tensor,
        layers:              nn.ModuleList,
        position_embeddings,
        causal_mask:         torch.Tensor,
        position_ids:        torch.Tensor,
        cache_position:      torch.Tensor,
    ) -> torch.Tensor:
        for layer in layers:
            out = layer(
                hidden_states,
                attention_mask      = causal_mask,
                position_ids        = position_ids,
                past_key_value      = None,
                output_attentions   = False,
                use_cache           = False,
                cache_position      = cache_position,
                position_embeddings = position_embeddings,
            )
            # transformers ≥5.x: returns Tensor; <5.x: returns tuple
            hidden_states = out[0] if isinstance(out, tuple) else out
        return hidden_states

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels:         Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        device = input_ids.device
        B, S   = input_ids.shape

        # 1. Embed
        h = self.embed_tokens(input_ids)                    # [B, S, d]

        # 2. Position ids + RoPE
        position_ids   = torch.arange(S, device=device).unsqueeze(0).expand(B, -1).contiguous()
        cache_position = torch.arange(S, device=device)

        if self._rotary_uses_position_ids:
            pos_emb = self.rotary_emb(h, position_ids)
        else:
            pos_emb = self.rotary_emb(h, seq_len=S)

        # 3. Causal mask [B, 1, S, S]
        causal_mask = self._make_causal_mask(B, S, device, h.dtype)
        if attention_mask is not None:
            pad = (1.0 - attention_mask[:, None, None, :].to(h.dtype)) \
                  * torch.finfo(h.dtype).min
            causal_mask = causal_mask + pad

        # ── Bug5 fix ──────────────────────────────────────────────────────
        # n_iter=0: TRUE BASELINE — run all 28 layers, identical to HF model.
        # Previous code skipped the loop block entirely, producing PPL ~10k
        # on a mutilated 16-layer forward pass.
        if self.cfg.n_iter == 0:
            with torch.no_grad():
                h = self._run_layers(h, self.prefix_layers,
                                     pos_emb, causal_mask, position_ids, cache_position)
                h = self._run_layers(h, self.loop_layers,
                                     pos_emb, causal_mask, position_ids, cache_position)
                h = self._run_layers(h, self.suffix_layers,
                                     pos_emb, causal_mask, position_ids, cache_position)
                h = self.norm(h)
            logits = self.lm_head(h)

            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss(ignore_index=-100)(
                    logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                    labels[..., 1:].contiguous().view(-1),
                )
            return CausalLMOutputWithPast(loss=loss, logits=logits)
        # ─────────────────────────────────────────────────────────────────

        # 4. Prefix (frozen, no grad needed — prefix params don't train)
        with torch.no_grad():
            h = self._run_layers(h, self.prefix_layers,
                                 pos_emb, causal_mask, position_ids, cache_position)

        # 5. Loop block × n_iter + connect layer (connect is trainable)
        h_prev = h
        for i in range(self.cfg.n_iter):
            # Truncated BPTT: detach h entering early iterations so gradient
            # doesn't flow through loop_block more than k_bptt times.
            # (frozen loop_block params don't accumulate grad regardless,
            #  but this saves activation memory for connect's gradient path)
            if (self.cfg.k_bptt is not None
                    and i < self.cfg.n_iter - self.cfg.k_bptt):
                h = h.detach()

            # Loop block: frozen weights, activations stay in autograd graph
            # so gradient can flow back to connect layer
            h_out = self._run_layers(h, self.loop_layers,
                                     pos_emb, causal_mask, position_ids, cache_position)

            # Connect layer (trainable)
            if isinstance(self.connect, GatedResidualConnectLayer):
                h = self.connect(h_out, h_prev)
            else:
                h = self.connect(h_out)

            h_prev = h

        # 6. Suffix: no torch.no_grad() — gradient must reach connect layer
        h = self._run_layers(h, self.suffix_layers,
                             pos_emb, causal_mask, position_ids, cache_position)
        h = self.norm(h)

        # 7. LM Head (frozen weights, activations flow grad back to connect)
        logits = self.lm_head(h)                            # [B, S, vocab]

        # 8. Loss
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[..., 1:].contiguous().view(-1),
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_causal_mask(B, S, device, dtype) -> torch.Tensor:
        mask = torch.full((S, S), torch.finfo(dtype).min, device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None].expand(B, 1, S, S).contiguous()

    def train(self, mode: bool = True):
        """
        Override train() để backbone luôn ở eval mode.
        Backbone có attention_dropout=0.0 nhưng một số layer có thể dùng dropout.
        Frozen params không train ước nhưng hành vi dropout vẫn bị ảnh hưởng.
        """
        super().train(mode)
        # Backbone luôn eval — frozen weights + deterministic behavior
        for module in [self.prefix_layers, self.loop_layers, self.suffix_layers,
                       self.embed_tokens, self.rotary_emb, self.norm, self.lm_head]:
            module.eval()
        # Chỉ connect layer theo mode được yêu cầu
        self.connect.train(mode)
        return self

    def trainable_parameters(self):
        return list(self.connect.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def frozen_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    def print_summary(self):
        total   = self.total_param_count()
        train   = self.trainable_param_count()
        frozen  = self.frozen_param_count()
        print(f"  Total params    : {total:,}")
        print(f"  Trainable params: {train:,}  ({100*train/total:.3f}%)")
        print(f"  Frozen params   : {frozen:,}")
        print(f"  Connect type    : {self.cfg.connect_type}")
        print(f"  Loop block      : layers [{self.cfg.loop_start}, {self.cfg.loop_end})")
        print(f"  n_iter          : {self.cfg.n_iter}")
        print(f"  n_iter=0        : runs all {len(self.prefix_layers)+len(self.loop_layers)+len(self.suffix_layers)} layers (true baseline)")

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

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
