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
from common.connect_layer import (
    ResidualConnectLayer,
    MLPConnectLayer,
    GatedResidualConnectLayer,
    IterationAwareConnectLayer,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Phase0Config:
    """Config cho Phase 0 — fixed-iteration loop, no exit gate."""

    model_name:           str            = "Qwen/Qwen3-1.7B"
    loop_start:           int            = 8
    loop_end:             int            = 20
    n_iter:               int            = 4
    connect_type:         str            = "residual"   # "residual" | "mlp" | "gated" | "iter_aware"
    bottleneck_ratio:     float          = 0.25
    max_iter_emb:         int            = 8
    k_bptt:               Optional[int]  = 2         # None = full unroll
    # Auxiliary multi-step loss (from Ouro paper Section 3.1)
    aux_loss_weight:      float          = 0.0   # weight of summed per-iter losses
    aux_loss_gamma:       float          = 0.5   # geometric decay: earlier iters weighted less
    # Optional convergence regularizer
    consistency_weight:   float          = 0.0   # L2 ||h_out_i - h_out_{i-1}||² / d

    def validate(self, num_layers: int):
        valid_connect = {"residual", "mlp", "gated", "iter_aware"}
        assert 0 <= self.loop_start < self.loop_end <= num_layers, (
            f"Invalid loop range [{self.loop_start}, {self.loop_end}) "
            f"for model with {num_layers} layers"
        )
        if self.connect_type not in valid_connect:
            raise ValueError(f"Invalid connect_type={self.connect_type!r}; expected one of {sorted(valid_connect)}")
        if self.n_iter < 0:
            raise ValueError("n_iter must be >= 0")
        if self.bottleneck_ratio <= 0:
            raise ValueError("bottleneck_ratio must be > 0")
        if self.k_bptt is not None and self.k_bptt <= 0:
            raise ValueError("k_bptt must be None or > 0")
        if self.max_iter_emb <= 0:
            raise ValueError("max_iter_emb must be > 0")


# ---------------------------------------------------------------------------
# Phase0Model
# ---------------------------------------------------------------------------

class Phase0Model(nn.Module):
    """
    Phase 0: Frozen backbone + trainable connect layer.

    n_iter=0  → TRUE BASELINE: run all 28 layers once, identical to HF model.
    n_iter=1  → single loop pass via loop code path (no connect called, PPL = baseline).
    n_iter>=2 → prefix → loop → (connect → loop) × (n_iter-1) → suffix.

    Note: n_iter=1 and n_iter=0 produce identical output (both = 28 layers, no connect).
    n_iter=1 is kept for completeness but is semantically equivalent to baseline.

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
        self._last_loss_metrics: dict[str, float] = {}
        self._last_connect_metrics: dict[str, float] = {}

        # Connect layer (ONLY trainable component)
        if cfg.connect_type == "residual":
            self.connect = ResidualConnectLayer(bb.d_model, cfg.bottleneck_ratio)
        elif cfg.connect_type == "iter_aware":
            max_iter = max(cfg.max_iter_emb, cfg.n_iter)
            self.connect = IterationAwareConnectLayer(bb.d_model, cfg.bottleneck_ratio, max_iter=max_iter)
        elif cfg.connect_type == "gated":
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
            # Multi-GPU: move all inputs to this layer's device if needed.
            # device_map="auto" can split layers across GPUs; manual forward
            # must handle cross-device moves explicitly.
            dev = self._dev(layer)
            if hidden_states.device != dev:
                hidden_states  = hidden_states.to(dev)
                causal_mask    = causal_mask.to(dev)
                position_ids   = position_ids.to(dev)
                cache_position = cache_position.to(dev)
                if isinstance(position_embeddings, (tuple, list)):
                    position_embeddings = tuple(
                        t.to(dev) if isinstance(t, torch.Tensor) else t
                        for t in position_embeddings
                    )
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

        # 1. Embed — move input_ids to embed_tokens' device (may differ in multi-GPU)
        embed_dev = self._dev(self.embed_tokens)
        h = self.embed_tokens(input_ids.to(embed_dev))     # [B, S, d]

        # 2. Position ids + RoPE
        position_ids   = torch.arange(S, device=h.device).unsqueeze(0).expand(B, -1).contiguous()
        cache_position = torch.arange(S, device=h.device)

        rotary_dev = self._dev(self.rotary_emb)
        if self._rotary_uses_position_ids:
            pos_emb = self.rotary_emb(h.to(rotary_dev), position_ids.to(rotary_dev))
        else:
            pos_emb = self.rotary_emb(h.to(rotary_dev), seq_len=S)

        # 3. Causal mask [B, 1, S, S] — created on h's device; _run_layers moves per-layer
        causal_mask = self._make_causal_mask(B, S, h.device, h.dtype)
        if attention_mask is not None:
            pad = (1.0 - attention_mask[:, None, None, :].to(h.dtype)) \
                  * torch.finfo(h.dtype).min
            causal_mask = causal_mask + pad.to(causal_mask.device)

        # ── Bug5 fix ──────────────────────────────────────────────────────
        # n_iter=0: TRUE BASELINE — run all 28 layers, identical to HF model.
        # Previous code treated n_iter<=1 as baseline, incorrectly bundling
        # n_iter=1 (single loop pass) with n_iter=0. Now only n_iter=0 takes
        # the fast baseline path; n_iter=1 falls through to the loop code below,
        # producing identical PPL (1 loop pass, no connect called) but correctly
        # reflecting that it runs via the loop architecture.
        if self.cfg.n_iter == 0:
            self._last_connect_metrics = {}
            with torch.no_grad():
                h = self._run_layers(h, self.prefix_layers,
                                     pos_emb, causal_mask, position_ids, cache_position)
                h = self._run_layers(h, self.loop_layers,
                                     pos_emb, causal_mask, position_ids, cache_position)
                h = self._run_layers(h, self.suffix_layers,
                                     pos_emb, causal_mask, position_ids, cache_position)
                h = self.norm(h.to(self._dev(self.norm)))
            logits = self.lm_head(h.to(self._dev(self.lm_head)))

            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss(ignore_index=-100)(
                    logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                    labels[..., 1:].contiguous().view(-1).to(logits.device),
                )
                self._last_loss_metrics = {"lm_loss": float(loss.detach().item())}
            return CausalLMOutputWithPast(loss=loss, logits=logits)
        # ─────────────────────────────────────────────────────────────────

        # 4. Prefix (frozen, no grad needed — prefix params don't train)
        with torch.no_grad():
            h = self._run_layers(h, self.prefix_layers,
                                 pos_emb, causal_mask, position_ids, cache_position)

        # 5. Loop block × n_iter. Connect maps loop output back to loop input
        # only between iterations. The final loop output is already in suffix
        # input space, so do not connect after the last iteration.
        h_prev = h
        aux_losses: list[torch.Tensor] = []   # per-iteration LM losses for auxiliary signal
        h_out_prev: Optional[torch.Tensor] = None
        consistency_losses: list[torch.Tensor] = []
        connect_update_ratios: list[torch.Tensor] = []
        connect_cosines: list[torch.Tensor] = []

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

            # Auxiliary LM loss at each iteration. h_out is layer loop_end-1
            # hidden state, so it must pass through the frozen suffix before
            # lm_head sees it.
            if labels is not None and self.cfg.aux_loss_weight > 0.0:
                h_aux = self._run_layers(h_out, self.suffix_layers,
                                         pos_emb, causal_mask, position_ids, cache_position)
                h_norm_i = self.norm(h_aux.to(self._dev(self.norm)))
                logits_i = self.lm_head(h_norm_i.to(self._dev(self.lm_head)))
                loss_i = nn.CrossEntropyLoss(ignore_index=-100)(
                    logits_i[..., :-1, :].contiguous().view(-1, logits_i.size(-1)),
                    labels[..., 1:].contiguous().view(-1).to(logits_i.device),
                )
                aux_losses.append(loss_i)

            # Consistency loss: encourage hidden states to converge across iters.
            # Uses cosine similarity (bounded [0,2]) instead of raw L2 to avoid
            # exploding loss from outlier features in large transformer hidden states
            # (Qwen3 has extreme outlier dims; raw L2 can reach 20000+ → spikes).
            if labels is not None and self.cfg.consistency_weight > 0.0:
                if h_out_prev is not None:
                    h_flat     = h_out.view(-1, h_out.size(-1))
                    h_prev_flat = h_out_prev.detach().view(-1, h_out_prev.size(-1))
                    diff = 1.0 - torch.nn.functional.cosine_similarity(
                        h_flat, h_prev_flat, dim=-1
                    ).mean()
                    consistency_losses.append(diff)
                h_out_prev = h_out

            if i == self.cfg.n_iter - 1:
                h = h_out
            else:
                # Connect layer (trainable): layer loop_end -> layer loop_start
                # Move to connect's device before calling (multi-GPU safe)
                connect_dev = self._dev(self.connect)
                h_out_c  = h_out.to(connect_dev)
                h_prev_c = h_prev.to(connect_dev)
                if isinstance(self.connect, IterationAwareConnectLayer):
                    h_next = self.connect(h_out_c, h_prev_c, step_idx=i)
                elif isinstance(self.connect, ResidualConnectLayer):
                    h_next = self.connect(h_out_c, h_prev_c)
                elif isinstance(self.connect, GatedResidualConnectLayer):
                    h_next = self.connect(h_out_c, h_prev_c)
                else:
                    h_next = self.connect(h_out_c, h_prev_c)

                # Cheap diagnostics: if update_norm_ratio is near 0, the
                # connect layer is effectively a no-op and extra loop compute
                # may not be buying quality.
                with torch.no_grad():
                    update = h_next - h_prev
                    update_norm = update.float().norm(dim=-1).mean()
                    prev_norm = h_prev.float().norm(dim=-1).mean().clamp_min(1e-12)
                    connect_update_ratios.append(update_norm / prev_norm)
                    connect_cosines.append(
                        torch.nn.functional.cosine_similarity(
                            h_next.float().view(-1, h_next.size(-1)),
                            h_prev.float().view(-1, h_prev.size(-1)),
                            dim=-1,
                        ).mean()
                    )

                h = h_next

                # Bound BPTT depth on the recurrent h_prev chain.
                h_prev = h.detach()

        if connect_update_ratios:
            ratios = torch.stack(connect_update_ratios)
            cosines = torch.stack(connect_cosines)
            self._last_connect_metrics = {
                "update_norm_ratio_mean": float(ratios.mean().item()),
                "update_norm_ratio_max": float(ratios.max().item()),
                "output_cosine_to_prev_mean": float(cosines.mean().item()),
                "output_cosine_to_prev_min": float(cosines.min().item()),
            }
        else:
            self._last_connect_metrics = {}

        # 6. Suffix: no torch.no_grad() — gradient must reach connect layer
        h = self._run_layers(h, self.suffix_layers,
                             pos_emb, causal_mask, position_ids, cache_position)
        h = self.norm(h.to(self._dev(self.norm)))

        # 7. LM Head (frozen weights, activations flow grad back to connect)
        logits = self.lm_head(h.to(self._dev(self.lm_head)))    # [B, S, vocab]

        # 8. Loss
        loss = None
        if labels is not None:
            lm_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits[..., :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[..., 1:].contiguous().view(-1).to(logits.device),
            )
            loss = lm_loss
            aux_total = None
            consistency_total = None

            # Auxiliary multi-step loss: geometric weighting, earlier iters less weight
            # L_total = L_final + aux_weight * sum_i( gamma^(n-1-i) * L_i )
            if aux_losses and self.cfg.aux_loss_weight > 0.0:
                n = len(aux_losses)
                aux_total = sum(
                    (self.cfg.aux_loss_gamma ** (n - 1 - i)) * aux_losses[i]
                    for i in range(n)
                )
                loss = loss + self.cfg.aux_loss_weight * aux_total

            # Consistency loss
            if consistency_losses and self.cfg.consistency_weight > 0.0:
                consistency_total = sum(consistency_losses)
                loss = loss + self.cfg.consistency_weight * consistency_total

            self._last_loss_metrics = {
                "lm_loss": float(lm_loss.detach().item()),
                "total_loss": float(loss.detach().item()),
            }
            if aux_total is not None:
                self._last_loss_metrics["aux_loss"] = float(aux_total.detach().item())
            if consistency_total is not None:
                self._last_loss_metrics["consistency_loss"] = float(consistency_total.detach().item())

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def last_loss_metrics(self) -> dict[str, float]:
        return dict(self._last_loss_metrics)

    def last_connect_metrics(self) -> dict[str, float]:
        return dict(self._last_connect_metrics)

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_causal_mask(B, S, device, dtype) -> torch.Tensor:
        mask = torch.full((S, S), torch.finfo(dtype).min, device=device, dtype=dtype)
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None].expand(B, 1, S, S).contiguous()

    @staticmethod
    def _dev(module: nn.Module) -> torch.device:
        """Return device of the first parameter or buffer in a module."""
        for p in module.parameters():
            return p.device
        for b in module.buffers():
            return b.device
        # Fallback: module has no params (e.g. identity) — return CPU
        return torch.device("cpu")

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
        print(f"  aux_loss_weight : {self.cfg.aux_loss_weight}  (gamma={self.cfg.aux_loss_gamma})")
        print(f"  consistency_w   : {self.cfg.consistency_weight}")
        print(f"  n_iter=0/1      : runs all {len(self.prefix_layers)+len(self.loop_layers)+len(self.suffix_layers)} layers once (true baseline)")

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
