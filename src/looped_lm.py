"""
LoopedLM — Phase 0 prototype
Wrap một Qwen3ForCausalLM (hoặc Qwen2ForCausalLM) và thêm loop block.

Architecture:
    Prefix  = layers [0, loop_start)         — frozen
    Loop    = layers [loop_start, loop_end)  — frozen, chạy n_iter lần
    Suffix  = layers [loop_end, num_layers)  — frozen
    Connect = MLPConnectLayer hoặc GatedResidualConnectLayer — trainable

Với Qwen3-1.7B / Qwen3.5-2B (d_model=2048, 28 layers):
    Mặc định: loop_start=8, loop_end=20, prefix=8, suffix=8

NOTE về Qwen3-1.7B vs Qwen3.5-2B:
    - Qwen3-1.7B: pure text, num_layers=28, d_model=2048  ← dùng cho Phase 0
    - Qwen3.5-2B: multimodal, phức tạp hơn → Phase 1+
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from .connect_layer import MLPConnectLayer, GatedResidualConnectLayer


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LoopedLMConfig:
    # Loop block boundaries
    loop_start: int = 8
    loop_end:   int = 20   # exclusive — loop block = layers [loop_start, loop_end)

    # Số vòng lặp trong Phase 0 (fixed, không có exit gate)
    n_iter: int = 3

    # Connect layer type: "mlp" (Phương án B) | "gated" (Phương án C)
    connect_type: str = "mlp"

    # Bottleneck ratio cho connect layer
    bottleneck_ratio: float = 0.25

    # Truncated BPTT: chỉ backprop qua k_bptt iterations cuối
    # None = backprop qua tất cả (đắt hơn)
    k_bptt: Optional[int] = 2

    # Nếu True: detach hidden state giữa mỗi iteration (rất stable, kém quality)
    full_detach: bool = False


# ---------------------------------------------------------------------------
# LoopedLM
# ---------------------------------------------------------------------------

class LoopedLM(nn.Module):
    """
    Wrap một pretrained Qwen3/Qwen2 model với loop block.

    Sử dụng:
        model = LoopedLM.from_pretrained("Qwen/Qwen3-1.7B", config=LoopedLMConfig())
        outputs = model(input_ids=..., labels=...)
        loss = outputs.loss
        loss.backward()   # chỉ connect layer nhận gradient
    """

    def __init__(self, base_model: nn.Module, cfg: LoopedLMConfig):
        super().__init__()
        self.cfg = cfg

        # Lấy inner Qwen2Model (backbone không có LM head)
        # Qwen3ForCausalLM.model trả về Qwen2Model
        self.backbone = base_model.model
        self.lm_head  = base_model.lm_head

        # Verify config
        layers = self.backbone.layers
        n = len(layers)
        assert 0 <= cfg.loop_start < cfg.loop_end <= n, (
            f"loop_start={cfg.loop_start}, loop_end={cfg.loop_end} "
            f"không hợp lệ với {n} layers"
        )

        # Freeze toàn bộ base model
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.lm_head.parameters():
            p.requires_grad_(False)

        # Lấy d_model từ config
        d_model = self.backbone.config.hidden_size

        # Khởi tạo connect layer (trainable)
        if cfg.connect_type == "gated":
            self.connect = GatedResidualConnectLayer(d_model, cfg.bottleneck_ratio)
        else:
            self.connect = MLPConnectLayer(d_model, cfg.bottleneck_ratio)

        # Tách layers thành 3 segment để dễ gọi
        # Dùng nn.ModuleList để không re-register params frozen
        self._prefix_layers     = layers[: cfg.loop_start]
        self._loop_layers       = layers[cfg.loop_start : cfg.loop_end]
        self._suffix_layers     = layers[cfg.loop_end :]

        # RotaryEmbedding và norm của backbone
        self._embed_tokens      = self.backbone.embed_tokens
        self._rotary_emb        = self.backbone.rotary_emb
        self._norm              = self.backbone.norm

    # -----------------------------------------------------------------------
    # Core forward helpers
    # -----------------------------------------------------------------------

    def _run_layers(
        self,
        hidden_states: torch.Tensor,
        layers,
        position_embeddings,
        attention_mask,
        position_ids,
    ) -> torch.Tensor:
        """
        Chạy một danh sách Qwen2/Qwen3 DecoderLayer tuần tự.

        Qwen2DecoderLayer.forward signature:
            hidden_states, attention_mask, position_ids, past_key_value,
            output_attentions, use_cache, cache_position, position_embeddings
        Returns: tuple (hidden_states, ...)
        """
        # cache_position: [0, 1, ..., S-1] — dùng cho static cache / causal mask
        seq_len = hidden_states.shape[1]
        cache_position = torch.arange(seq_len, device=hidden_states.device)

        for layer in layers:
            layer_out = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            # DecoderLayer trả về tuple; element đầu luôn là hidden_states
            hidden_states = layer_out[0]
        return hidden_states

    def _run_loop_block(
        self,
        h_in: torch.Tensor,
        position_embeddings,
        attention_mask,
        position_ids,
    ) -> torch.Tensor:
        """Một lần chạy qua loop block (frozen)."""
        return self._run_layers(
            h_in,
            self._loop_layers,
            position_embeddings,
            attention_mask,
            position_ids,
        )

    # -----------------------------------------------------------------------
    # Main forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        """
        Args:
            input_ids:      [B, S]
            attention_mask: [B, S] (optional)
            labels:         [B, S] (optional, -100 để ignore)
        Returns:
            CausalLMOutputWithPast với .loss (nếu labels được cung cấp)
        """
        device = input_ids.device
        B, S   = input_ids.shape

        # 1. Embedding
        hidden_states = self._embed_tokens(input_ids)   # [B, S, d]

        # 2. Position IDs
        position_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)  # [B, S]

        # 3. Rotary embeddings — truyền position_ids
        position_embeddings = self._rotary_emb(hidden_states, position_ids)  # (cos, sin)

        # 4. Causal attention mask [B, 1, S, S] với -inf ở upper triangle
        # HF transformers 4.44+ dùng float mask (không phải bool)
        causal_mask = self._make_causal_mask(B, S, device, hidden_states.dtype)
        if attention_mask is not None:
            # attention_mask: 1=keep, 0=mask (padding)
            # Convert sang additive mask
            pad_mask = attention_mask[:, None, None, :].to(hidden_states.dtype)   # [B,1,1,S]
            pad_mask = (1.0 - pad_mask) * torch.finfo(hidden_states.dtype).min
            causal_mask = causal_mask + pad_mask

        # 4. Prefix (frozen)
        with torch.no_grad():
            h = self._run_layers(
                hidden_states, self._prefix_layers,
                position_embeddings, causal_mask, position_ids,
            )

        # 5. Loop block — n_iter lần, connect layer trainable
        h_loop_in = h.clone()       # h_8^(0) — sẽ dùng cho gated residual
        h_prev    = h_loop_in       # reference cho gated connect

        for i in range(self.cfg.n_iter):
            # Detach để implement Truncated BPTT
            should_detach = (
                self.cfg.k_bptt is not None
                and i < self.cfg.n_iter - self.cfg.k_bptt
            )
            if should_detach or self.cfg.full_detach:
                h_loop_in = h_loop_in.detach()

            # Loop block (frozen)
            with torch.no_grad():
                h_loop_out = self._run_loop_block(
                    h_loop_in, position_embeddings, causal_mask, position_ids,
                )

            # Connect layer (trainable) — remap h_loop_out → h_loop_in space
            if isinstance(self.connect, GatedResidualConnectLayer):
                h_loop_in = self.connect(h_loop_out, h_prev)
            else:
                h_loop_in = self.connect(h_loop_out)

            h_prev = h_loop_in

        # 6. Suffix (frozen)
        with torch.no_grad():
            h = self._run_layers(
                h_loop_in, self._suffix_layers,
                position_embeddings, causal_mask, position_ids,
            )

            # Final norm
            h = self._norm(h)

        # 7. LM Head (frozen)
        logits = self.lm_head(h)   # [B, S, vocab]

        # 8. Compute loss nếu có labels
        loss = None
        if labels is not None:
            # Shift: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_causal_mask(
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Tạo causal mask [B, 1, S, S] với -inf ở upper triangle."""
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None, :, :].expand(batch_size, 1, -1, -1)

    def trainable_parameters(self):
        """Trả về chỉ các parameters cần train (connect layer)."""
        return list(self.connect.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters())

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def print_param_summary(self):
        total     = self.total_param_count()
        trainable = self.trainable_param_count()
        print(f"Total params    : {total:,}")
        print(f"Trainable params: {trainable:,}  ({100*trainable/total:.3f}%)")
        print(f"Frozen params   : {total - trainable:,}")
        print(f"Connect type    : {self.cfg.connect_type}")
        print(f"Loop block      : layers [{self.cfg.loop_start}, {self.cfg.loop_end})")
        print(f"n_iter          : {self.cfg.n_iter}")

    # -----------------------------------------------------------------------
    # Factory
    # -----------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        cfg: Optional[LoopedLMConfig] = None,
        torch_dtype=torch.bfloat16,
        device_map: str = "auto",
        **kwargs,
    ) -> "LoopedLM":
        """
        Load pretrained model và wrap thành LoopedLM.

        Ví dụ:
            model = LoopedLM.from_pretrained(
                "Qwen/Qwen3-1.7B",
                cfg=LoopedLMConfig(n_iter=3, connect_type="mlp"),
            )
        """
        if cfg is None:
            cfg = LoopedLMConfig()

        print(f"Loading base model: {model_name_or_path} ...")
        base = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            **kwargs,
        )
        base.eval()

        instance = cls(base, cfg)

        # Move connect layer to same device as model
        device = next(base.parameters()).device
        instance.connect = instance.connect.to(device=device, dtype=torch_dtype)

        return instance
