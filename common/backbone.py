"""
common/backbone.py — Tiện ích load và unpack pretrained Qwen3/Qwen2 model.

Trả về BackboneComponents: các thành phần cần thiết để build looped model,
tách biệt khỏi logic cụ thể của từng phase.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


@dataclass
class BackboneComponents:
    """
    Chứa tất cả thành phần đã được unpack từ một pretrained causal LM.
    Frozen hoàn toàn — không phase nào được unfreeze trực tiếp mà không
    wrap thêm (e.g., LoRA adapter).
    """
    layers:         nn.ModuleList   # tất cả decoder layers
    embed_tokens:   nn.Embedding    # input embedding
    rotary_emb:     nn.Module       # RoPE module
    norm:           nn.Module       # final RMSNorm
    lm_head:        nn.Linear       # vocab projection
    config:         object          # HuggingFace config object
    d_model:        int             # hidden_size
    num_layers:     int             # tổng số layers
    # API detection
    rotary_uses_position_ids: bool  # True nếu rotary_emb nhận position_ids


def load_backbone(
    model_name_or_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    **kwargs,
) -> BackboneComponents:
    """
    Load một pretrained Qwen3/Qwen2 ForCausalLM và unpack thành
    BackboneComponents. Freeze toàn bộ params.

    Args:
        model_name_or_path: HuggingFace model id hoặc local path
        torch_dtype:        dtype để load (bfloat16 recommended)
        device_map:         "auto" | "cpu" | "cuda:0" | ...

    Returns:
        BackboneComponents với toàn bộ params frozen

    Example:
        bb = load_backbone("Qwen/Qwen3-1.7B")
        # bb.layers[8] → DecoderLayer
        # bb.d_model   → 2048
    """
    print(f"[backbone] Loading: {model_name_or_path} ({torch_dtype}) ...")
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
        **kwargs,
    )
    base.eval()

    # Freeze toàn bộ
    for p in base.parameters():
        p.requires_grad_(False)

    backbone = base.model   # Qwen2Model / Qwen3Model

    # Detect rotary embedding API
    rotary_sig = inspect.signature(backbone.rotary_emb.forward).parameters
    rotary_uses_position_ids = "position_ids" in rotary_sig

    components = BackboneComponents(
        layers                   = backbone.layers,
        embed_tokens             = backbone.embed_tokens,
        rotary_emb               = backbone.rotary_emb,
        norm                     = backbone.norm,
        lm_head                  = base.lm_head,
        config                   = backbone.config,
        d_model                  = backbone.config.hidden_size,
        num_layers               = len(backbone.layers),
        rotary_uses_position_ids = rotary_uses_position_ids,
    )

    n_params = sum(p.numel() for p in base.parameters())
    print(f"[backbone] Loaded: {components.num_layers} layers, "
          f"d_model={components.d_model}, "
          f"params={n_params:,} (all frozen)")
    return components
