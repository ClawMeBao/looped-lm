"""
phase2/src/model.py — 🚧 SKELETON

Phase 2: Full pipeline
  - Iteration-aware connect (GatedResidual + IterEmbedding)
  - Exit gate với attention entropy signal
  - Linear attention state carry (Qwen3.5 specific)
  - LoRA unfreeze trên loop block (~1%)

Depends on:
  - Phase 0 ✅ (connect layer validated)
  - Phase 1 ⬜ (exit gate validated)

TODO:
  [ ] IterationAwareConnectLayer = GatedResidual + iter_emb
  [ ] Carry linear attention SSM state qua iterations
  [ ] LoRA adapter trên loop block layers
  [ ] Attention entropy extraction từ full attention layers
  [ ] Benchmark trên GSM8K, ARC, BoolQ
"""

raise NotImplementedError("Phase 2 — not yet implemented. Depends on Phase 1.")
