"""Token-to-layer routing for LTX-2 / Gemma conditioning.

Last Updated: 2026-01-16

This module provides per-token layer routing for LLM-DiT systems like LTX-2.
Instead of uniform layer blending, the router learns to dynamically select
which Gemma layers are most relevant for each token in the prompt.

Key insight from projection analysis (Session 27):
- Late layers (43-47) contribute ~25% of signal when accounting for activations
- Early layers (0-4) contribute <1% of signal
- BUT this is averaged across all tokens - different tokens likely benefit
  from different layers (e.g., style words vs object words vs action words)

The router learns this per-token specialization, enabling:
1. Better quality through optimized layer selection
2. Compute savings by dropping low-contribution layers per token
3. Interpretable routing patterns (which layers "understand" which concepts)

Architecture:
    166K parameters total
    - query_proj: [3840 → 64] = 245K params
    - layer_keys: [49 × 64] = 3.1K params

Usage:
    from llm_dit.router import TokenLayerRouter

    router = TokenLayerRouter()
    token_embeds = gemma.get_hidden_states(prompt)  # [B, T, 3840]
    layer_weights = router(token_embeds)  # [B, T, 49]

    # Apply weights during text embedding packing
    weighted_embeds = token_embeds.unsqueeze(-1) * layer_weights.unsqueeze(2)
"""

from llm_dit.router.token_layer_router import TokenLayerRouter

__all__ = ["TokenLayerRouter"]
