# Gemma3 Sub-Layer Architecture for Fine-Grained Routing

**Last Updated**: 2026-01-16

## Overview

This document details the internal structure of Gemma3 transformer layers and how to extract intermediate representations (attention outputs, MLP outputs) for finer-grained routing experiments in LTX-2.

## Motivation

Current LTX-2 routing uses whole-layer outputs (post-MLP states). However, attention and MLP components may contribute differently to different aspects of text understanding:

- **Attention layers**: Capture contextual relationships, token dependencies, semantic coherence
- **MLP layers**: Capture feature transformations, non-linear mappings, specialized knowledge

Routing at the sub-layer level could enable:
1. **Attention-specialized routing**: Route based on attention outputs for context-heavy prompts
2. **MLP-specialized routing**: Route based on MLP outputs for feature-heavy prompts
3. **Mixed routing**: Different combinations of attention/MLP from different layers
4. **Component ablation**: Test which component contributes more to generation quality

## Gemma3 Architecture

### Model Location

- **Transformers Library**: `.venv/lib/python3.13/site-packages/transformers/models/gemma3/modeling_gemma3.py`
- **Version**: 4.57.3
- **Class**: `Gemma3ForConditionalGeneration` (or `AutoModelForCausalLM` fallback)

### LTX-2 Integration

- **Custom Wrapper**: `src/llm_dit/encoders/gemma3.py`
- **Extraction Strategy**: All 49 decoder layer hidden states
- **Processing Pipeline**:
  1. Extract hidden states from all layers
  2. Per-layer normalization (8 * (x - mean) / range)
  3. Concatenate to 188,160 dimensions (49 × 3,840)
  4. Project via learned linear layer to 3,840 dimensions
  5. Pass to DiT cross-attention

## Gemma3DecoderLayer Structure

### Components

```python
class Gemma3DecoderLayer:
    # Attention components
    self.self_attn: Gemma3Attention          # Multi-head self-attention
    self.input_layernorm: RMSNorm            # Pre-attention normalization
    self.post_attention_layernorm: RMSNorm   # Post-attention normalization

    # MLP components
    self.mlp: Gemma3MLP                      # Feed-forward network
    self.pre_feedforward_layernorm: RMSNorm  # Pre-MLP normalization
    self.post_feedforward_layernorm: RMSNorm # Post-MLP normalization
```

### Forward Pass (Pre-Norm Architecture)

```python
def forward(hidden_states, attention_mask, output_attentions=False, ...):
    # ========================================
    # Attention Block
    # ========================================
    residual = hidden_states                                # [B, T, 3840]

    # Pre-normalization
    hidden_states = self.input_layernorm(hidden_states)    # [B, T, 3840]

    # Self-attention
    hidden_states, attn_weights = self.self_attn(
        hidden_states,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        ...
    )                                                       # [B, T, 3840]

    # Post-normalization
    hidden_states = self.post_attention_layernorm(hidden_states)  # [B, T, 3840]

    # EXTRACTION POINT 1: Attention output (before residual)
    attention_output = hidden_states  # Pure attention effect

    # Residual connection
    hidden_states = residual + hidden_states                # [B, T, 3840]

    # EXTRACTION POINT 2: Post-attention state (after residual)
    post_attention_state = hidden_states  # Attention integrated with input

    # ========================================
    # MLP Block
    # ========================================
    residual = hidden_states

    # Pre-normalization
    hidden_states = self.pre_feedforward_layernorm(hidden_states)  # [B, T, 3840]

    # Feed-forward network
    hidden_states = self.mlp(hidden_states)                # [B, T, 3840]

    # Post-normalization
    hidden_states = self.post_feedforward_layernorm(hidden_states)  # [B, T, 3840]

    # EXTRACTION POINT 3: MLP output (before residual)
    mlp_output = hidden_states  # Pure MLP effect

    # Residual connection
    hidden_states = residual + hidden_states                # [B, T, 3840]

    # EXTRACTION POINT 4: Post-MLP state (full layer output)
    # This is what output_hidden_states currently returns
    layer_output = hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_weights,)

    return outputs
```

## Four Extraction Points Per Layer

| Point | Location | Shape | What It Captures | Use Case |
|-------|----------|-------|------------------|----------|
| **Attention Output** | After `post_attention_layernorm`, before first residual | `[B, T, 3840]` | Pure attention effects isolated from prior state | Attention-only routing, ablation |
| **Post-Attention State** | After first residual, before MLP | `[B, T, 3840]` | Attention integrated with layer input | Intermediate state analysis |
| **MLP Output** | After `post_feedforward_layernorm`, before second residual | `[B, T, 3840]` | Pure MLP transformation effects | MLP-only routing, ablation |
| **Layer Output** | After second residual (current) | `[B, T, 3840]` | Complete layer transformation | Current LTX-2 routing (whole-layer) |

## Current Hidden States Collection

### Model-Level Loop

Located in `Gemma3TextModel.forward()` (modeling_gemma3.py:566-591):

```python
all_hidden_states = () if output_hidden_states else None

for decoder_layer in self.layers:
    if output_hidden_states:
        all_hidden_states += (hidden_states,)  # State BEFORE layer

    layer_outputs = decoder_layer(
        hidden_states,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
        ...
    )

    hidden_states = layer_outputs[0]  # State AFTER layer (post-MLP)

    if output_attentions:
        all_self_attns += (layer_outputs[1],)

# Final layer normalization
hidden_states = self.norm(hidden_states)

if output_hidden_states:
    all_hidden_states += (hidden_states,)  # Final normalized state
```

### Current Behavior

When `output_hidden_states=True`:
- Returns tuple of 50 tensors (embedding + 49 layers)
- **Index 0**: Embedding layer output (SKIPPED in LTX-2, line 420 of gemma3.py)
- **Indices 1-49**: Post-MLP output from each decoder layer
- **Index 50** (if included): Final layer norm output

LTX-2 currently uses indices 1-49 for routing.

## Extraction Strategies

### Option A: PyTorch Forward Hooks (Recommended)

**Advantages**:
- Non-invasive (no transformers library modification)
- Works across library updates
- Can be toggled on/off dynamically
- Already used in project (see `src/llm_dit/guidance/skip_layer.py`)

**Implementation**:

```python
class SubLayerExtractor:
    """
    Extract attention and MLP outputs using forward hooks.

    Usage:
        extractor = SubLayerExtractor(model, layer_indices=[0, 10, 20, 30, 40, 48])
        extractor.register()

        outputs = model(input_ids, attention_mask, output_hidden_states=True)

        sub_layers = extractor.get_stacked_outputs()
        # sub_layers['attention']: [B, T, 3840, 6]
        # sub_layers['mlp']: [B, T, 3840, 6]

        extractor.unregister()
    """

    def __init__(self, model, layer_indices: Optional[List[int]] = None):
        self.model = model
        self.layer_indices = layer_indices or list(range(len(model.model.layers)))

        # Storage for captured outputs
        self.attention_outputs = {}
        self.mlp_outputs = {}
        self.hooks = []

    def _make_attention_hook(self, layer_idx: int):
        """Create hook to capture attention output."""
        def hook(module, input, output):
            # Output of post_attention_layernorm is normalized attention output
            self.attention_outputs[layer_idx] = output.clone()
        return hook

    def _make_mlp_hook(self, layer_idx: int):
        """Create hook to capture MLP output."""
        def hook(module, input, output):
            # Output of post_feedforward_layernorm is normalized MLP output
            self.mlp_outputs[layer_idx] = output.clone()
        return hook

    def register(self):
        """Register hooks on specified layers."""
        for idx in self.layer_indices:
            layer = self.model.model.layers[idx]

            # Hook after post_attention_layernorm (captures attention output)
            attn_hook = layer.post_attention_layernorm.register_forward_hook(
                self._make_attention_hook(idx)
            )
            self.hooks.append(attn_hook)

            # Hook after post_feedforward_layernorm (captures MLP output)
            mlp_hook = layer.post_feedforward_layernorm.register_forward_hook(
                self._make_mlp_hook(idx)
            )
            self.hooks.append(mlp_hook)

    def unregister(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attention_outputs.clear()
        self.mlp_outputs.clear()

    def get_stacked_outputs(self):
        """
        Stack captured outputs into tensors.

        Returns:
            Dict with:
            - 'attention': [B, T, D, L] - Attention outputs
            - 'mlp': [B, T, D, L] - MLP outputs
        """
        if not self.attention_outputs:
            raise RuntimeError("No outputs captured. Did you call register() and run a forward pass?")

        attention_stack = torch.stack([
            self.attention_outputs[i]
            for i in sorted(self.attention_outputs.keys())
        ], dim=-1)  # [B, T, 3840, L]

        mlp_stack = torch.stack([
            self.mlp_outputs[i]
            for i in sorted(self.mlp_outputs.keys())
        ], dim=-1)  # [B, T, 3840, L]

        return {
            'attention': attention_stack,
            'mlp': mlp_stack,
        }
```

**Challenges**:
- Hooks capture normalized outputs (after RMSNorm), not raw sub-layer outputs
- Memory overhead for storing all intermediate activations
- Need to manage hook lifecycle carefully (register/unregister)

### Option B: Modified Decoder Layer (Invasive)

**Approach**: Monkey-patch `Gemma3DecoderLayer.forward()` to optionally return sub-layer outputs.

```python
class Gemma3DecoderLayerWithSubOutputs(Gemma3DecoderLayer):
    """Extended decoder layer that can output sub-layer states."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        output_sub_layers: bool = False,
        **kwargs
    ):
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out, attn_weights = self.self_attn(hidden_states, **kwargs)
        attn_out = self.post_attention_layernorm(attn_out)

        # Capture attention-only output
        attention_only = attn_out if output_sub_layers else None

        hidden_states = residual + attn_out

        # Capture post-attention state
        post_attention = hidden_states if output_sub_layers else None

        # MLP block
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        mlp_out = self.post_feedforward_layernorm(mlp_out)

        # Capture MLP-only output
        mlp_only = mlp_out if output_sub_layers else None

        hidden_states = residual + mlp_out

        # Build outputs
        outputs = (hidden_states,)
        if kwargs.get('output_attentions'):
            outputs += (attn_weights,)
        if output_sub_layers:
            outputs += (attention_only, post_attention, mlp_only)

        return outputs
```

**Advantages**:
- Clean interface with `output_sub_layers` flag
- Minimal memory overhead (conditional collection)
- Full control over extraction points

**Disadvantages**:
- Requires monkey-patching transformers library
- May break on library updates
- More complex to maintain
- Need to replace all layer instances in model

### Option C: Hybrid Approach (Recommended Initial Implementation)

Extend `Gemma3Encoder.encode_multilayer()` with hook-based sub-layer extraction:

```python
def encode_multilayer_with_sublayers(
    self,
    texts: Union[str, List[str]],
    extract_sub_layers: bool = False,
    layer_indices: Optional[List[int]] = None,
) -> dict:
    """
    Encode text with optional sub-layer extraction.

    Args:
        texts: Input text(s) to encode
        extract_sub_layers: If True, extract attention/MLP outputs separately
        layer_indices: Which layers to extract from (default: all)

    Returns:
        Dict with:
        - 'layer_stack': [B, T, D, L] - Full layer outputs (post-MLP)
        - 'attention_mask': [B, T]
        - 'projected': [B, T, 3840] - Projected embeddings (if return_projected=True)
        - 'seq_lengths': List[int]
        - 'attention_stack': [B, T, D, L] - Attention outputs (if extract_sub_layers=True)
        - 'mlp_stack': [B, T, D, L] - MLP outputs (if extract_sub_layers=True)
    """
    # Set up sub-layer extractor if requested
    extractor = None
    if extract_sub_layers:
        extractor = SubLayerExtractor(self._model, layer_indices)
        extractor.register()

    try:
        # Tokenize
        encoded = self._tokenizer(
            texts,
            padding="max_length",
            max_length=self._max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded.input_ids.to(self.device)
        attention_mask = encoded.attention_mask.to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

        # Extract standard layer stack
        hidden_states = outputs.hidden_states[1:]  # Skip embedding
        num_layers = min(len(hidden_states), GEMMA3_NUM_LAYERS)
        hidden_states = hidden_states[:num_layers]
        stacked = torch.stack(hidden_states, dim=-1)  # [B, T, D, L]

        seq_lengths = attention_mask.sum(dim=1).tolist()

        result = {
            'layer_stack': stacked,
            'attention_mask': attention_mask,
            'seq_lengths': [int(s) for s in seq_lengths],
        }

        # Add sub-layer outputs if requested
        if extract_sub_layers:
            sub_layers = extractor.get_stacked_outputs()
            result.update(sub_layers)

        return result

    finally:
        # Always clean up hooks
        if extractor is not None:
            extractor.unregister()
```

## Memory Considerations

### Current Memory Usage

- **Gemma3-12B Q4 QAT**: ~6GB VRAM
- **Hidden states (49 layers)**: `49 × B × T × 3840 × 2 bytes` (bf16)
  - For B=1, T=256: ~92MB

### Sub-Layer Extraction Overhead

- **Attention outputs (49 layers)**: ~92MB
- **MLP outputs (49 layers)**: ~92MB
- **Total additional overhead**: ~184MB

### Optimization Strategies

1. **Sparse extraction**: Extract only selected layers
   - Example: Every 5th layer (10 layers) → ~37MB overhead
   - Example: Early/middle/late (3 layers) → ~11MB overhead

2. **Immediate processing**: Clear intermediate storage after routing computation
   ```python
   sub_layers = extractor.get_stacked_outputs()
   routing_weights = router(sub_layers['attention'])
   del sub_layers  # Free memory
   torch.cuda.empty_cache()
   ```

3. **Streaming extraction**: Process layers in chunks for very long sequences

4. **Precision**: Use fp16 if needed (same memory as bf16)

### RTX 4090 Budget (24GB VRAM)

- **Gemma3 Q4**: ~6GB
- **LTX-2 DiT**: ~8-10GB
- **Latents + activations**: ~4-6GB
- **Available for experiments**: ~4GB
- **Sub-layer extraction**: ~0.2GB (full) or ~0.04GB (sparse)
- **Verdict**: Acceptable overhead, no issues

## Implementation Roadmap

### Phase 1: Basic Extraction (Week 1)

1. Add `SubLayerExtractor` class to `src/llm_dit/encoders/gemma3.py`
2. Write unit tests:
   - Verify hook registration/unregistration
   - Check output shapes match expected dimensions
   - Validate layer indexing
3. Create test script: `experiments/ltx2/test_sublayer_extraction.py`
   - Extract sub-layers for sample prompts
   - Print shapes and statistics
   - Verify memory usage

### Phase 2: Integration with Encoder (Week 1-2)

1. Add `extract_sub_layers` parameter to `encode_multilayer()`
2. Update docstrings and type hints
3. Test with existing routing code
4. Benchmark memory overhead and latency

### Phase 3: Routing Experiments (Week 2-3)

1. **Attention-only routing**:
   - Route using only attention outputs
   - Compare to full-layer routing
   - Measure SigLIP scores

2. **MLP-only routing**:
   - Route using only MLP outputs
   - Compare to full-layer routing

3. **Mixed routing**:
   - Different attention/MLP combinations
   - Learned weighting of attention vs MLP

4. **Component ablation**:
   - Zero out attention or MLP from specific layers
   - Measure impact on generation quality

### Phase 4: Documentation (Ongoing)

1. Document findings in `experiments/ltx2/docs/sublayer_routing_results.md`
2. Update `experiments/ltx2/AGENTS.md` with sub-layer architecture details
3. Create visualizations of attention vs MLP contributions
4. Write conclusions and recommendations

## Expected Research Questions

1. **Do attention and MLP contribute differently to text conditioning?**
   - Hypothesis: Attention captures context, MLP captures features
   - Test: Route using attention-only vs MLP-only

2. **Can sub-layer routing improve quality over whole-layer routing?**
   - Hypothesis: Finer granularity enables better layer selection
   - Test: Compare SigLIP scores of sub-layer routing vs current

3. **Which layers have the most important attention/MLP components?**
   - Hypothesis: Early layers = syntax (attention), late layers = semantics (MLP)
   - Test: Ablate specific components and measure impact

4. **Can we learn optimal attention/MLP routing weights?**
   - Hypothesis: Different prompts benefit from different sub-layer combinations
   - Test: Train small routing network to weight attention vs MLP per layer

## References

### Code Files

1. **Gemma3 Model**: `.venv/lib/python3.13/site-packages/transformers/models/gemma3/modeling_gemma3.py`
   - `Gemma3DecoderLayer` (line 344)
   - `Gemma3TextModel` (line 456)
   - Forward pass (lines 372-407, 566-591)

2. **LTX-2 Encoder**: `src/llm_dit/encoders/gemma3.py`
   - `Gemma3Encoder` class (line 127)
   - `encode_multilayer()` method (line 474)
   - Hidden state extraction (line 420)

3. **Hook Reference**: `src/llm_dit/guidance/skip_layer.py`
   - `_SkipHook` class (line 75)
   - Hook registration pattern (line 120)
   - Context manager usage (line 239)

### Related Documents

- `experiments/ltx2/AGENTS.md` - LTX-2 agent context
- `experiments/ltx2/docs/text_conditioning_architecture.md` - Full text conditioning pipeline
- `internal/log/log_2026-01-16.md` - Research log for this analysis

## Next Steps

1. Implement `SubLayerExtractor` class
2. Add unit tests for extraction correctness
3. Run initial extraction test on sample prompts
4. Verify memory usage is within bounds
5. Design first sub-layer routing experiment
6. Document results and iterate
