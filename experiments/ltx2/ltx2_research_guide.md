# LTX-2 Research Guide for Hobbyist Experimentation

## Document Purpose

This document provides context for Claude Code when working on LTX-2 related experiments. It summarizes architectural understanding, identifies high-value research directions achievable on hobbyist compute budgets, and outlines implementation approaches.

---

## Part 1: LTX-2 Architecture Overview

### High-Level Pipeline

```
Text Prompt
     ↓
Gemma3-12B (frozen) ──→ All Layer Hidden States [B, T, D, L]
     ↓
Feature Extractor ──→ Multi-layer projection via learned W ──→ [B, T, D]
     ↓
Text Connector ──→ Bidirectional attention + Thinking Tokens ──→ Refined Embeddings
     ↓
     ├──→ Video Text Connector ──→ Video DiT Stream (14B params)
     │                                      ↕ (Bidirectional Cross-Attention)
     └──→ Audio Text Connector ──→ Audio DiT Stream (5B params)
     ↓
Video VAE Decoder + Audio VAE Decoder + Vocoder ──→ Output
```

### Key Architectural Decisions

| Component | Design Choice | Rationale |
|-----------|---------------|-----------|
| Latent spaces | Separate VAEs for audio/video | Modality-appropriate compression, enables V2A/A2V |
| DiT streams | Asymmetric (14B video, 5B audio) | Video needs more capacity than audio |
| Cross-attention | Bidirectional with 1D temporal RoPE | Sub-frame audio-visual sync |
| Text encoder | Gemma3-12B (frozen) | Multilingual, phonetic precision for speech |
| Conditioning | Cross-modality AdaLN | Shared timestep conditioning across streams |

### Text Conditioning Pipeline (Most Relevant for This Research)

#### 1. Multi-Layer Feature Extraction

Problem: Decoder-only LLMs use causal attention. Earlier tokens don't see later context. Final layer embeddings aren't optimal for all uses.

Solution: Extract from ALL decoder layers, project down:

```python
# Their approach (simplified)
all_layers = gemma.forward(prompt, output_hidden_states=True)  # [L, B, T, D]
normalized = mean_center_scale(all_layers)
flattened = flatten(normalized)  # [B, T, D*L]
projected = flattened @ W  # [B, T, D] - learned projection
```

Key insight: Linguistic information is distributed across depth. Early layers capture phonetics, later layers capture semantics.

#### 2. Text Connector with Thinking Tokens

Problem: Projected embeddings still have causal limitations.

Solution: Bidirectional transformer blocks + learnable thinking tokens:

```python
# Thinking tokens appended to sequence
thinking_tokens = nn.Parameter(torch.randn(R, D))  # R = number of thinking tokens
input_sequence = concat([projected_embeddings, thinking_tokens])

# Bidirectional attention (full, not causal)
refined = bidirectional_transformer_blocks(input_sequence)
```

Thinking tokens serve as:
- Global information aggregators (see everything via bidirectional attention)
- Computational scratch space
- Carriers of implicit/contextual information

#### 3. Separate Connectors Per Modality

Video and audio streams get their own text connectors. This allows modality-specific text understanding (e.g., phonetics for audio, composition for video).

### Cross-Modal Attention

At each DiT layer:
1. Self-attention within modality
2. Text cross-attention for conditioning
3. Audio-visual cross-attention (bidirectional)
4. FFN

The AV cross-attention uses 1D temporal RoPE only (not spatial), focusing on time alignment.

---

## Part 2: Key Technical Concepts

### Why Thinking Tokens Matter

For models trained WITH thinking tokens (like LTX-2):
- The `<think>` token activates learned circuitry
- Reasoning tokens processed differently than normal text
- `</think>` triggers consolidation
- Conclusion tokens inherit enriched representations

For models NOT trained with thinking tokens:
- Appending learnable tokens at inference does nothing (untrained parameters)
- BUT: if using a thinking-capable encoder (Qwen3-4B with hybrid think mode), the encoder itself has learned what `<think>` means
- Encoding a reasoning trace produces different hidden states than encoding a terse prompt

### Inference-Time Thinking for DiT Conditioning

Pattern: Generate thinking, then encode the result:

```python
# User provides terse prompt
user_prompt = "a cat on a skateboard"

# LLM generates enriched description
expanded = llm.generate(f"<think>Consider composition, lighting, style...</think>{user_prompt}")

# Encode the full trace - encoder enters "reasoning mode" due to special tokens
hidden_states = llm.get_hidden_states(expanded)
```

The value is in what thinking produces AND in how the encoder processes `<think>` tokens (if trained with them).

### Attention Sinks

Phenomenon: Softmax must sum to 1. When model has nothing useful to attend to, probability mass goes to first/BOS tokens.

Relevance: 
- StreamingLLM uses this for infinite-length generation
- Thinking tokens serve similar role as intentional registers
- Understanding attention distribution helps diagnose conditioning quality

### Encoder Swapping Challenges

Direct encoder swapping doesn't work because:
- Dimension mismatch (different hidden sizes)
- Distribution shift (mean, variance, range differ)
- Semantic geometry (concepts live in different relative positions)
- Tokenization differences (same text → different token counts)

If you must swap encoders, options by cost:
1. Same family, different size: Linear projection + minimal tuning
2. Different families: Contrastive alignment + adapter
3. Most robust: Train cross-attention adapter at DiT input

---

## Part 3: Hobbyist Research Opportunities

### Why This Space is Tractable

Big labs optimize end-to-end. They don't:
- Ablate rigorously
- Explore inference-time alternatives
- Publish layer-by-layer analysis
- Compare across encoder architectures

Systematic methodology + careful analysis can contribute where compute cannot.

### High-Value Directions

#### Tier 1: Zero-Training Research

**Systematic prompt structure ablations**

```python
structures = [
    "terse prompt",
    "expanded description",
    "reasoning trace + conclusion",
    "conclusion only",
    "structured template (subject/action/lighting/camera)",
]
# Run same seeds across structures, measure FID/CLIP/human pref
```

Nobody has published rigorous comparisons for DiT conditioning specifically.

**Layer extraction comparisons**

```python
# Which layers matter for which attributes?
early_layers = layers[0:10]    # Phonetic/syntactic?
middle_layers = layers[10:30]  # Semantic?
late_layers = layers[30:40]    # Abstract?

# Does early layers give better text rendering?
# Does late layers give better composition?
```

**Activation steering for encoders**

```python
# Find directions in activation space
detailed_acts = mean([encoder(p) for p in detailed_prompts])
vague_acts = mean([encoder(p) for p in vague_prompts])
detail_direction = detailed_acts - vague_acts

# Steer at inference
hidden = encoder(user_prompt)
steered = hidden + alpha * detail_direction
```

Established for LLM behavior. Unexplored for DiT conditioning.

#### Tier 2: Lightweight Training ($10-100)

**Per-Token Layer Routing** (Primary recommendation)

Problem: Uniform layer blending averages out signal. Different tokens benefit from different layers.

```python
class TokenLayerRouter(nn.Module):
    """Learn which layers each token should draw from"""
    def __init__(self, hidden_dim, num_layers, bottleneck=64):
        self.to_query = nn.Linear(hidden_dim, bottleneck)
        self.layer_keys = nn.Parameter(torch.randn(num_layers, bottleneck))
    
    def forward(self, token_embeddings):
        queries = self.to_query(token_embeddings)  # [batch, seq, bottleneck]
        scores = torch.einsum('bsd,ld->bsl', queries, self.layer_keys)
        weights = F.softmax(scores, dim=-1)  # [batch, seq, num_layers]
        return weights
```

Parameters: ~166K for Gemma3. Very trainable.

**Cross-Attention Bottleneck Adapters**

```python
class CrossAttentionAdapter(nn.Module):
    def __init__(self, dim, bottleneck=64):
        self.down = nn.Linear(dim, bottleneck)
        self.up = nn.Linear(bottleneck, dim)
    
    def forward(self, encoder_hidden_states):
        return encoder_hidden_states + self.up(F.gelu(self.down(encoder_hidden_states)))
```

~50K parameters. Train at DiT input without touching encoder.

**Token Importance Learning**

```python
class TokenSelector(nn.Module):
    def __init__(self, dim):
        self.score = nn.Linear(dim, 1)
    
    def forward(self, hidden, k=0.5):
        scores = self.score(hidden).squeeze(-1)
        threshold = torch.quantile(scores, 1-k)
        mask = (scores > threshold).float()
        return hidden * mask.unsqueeze(-1)
```

Hypothesis: DiT attention diluted by uninformative tokens. Pruning helps.

#### Tier 3: Novel Directions

**Entropy-Guided Encoding**

```python
def entropy_weighted_encode(prompt):
    hidden, attention_maps = encoder(prompt, output_attentions=True)
    entropy = -torch.sum(attention_maps * torch.log(attention_maps + 1e-9), dim=-1)
    confidence = 1 / (entropy + 1)
    return hidden * confidence.unsqueeze(-1)
```

High entropy = model uncertain. Downweight those positions.

**Embedding-Space CFG**

```python
good_hidden = encoder("a majestic lion, golden hour lighting")
bad_hidden = encoder("a lion")
refined = good_hidden + alpha * (good_hidden - bad_hidden)
```

Classifier-free guidance applied to encoder output rather than DiT.

---

## Part 4: Implementation Guide for Token-Layer Routing

### Architecture

```python
class RoutedFeatureExtractor(nn.Module):
    """Replace LTX-2's fixed projection with per-token routing"""
    
    def __init__(self, num_layers, hidden_dim, bottleneck=64):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Router: determines layer weights per token
        self.to_query = nn.Linear(hidden_dim, bottleneck)
        self.layer_keys = nn.Parameter(torch.randn(num_layers, bottleneck))
        
        # Output projection for compatibility
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Temperature for sharpening (useful for distilled models)
        self.temperature = nn.Parameter(torch.tensor(1.0))
    
    def get_routing_weights(self, reference_hidden):
        """Compute per-token layer weights"""
        queries = self.to_query(reference_hidden)  # [batch, seq, bottleneck]
        scores = torch.einsum('bsd,ld->bsl', queries, self.layer_keys)
        weights = F.softmax(scores / self.temperature, dim=-1)
        return weights  # [batch, seq, num_layers]
    
    def forward(self, hidden_states_by_layer):
        """
        Args:
            hidden_states_by_layer: [num_layers, batch, seq, dim]
        Returns:
            routed: [batch, seq, dim]
        """
        # Use final layer as routing reference
        reference = hidden_states_by_layer[-1]
        weights = self.get_routing_weights(reference)
        
        # Weighted combination per token
        stacked = hidden_states_by_layer.permute(1, 2, 0, 3)  # [batch, seq, layers, dim]
        routed = torch.einsum('bsld,bsl->bsd', stacked, weights)
        
        return self.out_proj(routed)
```

### LTX-2 Specific: Dual-Stream Routing

```python
class DualStreamRouter(nn.Module):
    """Separate routing for audio vs video streams"""
    
    def __init__(self, hidden_dim, num_layers, bottleneck=64):
        super().__init__()
        
        # Shared query projection
        self.shared_query = nn.Linear(hidden_dim, bottleneck)
        
        # Modality-specific layer keys
        self.video_layer_keys = nn.Parameter(torch.randn(num_layers, bottleneck))
        self.audio_layer_keys = nn.Parameter(torch.randn(num_layers, bottleneck))
    
    def forward(self, hidden_states_by_layer, modality='video'):
        reference = hidden_states_by_layer[-1]
        queries = self.shared_query(reference)
        
        if modality == 'video':
            keys = self.video_layer_keys
        else:
            keys = self.audio_layer_keys
        
        scores = torch.einsum('bsd,ld->bsl', queries, keys)
        weights = F.softmax(scores, dim=-1)
        
        stacked = hidden_states_by_layer.permute(1, 2, 0, 3)
        routed = torch.einsum('bsld,bsl->bsd', stacked, weights)
        
        return routed
```

Rationale: Audio may benefit from earlier (phonetic) layers, video from later (semantic) layers.

### Training Signals

#### Option A: DiT Attention Mimicry (Recommended First)

Use DiT's own cross-attention as free supervision:

```python
def extract_dit_attention_preferences(pipeline, prompt, hidden_by_layer):
    """Extract what DiT naturally attends to"""
    with torch.no_grad():
        # Uniform blend for baseline conditioning
        uniform_hidden = hidden_by_layer.mean(dim=0)
        video_cond = pipeline.video_text_connector(uniform_hidden)
        audio_cond = pipeline.audio_text_connector(uniform_hidden)
        
        # Run single denoising step with attention capture
        noise_video = torch.randn(...)
        noise_audio = torch.randn(...)
        t = torch.tensor([0.5])  # Mid-timestep, most informative
        
        _, attentions = pipeline.dit(
            video_latent=noise_video,
            audio_latent=noise_audio,
            timestep=t,
            video_encoder_hidden_states=video_cond,
            audio_encoder_hidden_states=audio_cond,
            output_attentions=True
        )
        
        # Aggregate: which text tokens get attended to?
        video_text_attn = attentions['video_text_cross_attention']
        audio_text_attn = attentions['audio_text_cross_attention']
        
        # Mean across heads, layers, spatial positions
        video_importance = video_text_attn.mean(dim=(0, 1, 2))  # [seq]
        audio_importance = audio_text_attn.mean(dim=(0, 1, 2))  # [seq]
    
    return {'video': video_importance, 'audio': audio_importance}


def mimicry_loss(router, hidden_by_layer, dit_preferences):
    """Train router to match DiT's natural preferences"""
    router_weights = router(hidden_by_layer)  # [batch, seq, num_layers]
    
    # Aggregate to per-token importance
    router_importance = router_weights.sum(dim=-1)  # [batch, seq]
    router_importance = router_importance / router_importance.sum(dim=-1, keepdim=True)
    
    video_loss = F.mse_loss(router_importance, dit_preferences['video'])
    audio_loss = F.mse_loss(router_importance, dit_preferences['audio'])
    
    return video_loss + 0.5 * audio_loss
```

Why this works: You're learning to pre-emphasize what DiT already wants. No generation needed during training loop.

#### Option B: Contrastive Generation

```python
def contrastive_loss(router, encoder, dit, prompt):
    hidden_by_layer = encoder(prompt, output_hidden_states=True)
    
    # Routed generation
    routed_hidden = router(hidden_by_layer)
    image_routed = dit.generate(routed_hidden, seed=42)
    
    # Baseline generation
    uniform_hidden = hidden_by_layer.mean(dim=0)
    image_uniform = dit.generate(uniform_hidden, seed=42)
    
    # Reward difference
    score_routed = clip_score(image_routed, prompt)
    score_uniform = clip_score(image_uniform, prompt)
    
    margin = 0.05
    loss = F.relu(margin - (score_routed - score_uniform))
    return loss
```

More expensive (requires generation), but directly optimizes quality.

#### Option C: Reconstruction Pretraining (Cheap Warmup)

```python
def reconstruction_loss(router, hidden_by_layer):
    """Can router reconstruct final layer from weighted blend of earlier layers?"""
    target = hidden_by_layer[-1]
    
    # Route using only earlier layers
    routed = router(hidden_by_layer[:-1])
    
    return F.mse_loss(routed, target)
```

No DiT needed. Good for initializing router before fine-tuning.

### Training Loop

```python
class RouterTrainer:
    def __init__(self, pipeline, router, lr=1e-4):
        self.pipeline = pipeline  # Frozen
        self.router = router  # Trainable
        self.optimizer = torch.optim.AdamW(router.parameters(), lr=lr)
    
    def get_hidden_states(self, prompt):
        with torch.no_grad():
            outputs = self.pipeline.text_encoder(
                prompt,
                output_hidden_states=True
            )
            return torch.stack(outputs.hidden_states)
    
    def phase1_reconstruction(self, prompts, steps=1000):
        """Cheap pretraining"""
        for step in range(steps):
            prompt = random.choice(prompts)
            hidden_by_layer = self.get_hidden_states(prompt)
            
            loss = reconstruction_loss(self.router, hidden_by_layer)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if step % 100 == 0:
                print(f"Step {step}: reconstruction loss = {loss.item():.4f}")
    
    def phase2_mimicry(self, prompts, steps=2000):
        """Learn from DiT attention"""
        for step in range(steps):
            prompt = random.choice(prompts)
            hidden_by_layer = self.get_hidden_states(prompt)
            
            dit_prefs = extract_dit_attention_preferences(
                self.pipeline, prompt, hidden_by_layer
            )
            
            loss = mimicry_loss(self.router, hidden_by_layer, dit_prefs)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if step % 100 == 0:
                print(f"Step {step}: mimicry loss = {loss.item():.4f}")
    
    def phase3_generation(self, prompts, steps=1000):
        """Fine-tune with generation signal"""
        for step in range(steps):
            prompt = random.choice(prompts)
            
            loss = contrastive_loss(
                self.router, 
                self.pipeline.text_encoder,
                self.pipeline,
                prompt
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if step % 50 == 0:
                print(f"Step {step}: contrastive loss = {loss.item():.4f}")
```

### Evaluation

```python
def evaluate_router(router, pipeline, test_prompts):
    results = {
        'layer_entropy': [],
        'token_variance': [],
        'clip_baseline': [],
        'clip_routed': [],
        'output_difference': [],
    }
    
    for prompt in test_prompts:
        hidden_by_layer = get_hidden_states(prompt)
        weights = router.get_routing_weights(hidden_by_layer[-1])
        
        # Routing diagnostics
        entropy = -(weights * torch.log(weights + 1e-9)).sum(dim=-1).mean()
        token_var = weights.var(dim=1).mean()
        results['layer_entropy'].append(entropy.item())
        results['token_variance'].append(token_var.item())
        
        # Generation comparison
        uniform = hidden_by_layer.mean(dim=0)
        routed = router(hidden_by_layer)
        
        img_uniform = pipeline.generate(uniform, seed=42)
        img_routed = pipeline.generate(routed, seed=42)
        
        results['clip_baseline'].append(clip_score(img_uniform, prompt))
        results['clip_routed'].append(clip_score(img_routed, prompt))
        results['output_difference'].append(
            (img_uniform - img_routed).abs().mean().item()
        )
    
    # Summary
    print("=== Router Evaluation ===")
    print(f"Layer entropy: {np.mean(results['layer_entropy']):.4f}")
    print(f"  (Max entropy = {np.log(router.num_layers):.4f})")
    print(f"Token variance: {np.mean(results['token_variance']):.4f}")
    print(f"CLIP baseline: {np.mean(results['clip_baseline']):.4f}")
    print(f"CLIP routed: {np.mean(results['clip_routed']):.4f}")
    print(f"Output difference: {np.mean(results['output_difference']):.4f}")
    
    return results
```

Interpretation:
- If `token_variance ≈ 0`: Router collapsed to global weighting (bad)
- If `layer_entropy ≈ max`: Router isn't specializing (bad)
- If `output_difference ≈ 0`: Router not affecting outputs (wiring issue)
- Want: Moderate entropy, high token variance, positive CLIP improvement

---

## Part 5: Expected Outcomes

### Realistic Improvements

| Aspect | Likelihood | Magnitude |
|--------|------------|-----------|
| Complex prompt adherence | High | 5-15% relative |
| Cross-seed consistency | Medium-High | Measurable |
| Attribute accuracy | Medium | Prompt-dependent |
| Compositional accuracy | Medium | Modest |
| Text rendering | Low | Needs architectural changes |

### Quantitative Ballpark

```
Baseline CLIP score:           ~0.28
With reasoning traces:         ~0.29-0.31
With trained router:           ~0.30-0.33

Human preference vs baseline:  55-65% (good)
                               65-75% (very good)
                               >75% (significant finding)
```

### Failure Modes

**Reward hacking**: CLIP goes up but images look worse
- Fix: Add consistency loss, switch to preference training

**Mode collapse**: Router ignores routing, uses uniform weights
- Fix: Diversity regularization, entropy penalty

**Overfitting**: Works on training prompts, fails on new styles
- Fix: Diverse training set, regularization

**Noise amplification**: Router amplifies garbage dimensions
- Fix: Add dimension gating, stronger bottleneck

### Timeline

| Phase | Time | Compute Cost |
|-------|------|--------------|
| Setup, verify shapes | 1 day | Minimal |
| Minimal router, sanity check | 1 day | Minimal |
| Attention extraction | 1-2 days | ~$5 |
| Mimicry training | 3-4 days | ~$10-20 |
| Generation fine-tuning | 1 week | ~$20-50 |
| Ablations | 1 week | ~$10-20 |

### Success Criteria

**Minimal (worth continuing)**:
- 55%+ preference vs baseline
- Reduced variance across seeds
- Qualitative improvement on complex prompts

**Moderate (worth writing up)**:
- 65%+ preference
- Clear improvement on specific categories
- Interpretable learned weights

**Strong (publishable)**:
- 70%+ preference
- Generalizes across prompt styles
- Ablations show each component contributes

---

## Part 6: Quick Reference

### Commands for Claude Code

```bash
# Typical workflow
1. "Show me the text encoder integration in LTX-2 codebase"
2. "Find where hidden states are extracted from Gemma"
3. "Locate the text connector module"
4. "Find cross-attention implementation in the DiT"
5. "Show how to extract attention weights from forward pass"
```

### Key Files to Understand (Typical Structure)

```
ltx2/
├── models/
│   ├── text_encoder.py      # Gemma integration
│   ├── feature_extractor.py # Multi-layer projection
│   ├── text_connector.py    # Bidirectional + thinking tokens
│   ├── dit.py               # Dual-stream transformer
│   └── vae.py               # Video/audio VAEs
├── pipelines/
│   └── ltx2_pipeline.py     # Full inference pipeline
```

### Shapes to Expect (Approximate)

```python
# Gemma3-12B
num_layers = 40
hidden_dim = 3584  # Check actual value

# After feature extraction
text_embedding: [batch, seq_len, hidden_dim]

# After text connector (with thinking tokens)
video_conditioning: [batch, seq_len + num_thinking, hidden_dim]
audio_conditioning: [batch, seq_len + num_thinking, hidden_dim]

# DiT latents
video_latent: [batch, frames, height//8, width//8, channels]
audio_latent: [batch, time_steps, audio_channels]
```

### Minimal Test Script

```python
# Verify everything works before training
def sanity_check():
    # 1. Load pipeline
    pipeline = load_ltx2_pipeline()
    
    # 2. Extract hidden states
    prompt = "a cat on a skateboard"
    hidden_by_layer = get_all_hidden_states(pipeline, prompt)
    print(f"Hidden states shape: {hidden_by_layer.shape}")
    # Expected: [num_layers, 1, seq_len, hidden_dim]
    
    # 3. Initialize router
    router = RoutedFeatureExtractor(
        num_layers=hidden_by_layer.shape[0],
        hidden_dim=hidden_by_layer.shape[-1]
    )
    
    # 4. Test forward pass
    routed = router(hidden_by_layer)
    print(f"Routed shape: {routed.shape}")
    # Expected: [1, seq_len, hidden_dim]
    
    # 5. Check routing weights
    weights = router.get_routing_weights(hidden_by_layer[-1])
    print(f"Weights shape: {weights.shape}")
    print(f"Weights sum per token: {weights.sum(dim=-1)}")
    # Expected: all 1.0
    
    # 6. Generate with both
    uniform = hidden_by_layer.mean(dim=0)
    img_uniform = pipeline.generate(uniform, seed=42)
    img_routed = pipeline.generate(routed, seed=42)
    
    diff = (img_uniform - img_routed).abs().mean()
    print(f"Output difference: {diff}")
    # If ~0, router isn't connected properly
    
    print("Sanity check passed!")
```

---

## Appendix: Concepts from LLM Domain

These LLM concepts transfer to understanding DiT text conditioning:

| LLM Concept | DiT Analog | Notes |
|-------------|------------|-------|
| Attention sinks | Register tokens / thinking tokens | Global information aggregation |
| Layer-wise probing | Layer extraction for conditioning | Information at different depths |
| Activation steering | Embedding space manipulation | Add directions to shift output |
| Chain-of-thought | Reasoning traces for conditioning | Richer representations |
| KV cache | Cached text embeddings | Reuse across diffusion steps |
| Causal attention limits | Why text connector uses bidirectional | Fix the sequential blindness |

---

## Version History

- v1.0: Initial document summarizing research discussion
