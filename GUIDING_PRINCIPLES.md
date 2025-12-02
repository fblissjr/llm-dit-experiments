# Guiding Principles

> Core architectural decisions and their rationale. Read before making significant changes.

---

## Principle 0: Experimentation First

This is a hobbyist exploration platform. Design choices prioritize:
1. **Understandability** over performance optimization
2. **Flexibility** over convenience
3. **Reproducibility** over speed

We want to understand *why* things work, not just *that* they work.

---

## Core Architectural Decisions

### 1. Protocol-Based LLM Backend Abstraction

**Decision**: Use Python Protocol (duck typing) instead of ABC inheritance.

**Rationale**:
- Allows any backend that implements the interface
- No forced inheritance hierarchy
- Easy to add new backends (vLLM, SGLang, mlx) without modifying existing code
- Matches Python's "batteries included" philosophy

**Interface**:
```python
@runtime_checkable
class TextEncoderBackend(Protocol):
    @property
    def embedding_dim(self) -> int: ...
    def encode_batch(self, texts: List[str]) -> tuple[Tensor, Tensor]: ...
```

### 2. Manual Chat Template Construction

**Decision**: Construct Qwen3-4B chat templates manually in `Qwen3Formatter`.

**Alternatives Considered**:
- Use `tokenizer.apply_chat_template()` (simpler, less control)
- Let backends handle templating (inconsistent across backends)

**Rationale**:
- Full control over thinking block injection
- Consistent behavior across all backends
- Aligns with our ComfyUI implementation in spirit, with differences in implementation details
- Enables experimentation with template variations

### 3. Match Diffusers Embedding Extraction

**Decision**: Extract only valid tokens (filter by attention mask).

```python
# Like diffusers, not like ComfyUI (which returns padded sequences)
embeddings_list = [embeds[mask] for embeds, mask in zip(batch_embeds, masks)]
```

**Rationale**:
- Memory efficient
- Matches reference implementation
- DiT expects variable-length sequences with its own padding

### 4. Living Documentation

**Decision**: Maintain living docs that evolve with the project.

**Key Files**:
- `SESSION_CONTINUITY.md` - Where we are, what's next
- `GUIDING_PRINCIPLES.md` - Why we made decisions (this file)
- `CHANGELOG.md` - What changed (semantic versioning)

**Rationale**:
- Every session can start with context
- Decisions are documented with rationale
- History is preserved without cluttering code comments

### 5. Experiment-as-Code

**Decision**: Each experiment is a self-contained directory with config, script, results.

```
experiments/001_name/
    README.md       # Hypothesis, methodology
    config.yaml     # Frozen parameters
    run.py          # Executable script
    results/        # Outputs
    FINDINGS.md     # Conclusions
```

**Rationale**:
- Reproducible by design
- Easy to compare experiments
- Version-controlled findings

---

## What We Don't Do

### No Over-Abstraction

Don't add abstraction layers until patterns emerge from actual usage.

**Example**: Don't create a `DiTBackend` Protocol until we actually support multiple DiT implementations.

### No MLflow/W&B Overhead

Simple JSON/YAML tracking is sufficient. Complexity deferred until proven necessary.

### No Premature Optimization

Start with `transformers` backend. Add vLLM when we actually need batch throughput.

---

## Technical Specifications (Z-Image)

These are fixed by the model architecture:

| Spec | Value | Source |
|------|-------|--------|
| Text encoder hidden | 2560 | Qwen3-4B config |
| DiT hidden | 3840 | Z-Image DiT |
| Context refiner layers | 2 | No timestep modulation |
| Main DiT layers | 30 | With timestep modulation |
| CFG scale | 0.0 | Decoupled-DMD training |
| Steps | 8-9 | Turbo distillation |
| VAE channels | 16 | Wan-family |

---

## Future Considerations

Things we might need but don't implement yet:

1. **vLLM/SGLang backends** - When batch throughput matters
2. **mlx backend** - For Apple Silicon users
3. **Quantized backends** - When VRAM is constrained
4. **Multi-turn encoding** - Ported from ComfyUI ZImageTurnBuilder
5. **Vision encoder support** - For future Tongyi models with vision

---

## References

- Z-Image paper analysis: `~/workspace/ComfyUI-QwenImageWanBridge/internal/z_image_paper_analysis/`
- DiffSynth reference: `~/workspace/ComfyUI-QwenImageWanBridge/coderef/DiffSynth-Studio/`
- diffusers pipeline: `~/workspace/ComfyUI-QwenImageWanBridge/coderef/diffusers/`
