# chat template format

*last updated: 2025-12-22*

## model-specific template behavior

**Qwen3-4B** and **Qwen3-VL** models have DIFFERENT chat template capabilities:

| Feature | Qwen3-4B | Qwen3-VL-4B-Instruct | Qwen3-VL-4B-Thinking |
|---------|----------|----------------------|----------------------|
| `enable_thinking` parameter | Supported | **NOT supported** | **NOT supported** |
| Auto think block generation | Yes (via template) | No | **Yes (native)** |
| Manual token injection needed | No | **Yes** | No |

## qwen3-4b chat template

The tokenizer supports `enable_thinking` with counterintuitive naming:
- `enable_thinking=True` -> NO think block (model CAN think on its own)
- `enable_thinking=False` -> ADD empty `<think>\n\n</think>\n\n` (skip thinking)

The official Z-Image HuggingFace Space uses `enable_thinking=True`:

```
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
```

## qwen3-vl chat template (different)

**Qwen3-VL's tokenizer does NOT support `enable_thinking`.**

For Qwen3-VL, manually inject think block tokens:

```python
from llm_dit.constants import (
    THINK_START_TOKEN_ID,  # 151667
    THINK_END_TOKEN_ID,    # 151668
    DOUBLE_NEWLINE_TOKEN_ID,  # 271
)

think_tokens = [THINK_START_TOKEN_ID, DOUBLE_NEWLINE_TOKEN_ID,
                THINK_END_TOKEN_ID, DOUBLE_NEWLINE_TOKEN_ID]
```

## full format (all components)

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
<think>
{thinking_content}
</think>

{assistant_content}
```

## content-driven think block logic

| Condition | Result |
|-----------|--------|
| Default (no thinking_content, no force_think_block) | No think block (matches official) |
| `thinking_content` provided | Add think block with content |
| `force_think_block=True` | Add empty think block |

## token ids (shared by qwen3-4b and qwen3-vl)

| Token | ID | Constant | Usage |
|-------|-----|----------|-------|
| `<\|endoftext\|>` | 151643 | `ENDOFTEXT_TOKEN_ID` | PAD token |
| `<\|im_start\|>` | 151644 | `IM_START_TOKEN_ID` | Chat message start |
| `<\|im_end\|>` | 151645 | `IM_END_TOKEN_ID` | Chat message end / EOS |
| `<\|vision_start\|>` | 151652 | `VISION_START_TOKEN_ID` | Vision content start (VL) |
| `<\|vision_end\|>` | 151653 | `VISION_END_TOKEN_ID` | Vision content end (VL) |
| `<\|image_pad\|>` | 151655 | `IMAGE_PAD_TOKEN_ID` | Image placeholder (VL) |
| `<think>` | 151667 | `THINK_START_TOKEN_ID` | Thinking block start |
| `</think>` | 151668 | `THINK_END_TOKEN_ID` | Thinking block end |
| `\n\n` | 271 | `DOUBLE_NEWLINE_TOKEN_ID` | Double newline |

## qwen3 sampler settings

**For text generation (rewriting)** - NOT for encoding:

| Parameter | Thinking Mode | Non-Thinking |
|-----------|---------------|--------------|
| temperature | 0.6 | 0.7 |
| top_p | 0.95 | 0.8 |
| top_k | 20 | 20 |
| min_p | 0.0 | 0.0 |

**Never use temperature=0** for Qwen3 - causes endless repetition.

## compatible models

For Z-Image text encoding, the model must have **2560 hidden dimensions**:

| Model | Hidden Dim | Compatible |
|-------|-----------|------------|
| Qwen3-4B | 2560 | Yes |
| Qwen3-VL-4B-Instruct | 2560 | Yes |
| Qwen3-VL-4B-Thinking | 2560 | Yes |
| Qwen3-4B-Instruct-2507 | 2560 | Yes |
| Qwen3-8B+ | 4096+ | No |
