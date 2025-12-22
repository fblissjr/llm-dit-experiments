# lora guide

*last updated: 2025-12-22*

LoRAs are loaded and fused into the transformer weights at startup.

## via cli

```bash
# Single LoRA
--lora style.safetensors:0.8

# Multiple LoRAs (stackable)
--lora style.safetensors:0.5 --lora detail.safetensors:0.3
```

## via config.toml

```toml
[default.lora]
paths = ["style.safetensors", "detail.safetensors"]
scales = [0.5, 0.3]
```

## via python

```python
pipe = ZImagePipeline.from_pretrained(...)
pipe.load_lora("style.safetensors", scale=0.8)
pipe.load_lora(["lora1.safetensors", "lora2.safetensors"], scale=[0.5, 0.3])
```

## important notes

- LoRAs are **fused** (permanently merged) into weights
- To remove a LoRA, you must **reload the model**
- Multiple LoRAs are stackable with independent scales
- Scale controls LoRA influence: 0.0 = no effect, 1.0 = full effect

## implementation

LoRA loading is implemented in `src/llm_dit/utils/lora.py`. The fusion happens at model load time, not during inference, so there's no runtime overhead.
