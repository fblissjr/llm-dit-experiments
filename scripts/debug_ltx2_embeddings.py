#!/usr/bin/env python3
"""
Debug LTX-2 Embeddings - Check hidden states and connector flow

Last Updated: 2026-01-15

This script traces the text embedding flow to identify where things break.
Uses the pipeline's encode_prompt method with CPU offload to fit in VRAM.
"""

import gc

import torch

print("=" * 60)
print("LTX-2 Embedding Debug")
print("=" * 60)

# Clear CUDA cache
gc.collect()
torch.cuda.empty_cache()

print(f"\nCUDA device: {torch.cuda.get_device_name(0)}")
print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from diffusers import LTX2Pipeline
from diffusers.utils import export_to_video

MODEL_PATH = "models/LTX-2"
PROMPT = "A cat walking"

print(f"\nModel path: {MODEL_PATH}")
print(f"Prompt: {PROMPT}")

# Load pipeline
print("\n1. Loading pipeline...")
pipe = LTX2Pipeline.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
)

# Enable CPU offload for memory management
print("   Enabling sequential CPU offload...")
pipe.enable_sequential_cpu_offload()

print("\n2. Component Info...")
print(f"   text_encoder: {type(pipe.text_encoder).__name__}")
print(f"   tokenizer: {type(pipe.tokenizer).__name__}")
print(f"   connectors: {type(pipe.connectors).__name__}")
print(f"   transformer: {type(pipe.transformer).__name__}")
print(f"   vae: {type(pipe.vae).__name__}")

# Check connector config
print("\n3. Connector Config...")
print(f"   text_proj_in_factor: {pipe.connectors.config.text_proj_in_factor}")
print(f"   caption_channels: {pipe.connectors.config.caption_channels}")
expected_input = (
    pipe.connectors.config.caption_channels * pipe.connectors.config.text_proj_in_factor
)
print(f"   Expected input dim: {expected_input}")

# Check number of hidden states the text encoder would return
print("\n4. Text Encoder Config...")
te_config = pipe.text_encoder.config
if hasattr(te_config, "text_config"):
    text_cfg = te_config.text_config
    print(f"   text_config.num_hidden_layers: {text_cfg.num_hidden_layers}")
    print(f"   text_config.hidden_size: {text_cfg.hidden_size}")
    num_hidden_states = text_cfg.num_hidden_layers + 1  # +1 for embedding
    packed_dim = num_hidden_states * text_cfg.hidden_size
    print(f"   Expected hidden states: {num_hidden_states}")
    print(f"   Expected packed dim: {packed_dim}")
else:
    print(f"   (Using top-level config)")
    if hasattr(te_config, "num_hidden_layers"):
        print(f"   num_hidden_layers: {te_config.num_hidden_layers}")
        print(f"   hidden_size: {te_config.hidden_size}")

# Test encode_prompt method (handles CPU offload automatically)
print("\n5. Testing encode_prompt method...")
prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
    prompt=PROMPT,
    negative_prompt=None,
    do_classifier_free_guidance=False,
    num_videos_per_prompt=1,
    max_sequence_length=128,
    device="cuda",
    dtype=torch.bfloat16,
)

print(f"   prompt_embeds shape: {prompt_embeds.shape}")
print(f"   prompt_attention_mask shape: {prompt_attention_mask.shape}")
print(f"   prompt_embeds dtype: {prompt_embeds.dtype}")

# Check embedding stats
print("\n6. Checking prompt_embeds statistics...")
print(f"   mean: {prompt_embeds.mean():.4f}")
print(f"   std: {prompt_embeds.std():.4f}")
print(f"   min: {prompt_embeds.min():.4f}")
print(f"   max: {prompt_embeds.max():.4f}")
print(f"   has NaN: {torch.isnan(prompt_embeds).any()}")
print(f"   has Inf: {torch.isinf(prompt_embeds).any()}")

# Check if embeddings are all zeros or constant
is_constant = prompt_embeds.std() < 1e-6
is_all_zeros = prompt_embeds.abs().max() < 1e-6
print(f"   is_constant: {is_constant}")
print(f"   is_all_zeros: {is_all_zeros}")

# Check attention mask
print("\n7. Checking attention mask...")
print(f"   mask sum: {prompt_attention_mask.sum()}")
print(f"   mask non-zero positions: {(prompt_attention_mask != 0).sum()}")

# Check connector weights (should have been loaded correctly)
print("\n8. Checking connector weights...")
proj = pipe.connectors.text_proj_in
print(f"   text_proj_in weight shape: {proj.weight.shape}")
print(f"   text_proj_in weight dtype: {proj.weight.dtype}")
print(f"   text_proj_in weight mean: {proj.weight.float().mean():.6f}")
print(f"   text_proj_in weight std: {proj.weight.float().std():.6f}")
print(f"   text_proj_in weight has NaN: {torch.isnan(proj.weight).any()}")

# Check learnable registers
print("\n9. Checking learnable registers...")
vc = pipe.connectors.video_connector
if hasattr(vc, "learnable_registers") and vc.learnable_registers is not None:
    regs = vc.learnable_registers
    print(f"   shape: {regs.shape}")
    print(f"   dtype: {regs.dtype}")
    print(f"   mean: {regs.float().mean():.6f}")
    print(f"   std: {regs.float().std():.6f}")
    print(f"   has NaN: {torch.isnan(regs).any()}")

# Check transformer config
print("\n10. Checking transformer config...")
tf_config = pipe.transformer.config
print(f"   in_channels: {tf_config.in_channels}")
print(f"   out_channels: {tf_config.out_channels}")
print(f"   num_attention_heads: {tf_config.num_attention_heads}")
print(f"   attention_head_dim: {tf_config.attention_head_dim}")
print(f"   num_layers: {tf_config.num_layers}")
print(f"   caption_channels: {tf_config.caption_channels}")

# Check scheduler
print("\n11. Checking scheduler...")
print(f"   Scheduler type: {type(pipe.scheduler).__name__}")
if hasattr(pipe.scheduler, "config"):
    sched_config = pipe.scheduler.config
    print(f"   num_train_timesteps: {getattr(sched_config, 'num_train_timesteps', 'N/A')}")
    print(f"   beta_schedule: {getattr(sched_config, 'beta_schedule', 'N/A')}")
    print(f"   prediction_type: {getattr(sched_config, 'prediction_type', 'N/A')}")

# Check VAE
print("\n12. Checking VAE...")
print(f"   VAE type: {type(pipe.vae).__name__}")
if hasattr(pipe.vae, "config"):
if hasattr(pipe.vae, 'config'):
    vae_config = pipe.vae.config
    print(f"   in_channels: {getattr(vae_config, 'in_channels', 'N/A')}")
    print(f"   out_channels: {getattr(vae_config, 'out_channels', 'N/A')}")
    print(f"   latent_channels: {getattr(vae_config, 'latent_channels', 'N/A')}")

print("\n" + "=" * 60)
print("Debug Complete")
print("=" * 60)

# Cleanup
del pipe
gc.collect()
torch.cuda.empty_cache()
