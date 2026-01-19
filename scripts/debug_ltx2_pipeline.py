#!/usr/bin/env python3
"""
Debug LTX-2 Pipeline - Trace through each component

Last Updated: 2026-01-15

This script checks each component of the LTX-2 pipeline to identify issues.
"""

import gc

import torch

print("=" * 60)
print("LTX-2 Pipeline Debug")
print("=" * 60)

# Clear CUDA cache
gc.collect()
torch.cuda.empty_cache()

print(f"\nCUDA device: {torch.cuda.get_device_name(0)}")
print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Import diffusers
from diffusers import LTX2Pipeline

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

print("\n2. Checking components...")
print(f"   text_encoder: {type(pipe.text_encoder).__name__}")
print(f"   tokenizer: {type(pipe.tokenizer).__name__}")
print(f"   connectors: {type(pipe.connectors).__name__}")
print(f"   transformer: {type(pipe.transformer).__name__}")
print(f"   vae: {type(pipe.vae).__name__}")

# Check text encoder hidden state count
print("\n3. Testing text encoder...")
pipe.text_encoder.to("cuda")

inputs = pipe.tokenizer(
    [PROMPT],
    padding="max_length",
    max_length=128,
    truncation=True,
    return_tensors="pt",
).to("cuda")

print(f"   input_ids shape: {inputs.input_ids.shape}")

with torch.no_grad():
    outputs = pipe.text_encoder(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        output_hidden_states=True,
    )

print(f"   Number of hidden states: {len(outputs.hidden_states)}")
print(f"   Each hidden state shape: {outputs.hidden_states[0].shape}")

# Stack hidden states
stacked = torch.stack(outputs.hidden_states, dim=-1)
print(f"   Stacked shape: {stacked.shape}")
# Should be [B, T, D, L] = [1, 128, 3840, num_layers]

# Check _pack_text_embeds
print("\n4. Testing _pack_text_embeds...")
seq_lengths = inputs.attention_mask.sum(dim=-1)
print(f"   Sequence lengths: {seq_lengths}")

packed = pipe._pack_text_embeds(
    stacked,
    seq_lengths,
    device="cuda",
    padding_side=pipe.tokenizer.padding_side,
)
print(f"   Packed shape: {packed.shape}")
# Should be [B, T, D*L] = [1, 128, 188160] for 49 layers

# Check connectors
print("\n5. Testing connectors...")
pipe.connectors.to("cuda")

# Create attention mask for connectors
additive_mask = (1 - inputs.attention_mask.to(packed.dtype)) * -1000000.0
print(f"   Additive mask shape: {additive_mask.shape}")

video_embeds, audio_embeds, out_mask = pipe.connectors(packed, additive_mask, additive_mask=True)
print(f"   Video embeds shape: {video_embeds.shape}")
print(f"   Audio embeds shape: {audio_embeds.shape}")
print(f"   Output mask shape: {out_mask.shape}")

# Check embeddings stats
print("\n6. Checking embedding statistics...")
print(
    f"   Packed - mean: {packed.mean():.4f}, std: {packed.std():.4f}, min: {packed.min():.4f}, max: {packed.max():.4f}"
)
print(
    f"   Video - mean: {video_embeds.mean():.4f}, std: {video_embeds.std():.4f}, min: {video_embeds.min():.4f}, max: {video_embeds.max():.4f}"
)

# Check for NaN/Inf
print("\n7. Checking for NaN/Inf...")
print(f"   Packed has NaN: {torch.isnan(packed).any()}")
print(f"   Packed has Inf: {torch.isinf(packed).any()}")
print(f"   Video embeds has NaN: {torch.isnan(video_embeds).any()}")
print(f"   Video embeds has Inf: {torch.isinf(video_embeds).any()}")

# Check connectors weights
print("\n8. Checking connector weights...")
if hasattr(pipe.connectors, "text_proj_in"):
    proj = pipe.connectors.text_proj_in
    print(f"   text_proj_in weight shape: {proj.weight.shape}")
    print(f"   text_proj_in weight mean: {proj.weight.mean():.6f}")
    print(f"   text_proj_in weight std: {proj.weight.std():.6f}")

if hasattr(pipe.connectors, "video_connector"):
    vc = pipe.connectors.video_connector
    if hasattr(vc, "learnable_registers"):
        print(f"   video_connector learnable_registers shape: {vc.learnable_registers.shape}")
        print(f"   video_connector learnable_registers mean: {vc.learnable_registers.mean():.6f}")

print("\n" + "=" * 60)
print("Debug Complete")
print("=" * 60)

# Cleanup
del pipe
gc.collect()
torch.cuda.empty_cache()
