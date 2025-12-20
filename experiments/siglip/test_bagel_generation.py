#!/usr/bin/env python3
"""
last updated: 2025-12-19

Full generation test with Bagel's adapted connector weights.
"""

import sys
from pathlib import Path

DIFFUSERS_PR_PATH = Path(__file__).parent.parent.parent / "coderef/diffusers/src"
sys.path.insert(0, str(DIFFUSERS_PR_PATH))

import torch
from PIL import Image
from safetensors import safe_open

# Paths
BAGEL_PATH = Path.home() / "Storage/ByteDance-Seed_BAGEL-7B-MoT/ema.safetensors"
ZIMAGE_PATH = Path.home() / "Storage/Tongyi-MAI_Z-Image-Turbo"
SIGLIP_PATH = Path.home() / "Storage/google_siglip2-so400m-patch14-384"
QWEN3_PATH = Path.home() / "Storage/Qwen3-4B"

OUTPUT_DIR = Path(__file__).parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_and_adapt_bagel_connector(target_dim=3840):
    """Load Bagel's connector and adapt to Z-Image dimensions."""
    print("Loading Bagel connector...")

    with safe_open(str(BAGEL_PATH), framework='pt', device='cpu') as f:
        fc1_weight = f.get_tensor('connector.fc1.weight')  # (3584, 1152)
        fc1_bias = f.get_tensor('connector.fc1.bias')

    current_out = fc1_weight.shape[0]
    pad_size = target_dim - current_out

    # Pad with small random values
    std = fc1_weight.float().std().item()
    pad_weight = torch.randn(pad_size, fc1_weight.shape[1], dtype=fc1_weight.dtype) * (std * 0.1)
    pad_bias = torch.zeros(pad_size, dtype=fc1_bias.dtype)

    adapted_weight = torch.cat([fc1_weight, pad_weight], dim=0)
    adapted_bias = torch.cat([fc1_bias, pad_bias], dim=0)

    print(f"  Adapted: {fc1_weight.shape} -> {adapted_weight.shape}")
    return adapted_weight, adapted_bias


def load_pipeline_with_bagel():
    """Load Z-Image pipeline with Bagel connector weights."""
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
    from transformers import AutoModel, AutoTokenizer, AutoProcessor

    print("\nLoading pipeline components...")

    # Load VAE
    print("  VAE...")
    vae = AutoencoderKL.from_pretrained(ZIMAGE_PATH, subfolder="vae", torch_dtype=torch.bfloat16)

    # Load text encoder
    print("  Text encoder...")
    text_encoder = AutoModel.from_pretrained(QWEN3_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_PATH, trust_remote_code=True)

    # Load scheduler
    print("  Scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ZIMAGE_PATH, subfolder="scheduler")

    # Load SigLIP
    print("  SigLIP...")
    siglip_full = AutoModel.from_pretrained(SIGLIP_PATH)
    siglip = siglip_full.vision_model
    siglip_processor = AutoProcessor.from_pretrained(SIGLIP_PATH)

    # Load transformer with siglip support
    print("  Transformer...")
    base_transformer = ZImageTransformer2DModel.from_pretrained(
        ZIMAGE_PATH, subfolder="transformer", torch_dtype=torch.bfloat16
    )

    # Create new transformer with siglip_feat_dim
    config = dict(base_transformer.config)
    config["siglip_feat_dim"] = 1152

    transformer = ZImageTransformer2DModel(**config)

    # Copy base weights
    missing, _ = transformer.load_state_dict(base_transformer.state_dict(), strict=False)
    print(f"  Missing keys: {len(missing)}")

    # Load Bagel connector weights
    print("  Loading Bagel connector weights...")
    adapted_weight, adapted_bias = load_and_adapt_bagel_connector(3840)

    # Inject into siglip_embedder
    siglip_linear = transformer.siglip_embedder[1]
    with torch.no_grad():
        siglip_linear.weight.copy_(adapted_weight.to(siglip_linear.weight.dtype))
        siglip_linear.bias.copy_(adapted_bias.to(siglip_linear.bias.dtype))
    print("  Bagel weights injected!")

    # Initialize siglip_refiner with small random values
    print("  Initializing siglip_refiner...")
    for name, param in transformer.named_parameters():
        if "siglip_refiner" in name:
            if param.dim() >= 2:
                torch.nn.init.xavier_uniform_(param, gain=0.1)
            else:
                torch.nn.init.zeros_(param)

    transformer = transformer.to(torch.bfloat16)

    return {
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "transformer": transformer,
        "siglip": siglip,
        "siglip_processor": siglip_processor,
    }


def encode_prompt(tokenizer, text_encoder, prompt, device):
    """Encode text prompt."""
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted, padding="max_length", max_length=512, truncation=True, return_tensors="pt")

    with torch.no_grad():
        outputs = text_encoder(
            input_ids=inputs.input_ids.to(device),
            attention_mask=inputs.attention_mask.to(device),
            output_hidden_states=True,
        )
        embeddings = outputs.hidden_states[-2]

    mask = inputs.attention_mask.bool()
    embeddings = embeddings[0][mask[0]]
    return embeddings


def encode_siglip(siglip, processor, image, device, scale_factor=11.0):
    """Encode image with SigLIP and scale to match text embedding stats."""
    inputs = processor(images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = siglip(pixel_values=pixel_values)
        hidden_state = outputs.last_hidden_state

    B, N, C = hidden_state.shape
    H = W = int(N ** 0.5)
    hidden_state = hidden_state.squeeze(0).view(H, W, C)

    # Scale to match text embedding statistics (std ~37 vs ~3.3)
    hidden_state = hidden_state * scale_factor

    return hidden_state


def generate(
    components,
    prompt,
    reference_image=None,
    height=512,
    width=512,
    num_steps=4,
    device="cuda",
):
    """Run generation with or without reference image."""
    vae = components["vae"].to(device)
    text_encoder = components["text_encoder"].to(device)
    transformer = components["transformer"].to(device)
    siglip = components["siglip"].to(device)
    scheduler = components["scheduler"]
    tokenizer = components["tokenizer"]
    siglip_processor = components["siglip_processor"]

    dtype = torch.bfloat16

    print(f"\nGenerating {width}x{height}...")
    print(f"  Prompt: {prompt}")
    print(f"  Reference: {'Yes' if reference_image else 'No'}")

    # Encode prompt
    cap_feats = encode_prompt(tokenizer, text_encoder, prompt, device).to(dtype)

    # Prepare latent
    latent_h, latent_w = height // 16, width // 16
    latents = torch.randn(16, 1, latent_h, latent_w, device=device, dtype=dtype)

    # Simple text-only mode first (like basic Z-Image)
    # cap_feats should be a single tensor, not a list of lists for basic mode
    if reference_image is None:
        cap_feats_list = [cap_feats]  # List of tensors, one per batch
        siglip_feats = None
        cond_latents = None
    else:
        # Omni mode with SigLIP
        siglip_emb = encode_siglip(siglip, siglip_processor, reference_image, device).to(dtype)
        print(f"  SigLIP embedding: {siglip_emb.shape}, std={siglip_emb.std():.2f}")

        # For omni mode with siglip, cap_feats is still a list of tensors
        # siglip_feats is list of lists: [[emb1, emb2, ...], ...] per batch
        siglip_feats = [[siglip_emb]]  # One reference image
        cap_feats_list = [cap_feats]   # Text embeddings as tensor
        cond_latents = None  # No VAE conditioning for now

    # Scheduler setup
    image_seq_len = latent_h * latent_w // 4
    mu = 0.5 + (1.15 - 0.5) * (image_seq_len - 256) / (4096 - 256)
    mu = max(0.5, min(1.15, mu))
    scheduler.set_timesteps(num_steps, device=device, mu=mu)

    # Denoising
    print("  Denoising...")
    for i, t in enumerate(scheduler.timesteps):
        timestep = t.expand(1)
        timestep_norm = (1000 - timestep) / 1000

        x = [latents]

        with torch.no_grad():
            if siglip_feats is not None:
                output = transformer(
                    x=x,
                    t=timestep_norm,
                    cap_feats=cap_feats_list,
                    siglip_feats=siglip_feats,
                    return_dict=False,
                )
            else:
                output = transformer(
                    x=x,
                    t=timestep_norm,
                    cap_feats=cap_feats_list,
                    return_dict=False,
                )

        noise_pred = output[0][0]
        noise_pred = -noise_pred.squeeze(1)

        latents_for_step = latents.squeeze(1)
        latents = scheduler.step(noise_pred.float(), t, latents_for_step.float(), return_dict=False)[0]
        latents = latents.unsqueeze(1).to(dtype)

    # Decode
    print("  Decoding...")
    latents_for_decode = latents.squeeze(1).unsqueeze(0)
    latents_for_decode = (latents_for_decode / vae.config.scaling_factor) + vae.config.shift_factor

    with torch.no_grad():
        image = vae.decode(latents_for_decode, return_dict=False)[0]

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image[0] * 255).astype("uint8")
    return Image.fromarray(image)


def main():
    print("="*60)
    print("Z-Image Omni with Bagel Connector - Full Generation Test")
    print("="*60)

    # Load pipeline
    components = load_pipeline_with_bagel()

    # Test 1: Basic generation (no reference)
    print("\n" + "="*60)
    print("Test 1: Basic generation (text-only)")
    print("="*60)

    image = generate(
        components,
        prompt="A red apple on a wooden table, photorealistic",
        reference_image=None,
        height=512,
        width=512,
        num_steps=4,
        device=DEVICE,
    )
    image.save(OUTPUT_DIR / "bagel_test_basic.png")
    print(f"  Saved to {OUTPUT_DIR / 'bagel_test_basic.png'}")

    # Test 2: With reference image (omni mode)
    print("\n" + "="*60)
    print("Test 2: Omni generation (with reference)")
    print("="*60)

    # Create a colorful reference image
    ref_image = Image.new("RGB", (384, 384), (255, 100, 50))

    image = generate(
        components,
        prompt="A similar scene with warm orange tones",
        reference_image=ref_image,
        height=512,
        width=512,
        num_steps=4,
        device=DEVICE,
    )
    image.save(OUTPUT_DIR / "bagel_test_omni.png")
    print(f"  Saved to {OUTPUT_DIR / 'bagel_test_omni.png'}")

    # Test 3: With a real image if available
    real_ref_path = OUTPUT_DIR / "test_reference.jpg"
    if real_ref_path.exists():
        print("\n" + "="*60)
        print("Test 3: Omni with real reference image")
        print("="*60)

        ref_image = Image.open(real_ref_path).convert("RGB")
        image = generate(
            components,
            prompt="A variation of this image",
            reference_image=ref_image,
            height=512,
            width=512,
            num_steps=4,
            device=DEVICE,
        )
        image.save(OUTPUT_DIR / "bagel_test_real_ref.png")
        print(f"  Saved to {OUTPUT_DIR / 'bagel_test_real_ref.png'}")

    print("\n" + "="*60)
    print("DONE - Check the output images!")
    print("="*60)


if __name__ == "__main__":
    main()
