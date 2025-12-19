#!/usr/bin/env python3
"""
last updated: 2025-12-19

Full end-to-end test of Z-Image Omni generation.

This loads the actual Z-Image Turbo model and SigLIP2 encoder,
then runs generation with image conditioning.

Note: siglip_embedder/refiner are randomly initialized (not trained).
Results will be garbage but validate the architecture works.

Usage:
    uv run python experiments/siglip/test_full_omni_generation.py
"""

import sys
from pathlib import Path

# Add diffusers PR to path
DIFFUSERS_PR_PATH = Path(__file__).parent.parent.parent / "coderef/diffusers/src"
sys.path.insert(0, str(DIFFUSERS_PR_PATH))

import torch
from PIL import Image


# Model paths
ZIMAGE_PATH = Path.home() / "Storage/Tongyi-MAI_Z-Image-Turbo"
SIGLIP_PATH = Path.home() / "Storage/google_siglip2-so400m-patch14-384"  # 1152 hidden dim
QWEN3_PATH = Path.home() / "Storage/Qwen3-4B"


def load_pipeline():
    """Load Z-Image Omni pipeline with SigLIP support."""
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
    from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
    from transformers import AutoModel, AutoTokenizer, AutoProcessor

    print("Loading components...")

    # Load VAE
    print("  Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        ZIMAGE_PATH,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )

    # Load text encoder
    print("  Loading text encoder (Qwen3-4B)...")
    text_encoder = AutoModel.from_pretrained(
        QWEN3_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_PATH, trust_remote_code=True)

    # Load scheduler
    print("  Loading scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        ZIMAGE_PATH,
        subfolder="scheduler",
    )

    # Load SigLIP
    print("  Loading SigLIP...")
    siglip_full = AutoModel.from_pretrained(SIGLIP_PATH)
    siglip = siglip_full.vision_model
    siglip_processor = AutoProcessor.from_pretrained(SIGLIP_PATH)
    siglip_feat_dim = siglip.config.hidden_size

    # Load transformer
    print("  Loading transformer...")
    # First load the base config
    transformer = ZImageTransformer2DModel.from_pretrained(
        ZIMAGE_PATH,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )

    # Check if siglip components exist
    if transformer.siglip_embedder is None:
        print(f"  Adding SigLIP components (siglip_feat_dim={siglip_feat_dim})...")
        # Need to reinitialize with siglip_feat_dim
        # This is the tricky part - we need to add the siglip components
        # while keeping the existing weights

        # Get current config
        config = transformer.config

        # Create new transformer with siglip support
        new_config = dict(config)
        new_config["siglip_feat_dim"] = siglip_feat_dim

        from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel as ZImageTransformer
        new_transformer = ZImageTransformer(**new_config)

        # Copy existing weights
        print("  Copying existing weights to new transformer...")
        missing, unexpected = new_transformer.load_state_dict(transformer.state_dict(), strict=False)
        print(f"    Missing keys (expected - siglip components): {len(missing)}")
        print(f"    Unexpected keys: {len(unexpected)}")

        # Initialize siglip components with small random values
        print("  Initializing SigLIP components with random weights...")
        for name, param in new_transformer.named_parameters():
            if "siglip" in name:
                if param.dim() >= 2:
                    torch.nn.init.xavier_uniform_(param)
                else:
                    torch.nn.init.zeros_(param)

        transformer = new_transformer

    transformer = transformer.to(torch.bfloat16)

    print("All components loaded!")

    return {
        "vae": vae,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
        "transformer": transformer,
        "siglip": siglip,
        "siglip_processor": siglip_processor,
    }


def encode_prompt(tokenizer, text_encoder, prompt: str, device: str = "cpu"):
    """Encode a text prompt."""
    # Simple format (without vision tokens for now)
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(
        formatted,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        embeddings = outputs.hidden_states[-2]

    # Extract non-padding tokens
    mask = attention_mask.bool()
    embeddings = embeddings[0][mask[0]]

    return embeddings


def encode_siglip(siglip, siglip_processor, image: Image.Image, device: str = "cpu"):
    """Encode an image with SigLIP."""
    inputs = siglip_processor(images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = siglip(pixel_values=pixel_values)
        hidden_state = outputs.last_hidden_state

    # Reshape to (H, W, C)
    B, N, C = hidden_state.shape
    H = W = int(N ** 0.5)
    hidden_state = hidden_state.squeeze(0).view(H, W, C)

    return hidden_state


def encode_image_latent(vae, image: Image.Image, device: str = "cpu"):
    """Encode an image to VAE latent."""
    from diffusers.image_processor import VaeImageProcessor

    processor = VaeImageProcessor(vae_scale_factor=16)
    image_tensor = processor.preprocess(image)
    image_tensor = image_tensor.to(device=device, dtype=vae.dtype)

    with torch.no_grad():
        latent = vae.encode(image_tensor).latent_dist.mode()
        latent = (latent - vae.config.shift_factor) * vae.config.scaling_factor

    # Add frame dimension: (B, C, H, W) -> (C, F, H, W)
    latent = latent.squeeze(0).unsqueeze(1)

    return latent


def run_generation(
    components: dict,
    prompt: str,
    reference_image: Image.Image = None,
    height: int = 512,
    width: int = 512,
    num_steps: int = 4,
    device: str = "cuda",
):
    """Run image generation."""
    vae = components["vae"].to(device)
    text_encoder = components["text_encoder"].to(device)
    transformer = components["transformer"].to(device)
    siglip = components["siglip"].to(device)
    scheduler = components["scheduler"]
    tokenizer = components["tokenizer"]
    siglip_processor = components["siglip_processor"]

    dtype = torch.bfloat16

    print(f"\nGenerating {width}x{height} image...")
    print(f"  Prompt: {prompt}")
    print(f"  Reference image: {'Yes' if reference_image else 'No'}")
    print(f"  Steps: {num_steps}")

    # Encode prompt
    print("  Encoding prompt...")
    cap_feats = encode_prompt(tokenizer, text_encoder, prompt, device)
    cap_feats = cap_feats.to(dtype)

    # Prepare latent shape
    latent_h = height // 16
    latent_w = width // 16

    # Create noise latent (target)
    print("  Creating noise latent...")
    latents = torch.randn(16, 1, latent_h, latent_w, device=device, dtype=dtype)

    # Prepare conditioning
    if reference_image:
        print("  Encoding reference image...")
        # Resize reference to match output
        ref_resized = reference_image.resize((width, height))

        # Get VAE latent of reference
        cond_latent = encode_image_latent(vae, ref_resized, device).to(dtype)

        # Get SigLIP embedding
        siglip_emb = encode_siglip(siglip, siglip_processor, reference_image, device).to(dtype)

        # Format for transformer
        cond_latents = [[cond_latent]]
        siglip_feats = [[siglip_emb, None]]  # None for target placeholder
        cap_feats_list = [[cap_feats[:30], cap_feats[30:]]]  # Split for vision tokens
    else:
        cond_latents = None
        siglip_feats = None
        cap_feats_list = [cap_feats]

    # Set up scheduler
    print("  Setting up scheduler...")

    # Calculate shift for this resolution
    image_seq_len = latent_h * latent_w // 4  # 2x2 patches
    mu = 0.5 + (1.15 - 0.5) * (image_seq_len - 256) / (4096 - 256)
    mu = max(0.5, min(1.15, mu))

    scheduler.set_timesteps(num_steps, device=device, mu=mu)
    timesteps = scheduler.timesteps

    # Denoising loop
    print("  Denoising...")
    for i, t in enumerate(timesteps):
        print(f"    Step {i+1}/{num_steps}...")

        timestep = t.expand(1)
        timestep_norm = (1000 - timestep) / 1000

        # Prepare inputs
        x = [latents]

        # Forward pass
        with torch.no_grad():
            if cond_latents is not None:
                output = transformer(
                    x=x,
                    t=timestep_norm,
                    cap_feats=cap_feats_list,
                    cond_latents=cond_latents,
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
        noise_pred = -noise_pred.squeeze(1)  # Remove frame dim

        # Scheduler step
        latents_for_step = latents.squeeze(1)
        latents = scheduler.step(noise_pred.float(), t, latents_for_step.float(), return_dict=False)[0]
        latents = latents.unsqueeze(1).to(dtype)

    # Decode using installed diffusers VAE (not PR version)
    print("  Decoding...")
    # latents is (C, F, H, W) = (16, 1, 32, 32)
    # VAE expects (B, C, H, W) = (1, 16, 32, 32)
    latents_for_decode = latents.squeeze(1).unsqueeze(0)  # (16, 1, H, W) -> (16, H, W) -> (1, 16, H, W)
    latents_for_decode = (latents_for_decode / vae.config.scaling_factor) + vae.config.shift_factor

    # Import the installed diffusers VAE decode function
    import diffusers.models.autoencoders.autoencoder_kl as installed_vae
    with torch.no_grad():
        # Use the decoder directly to avoid potential PR version issues
        try:
            image = vae.decode(latents_for_decode, return_dict=False)[0]
        except RuntimeError as e:
            # Fallback: just return the latents as grayscale for debugging
            print(f"    VAE decode failed, returning latent visualization")

            # Debug: check latent shape
            print(f"    latents_for_decode: shape={latents_for_decode.shape}, is_sparse={latents_for_decode.is_sparse}")

            # latents_for_decode shape: (1, 16, H, W)
            latent_cpu = latents_for_decode.detach().clone().cpu().float()
            latent_vis = latent_cpu[0, 0:3]  # (3, H, W)
            print(f"    latent_vis shape: {latent_vis.shape}")

            latent_vis = (latent_vis - latent_vis.min()) / (latent_vis.max() - latent_vis.min() + 1e-8)

            # Convert to numpy properly
            import numpy as np
            latent_np = latent_vis.numpy()  # (3, H, W)
            latent_np = np.transpose(latent_np, (1, 2, 0))  # (H, W, 3)
            latent_np = (latent_np * 255).astype(np.uint8)

            # Resize to target size
            from PIL import Image as PILImage
            image = PILImage.fromarray(latent_np).resize((width, height), PILImage.BILINEAR)
            return image

    # Post-process
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image[0] * 255).astype("uint8")
    image = Image.fromarray(image)

    return image


def main():
    print("=" * 60)
    print("Z-Image Omni Full Generation Test")
    print("=" * 60)

    # Check paths
    if not ZIMAGE_PATH.exists():
        print(f"Error: Z-Image model not found at {ZIMAGE_PATH}")
        return

    if not SIGLIP_PATH.exists():
        print(f"Error: SigLIP model not found at {SIGLIP_PATH}")
        return

    if not QWEN3_PATH.exists():
        print(f"Error: Qwen3-4B not found at {QWEN3_PATH}")
        return

    # Load components
    print("\n1. Loading pipeline components...")
    try:
        components = load_pipeline()
    except Exception as e:
        print(f"Error loading pipeline: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test without reference image first (basic mode)
    print("\n2. Testing basic generation (no reference)...")
    try:
        image = run_generation(
            components,
            prompt="A red apple on a wooden table",
            reference_image=None,
            height=512,
            width=512,
            num_steps=4,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        image.save("experiments/siglip/test_basic.png")
        print(f"  Saved to experiments/siglip/test_basic.png")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    # Test with reference image (omni mode)
    print("\n3. Testing omni generation (with reference)...")
    try:
        # Create a simple reference image
        ref_image = Image.new("RGB", (256, 256), (100, 150, 200))

        image = run_generation(
            components,
            prompt="A similar image with different colors",
            reference_image=ref_image,
            height=512,
            width=512,
            num_steps=4,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        image.save("experiments/siglip/test_omni.png")
        print(f"  Saved to experiments/siglip/test_omni.png")
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print("""
Note: The SigLIP components (siglip_embedder, siglip_refiner) are
randomly initialized. The omni mode output will be garbage until
proper trained weights are available.

The basic mode (without reference image) should produce normal output
since it uses the original Z-Image weights.
""")


if __name__ == "__main__":
    main()
