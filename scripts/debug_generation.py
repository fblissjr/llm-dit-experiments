"""
Debug the generation pipeline to find why output is black.

Last Updated: 2026-01-19

Checks:
1. Sigma schedule values
2. Latent statistics before/after denoising
3. VAE input/output statistics
"""
import torch
from pathlib import Path


def main():
    print("=== Generation Debug ===\n")

    from llm_dit.pipelines.generate import GenerationConfig
    from llm_dit.schedulers.ltx2_scheduler import LTX2Scheduler

    # Same config as smoke test
    config = GenerationConfig(
        num_frames=9,
        height=256,
        width=384,
        num_inference_steps=8,
        guidance_scale=3.0,
        seed=10,
    )

    print(f"Config: {config.num_frames} frames @ {config.height}x{config.width}")
    print(f"Latent dims: {config.latent_dims}")
    print(f"Num tokens: {config.num_tokens}")

    # Check scheduler
    print("\n--- Scheduler ---")
    scheduler = LTX2Scheduler()
    t_latent, h_latent, w_latent = config.latent_dims
    mock_latent = torch.empty(1, 128, t_latent, h_latent, w_latent)

    sigmas = scheduler.execute(
        steps=config.num_inference_steps,
        latent=mock_latent,
        max_shift=config.max_shift,
        base_shift=config.base_shift,
        stretch=config.stretch,
        terminal=config.terminal,
    )

    print(f"Sigmas shape: {sigmas.shape}")
    print(f"Sigmas: {sigmas.tolist()}")
    print(f"Sigma range: {sigmas.min():.4f} to {sigmas.max():.4f}")

    # Test timesteps
    print("\n--- Timesteps (sigma * 1000) ---")
    timesteps = sigmas * 1000
    print(f"Timesteps: {timesteps.tolist()}")

    # Now load encoder and test embeddings
    print("\n--- Loading encoder to check embeddings ---")
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    encoder = Gemma3Encoder(
        model_id="models/LTX-2/text_encoder",
        device="cuda",
        dtype=torch.bfloat16,
        load_in_8bit=True,
    )

    prompt = "A cat walking"
    output = encoder.encode([prompt])
    embeddings = output.embeddings[0]
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embeddings: mean={embeddings.float().mean():.4f}, std={embeddings.float().std():.4f}")
    print(f"Embeddings: min={embeddings.float().min():.4f}, max={embeddings.float().max():.4f}")

    # Clean up encoder
    del encoder
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Now test transformer forward with one step
    print("\n--- Loading transformer to test one step ---")
    from llm_dit.models.ltx2 import load_ltx2_transformer_fp8_native
    from llm_dit.models.ltx2.modality import Modality
    from llm_dit.pipelines.generate import create_position_indices

    model = load_ltx2_transformer_fp8_native(
        "models/LTX-2/transformer",
        dtype=torch.bfloat16,
        device="cuda",
        video_only=True,
        verbose=True,
    )

    # Create test inputs
    num_tokens = config.num_tokens
    latents = torch.randn(1, num_tokens, 128, device="cuda", dtype=torch.bfloat16)
    print(f"\nInitial latents: mean={latents.float().mean():.4f}, std={latents.float().std():.4f}")

    # Timestep for first step
    sigma = sigmas[0]
    timestep = (sigma * 1000).expand(1, num_tokens).to("cuda", torch.bfloat16)
    print(f"Timestep: {timestep[0, 0]:.2f}")

    # Positions
    positions = create_position_indices(1, config.num_frames, config.height, config.width, "cuda")
    print(f"Positions shape: {positions.shape}")

    # Embeddings (ensure on GPU and right dtype)
    prompt_embeds = embeddings.unsqueeze(0).to("cuda", torch.bfloat16)
    print(f"Prompt embeds shape: {prompt_embeds.shape}")

    # Create modality
    modality = Modality(
        latent=latents,
        timesteps=timestep,
        positions=positions,
        context=prompt_embeds,
    )

    # Forward pass (set to inference mode)
    print("\n--- Forward pass ---")
    model.train(False)  # Set to evaluation mode
    with torch.no_grad():
        velocity, _ = model(video=modality)

    print(f"Velocity shape: {velocity.shape}")
    print(f"Velocity: mean={velocity.float().mean():.4f}, std={velocity.float().std():.4f}")
    print(f"Velocity: min={velocity.float().min():.4f}, max={velocity.float().max():.4f}")

    # Euler step
    dt = sigmas[1] - sigmas[0]
    denoised = latents + velocity * dt.to("cuda", torch.bfloat16)
    print(f"\nAfter 1 Euler step:")
    print(f"Denoised: mean={denoised.float().mean():.4f}, std={denoised.float().std():.4f}")

    # Clean up
    del model, latents, velocity, denoised
    gc.collect()
    torch.cuda.empty_cache()

    print("\n--- Loading VAE to test decode ---")
    from llm_dit.models.ltx2.vae import load_ltx2_vae_decoder

    # Create some test latents (simulating denoised output)
    test_latents = torch.randn(1, 128, t_latent, h_latent, w_latent, device="cuda", dtype=torch.bfloat16)
    print(f"Test latents shape: {test_latents.shape}")
    print(f"Test latents: mean={test_latents.float().mean():.4f}, std={test_latents.float().std():.4f}")

    vae = load_ltx2_vae_decoder("models/LTX-2/vae", dtype=torch.bfloat16, device="cpu").to("cuda")

    with torch.no_grad():
        video = vae(test_latents)

    print(f"Video shape: {video.shape}")
    print(f"Video (raw): mean={video.float().mean():.4f}, std={video.float().std():.4f}")
    print(f"Video (raw): min={video.float().min():.4f}, max={video.float().max():.4f}")

    # Convert to uint8
    video_uint8 = ((video + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
    print(f"Video (uint8): mean={video_uint8.float().mean():.1f}, min={video_uint8.min()}, max={video_uint8.max()}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
