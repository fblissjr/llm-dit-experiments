"""
Trace memory through the full pipeline to identify OOM source.

Last Updated: 2026-01-19

Tests each stage's memory usage to find where the 23GB spike happens.
"""
import gc
import torch


def get_memory():
    """Get current GPU memory in GB."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0


def log_memory(stage: str):
    """Log memory at a specific stage."""
    allocated, reserved = get_memory()
    print(f"[{stage}] allocated={allocated:.2f}GB, reserved={reserved:.2f}GB")


def cleanup():
    """Full memory cleanup."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    print("=== Full Pipeline Memory Trace ===\n")
    log_memory("Initial")

    # =========================================================================
    # Stage 1: Text Encoder (like the pipeline does)
    # =========================================================================
    print("\n--- STAGE 1: Text Encoder ---")
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    print("Loading text encoder...")
    text_encoder = Gemma3Encoder(
        model_id="models/LTX-2/text_encoder",
        device="cuda",
        dtype=torch.bfloat16,
        load_in_8bit=True,
    )
    log_memory("After text encoder load")

    print("Encoding prompt...")
    encoding_output = text_encoder.encode(["A cat walking"])
    prompt_embeds = encoding_output.embeddings[0].unsqueeze(0)
    attention_mask = encoding_output.attention_masks[0].unsqueeze(0)
    print(f"Embeddings shape: {prompt_embeds.shape}")
    log_memory("After encoding")

    # Keep embeddings, unload encoder
    prompt_embeds = prompt_embeds.to("cuda", torch.bfloat16)
    attention_mask = attention_mask.to("cuda")

    del text_encoder
    cleanup()
    log_memory("After encoder unload")

    # =========================================================================
    # Stage 2: Transformer + Connectors
    # =========================================================================
    print("\n--- STAGE 2: Transformer ---")
    from llm_dit.models.ltx2.loader import load_ltx2_transformer_fp8_native

    print("Loading transformer (FP8 native)...")
    model = load_ltx2_transformer_fp8_native(
        "models/LTX-2/transformer",
        dtype=torch.bfloat16,
        device="cuda"
    )
    model.eval()
    log_memory("After transformer load")

    print("\n--- Loading Connectors ---")
    from llm_dit.models.ltx2.connectors import load_ltx2_connectors

    connectors = load_ltx2_connectors(
        "models/LTX-2/connectors",
        device="cuda",
        dtype=torch.bfloat16,
    )
    log_memory("After connectors load")

    # =========================================================================
    # Stage 3: Process embeddings through connectors
    # =========================================================================
    print("\n--- Processing through connectors ---")
    print(f"Input embeddings: {prompt_embeds.shape}")

    # Run connectors
    video_embeds, _, _ = connectors(
        prompt_embeds,
        attention_mask,
        additive_mask=False,
    )
    print(f"Output embeddings: {video_embeds.shape}")
    log_memory("After connector forward")

    # =========================================================================
    # Stage 4: Single transformer forward pass
    # =========================================================================
    print("\n--- Single forward pass ---")

    # Create inputs for smoke test dimensions
    from llm_dit.models.ltx2.components import Modality
    from llm_dit.pipelines.generate import create_position_indices

    num_frames = 9
    height = 256
    width = 384
    t_latent = (num_frames - 1) // 8 + 1  # 2
    h_latent = height // 32  # 8
    w_latent = width // 32  # 12
    num_tokens = t_latent * h_latent * w_latent  # 192

    latent = torch.randn(1, num_tokens, 128, device="cuda", dtype=torch.bfloat16)
    timestep = torch.ones(1, num_tokens, device="cuda", dtype=torch.bfloat16) * 500  # Mid timestep
    positions = create_position_indices(1, num_frames, height, width, torch.device("cuda"))

    video_modality = Modality(
        latent=latent,
        timesteps=timestep,
        positions=positions,
        context=video_embeds,  # Use connector output
        enabled=True,
        context_mask=None,
    )

    torch.cuda.reset_peak_memory_stats()
    log_memory("Before forward")

    with torch.no_grad():
        output, _ = model(video=video_modality)

    peak = torch.cuda.max_memory_allocated() / 1024**3
    log_memory("After forward")
    print(f"Peak memory during forward: {peak:.2f}GB")

    # =========================================================================
    # Stage 5: CFG scenario (two forward passes)
    # =========================================================================
    print("\n--- CFG test (2 forward passes) ---")
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        # Unconditional
        uncond_modality = Modality(
            latent=latent,
            timesteps=timestep,
            positions=positions,
            context=torch.zeros_like(video_embeds),
            enabled=True,
            context_mask=None,
        )
        velocity_uncond, _ = model(video=uncond_modality)
        del uncond_modality
        log_memory("After uncond forward")

        # Conditional
        cond_modality = Modality(
            latent=latent,
            timesteps=timestep,
            positions=positions,
            context=video_embeds,
            enabled=True,
            context_mask=None,
        )
        velocity_cond, _ = model(video=cond_modality)
        del cond_modality
        log_memory("After cond forward")

        # CFG blend
        velocity = velocity_cond + 2.0 * (velocity_cond - velocity_uncond)  # CFG 3.0
        del velocity_uncond, velocity_cond

    peak = torch.cuda.max_memory_allocated() / 1024**3
    log_memory("After CFG")
    print(f"Peak memory during CFG: {peak:.2f}GB")

    # =========================================================================
    # Cleanup
    # =========================================================================
    print("\n--- Cleanup ---")
    del model, connectors, prompt_embeds, attention_mask, video_embeds, latent
    cleanup()
    log_memory("Final")


if __name__ == "__main__":
    main()
