"""
LTX-2.3 Forward Pass Diagnostics -- verify fp8-cast model produces sane outputs.

Last Updated: 2026-03-08

Loads the actual V2.3 fp8-cast transformer from models/LTX-2.3/ using the same
code path as generate.py (load_ltx2_transformer_fp8_cast + amend_forward_with_upcast),
then runs a single forward pass and checks output statistics for NaN, Inf, and
reasonable magnitudes.

This is the minimal diagnostic needed to distinguish "model math is correct but
weights are garbled" from "model math is wrong".

Run with: uv run pytest tests/integration/test_ltx23_forward_diagnostic.py -v -s
"""

import gc
from pathlib import Path

import pytest
import torch

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

MODEL_PATH = Path("models/LTX-2.3")
TRANSFORMER_FILE = "ltx-2.3-transformer-fp8.safetensors"
CONNECTORS_FILE = "ltx-2.3-connectors.safetensors"


def models_available() -> bool:
    """Check if LTX-2.3 split model files exist."""
    return (MODEL_PATH / TRANSFORMER_FILE).exists()


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _tensor_stats(t: torch.Tensor) -> dict:
    """Compute diagnostic statistics for a tensor."""
    t_f = t.float()
    return {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
        "nan_count": int(torch.isnan(t_f).sum()),
        "inf_count": int(torch.isinf(t_f).sum()),
        "mean": float(t_f.mean()),
        "std": float(t_f.std()),
        "min": float(t_f.min()),
        "max": float(t_f.max()),
        "abs_mean": float(t_f.abs().mean()),
    }


def _print_stats(name: str, stats: dict):
    """Print formatted tensor statistics."""
    print(f"\n  {name}:")
    print(f"    shape={stats['shape']} dtype={stats['dtype']}")
    print(f"    NaN={stats['nan_count']} Inf={stats['inf_count']}")
    print(f"    mean={stats['mean']:.6f} std={stats['std']:.6f}")
    print(f"    min={stats['min']:.6f} max={stats['max']:.6f}")
    print(f"    abs_mean={stats['abs_mean']:.6f}")


@pytest.mark.skipif(not models_available(), reason="LTX-2.3 models not found")
class TestTransformerFP8CastForward:
    """Load the real fp8-cast transformer and run a diagnostic forward pass."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        cleanup_gpu()
        yield
        cleanup_gpu()

    def test_load_and_forward_video_only(self):
        """Load fp8-cast model (video-only), run forward, check output stats.

        This mimics the reconstruction path from _reconstruct_transformer_from_cache
        but loads fresh from disk to test the complete loading pipeline.
        """
        from llm_dit.models.ltx2 import Modality, load_ltx2_transformer_fp8_cast
        from llm_dit.quantization.fp8_cast import amend_forward_with_upcast

        tf_path = MODEL_PATH / TRANSFORMER_FILE

        print("\n--- Loading fp8-cast transformer (video-only) ---")
        model = load_ltx2_transformer_fp8_cast(
            tf_path, dtype=torch.bfloat16, device="cpu", video_only=True,
        )

        # Patch forwards for per-forward upcast (same as _reconstruct_transformer_from_cache)
        patched = amend_forward_with_upcast(model)
        print(f"Patched {patched} linear layers for per-forward upcast")

        # Move to GPU
        model = model.to("cuda")

        # Verify fp8 weights survived
        fp8_count = sum(1 for p in model.parameters() if p.dtype == torch.float8_e4m3fn)
        bf16_count = sum(1 for p in model.parameters() if p.dtype == torch.bfloat16)
        total = sum(1 for _ in model.parameters())
        print(f"Parameters: {total} total, {fp8_count} fp8, {bf16_count} bf16")
        assert fp8_count > 0, "No fp8 parameters found -- loading is broken"

        mem_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory: {mem_gb:.2f} GB")

        # Create synthetic inputs matching a small generation config
        # 9 frames, 384x256 -> t=2, h=12, w=8 -> 192 tokens
        batch_size = 1
        num_tokens = 192
        latent_dim = 128
        context_dim = 4096  # Post-projection dim (FeatureExtractorV2 output)
        context_len = 50

        sigma_value = 0.9  # High sigma = early in denoising (near pure noise)

        with torch.no_grad():
            latent = torch.randn(batch_size, num_tokens, latent_dim,
                                 dtype=torch.bfloat16, device="cuda")
            # Per-token timesteps: sigma * ones
            timesteps = torch.full((batch_size, num_tokens), sigma_value,
                                   dtype=torch.bfloat16, device="cuda")
            # Simple position indices [B, 3, T, 2]
            positions = torch.zeros(batch_size, 3, num_tokens, 2,
                                    dtype=torch.bfloat16, device="cuda")
            # Fill with plausible position values
            t_latent, h_latent, w_latent = 2, 12, 8
            for token_idx in range(num_tokens):
                t_idx = token_idx // (h_latent * w_latent)
                h_idx = (token_idx % (h_latent * w_latent)) // w_latent
                w_idx = token_idx % w_latent
                # Temporal: in seconds (divided by fps=24)
                positions[0, 0, token_idx, 0] = t_idx / 24.0
                positions[0, 0, token_idx, 1] = (t_idx + 1) / 24.0
                # Spatial: in pixel-space (multiplied by vae factor 32)
                positions[0, 1, token_idx, 0] = h_idx * 32
                positions[0, 1, token_idx, 1] = (h_idx + 1) * 32
                positions[0, 2, token_idx, 0] = w_idx * 32
                positions[0, 2, token_idx, 1] = (w_idx + 1) * 32

            # Random context (simulating text encoder output)
            context = torch.randn(batch_size, context_len, context_dim,
                                  dtype=torch.bfloat16, device="cuda")
            # Scale context to reasonable magnitude (Gemma3 outputs are ~O(1))
            context = context * 0.1

            sigma = torch.tensor([sigma_value], dtype=torch.bfloat16, device="cuda")

            video = Modality(
                latent=latent,
                sigma=sigma,
                timesteps=timesteps,
                positions=positions,
                context=context,
                enabled=True,
            )

            print("\n--- Running forward pass ---")
            output, _ = model(video=video)

        # Diagnostic output
        input_stats = _tensor_stats(latent)
        output_stats = _tensor_stats(output)

        print("\n--- Input Statistics ---")
        _print_stats("latent (input)", input_stats)

        print("\n--- Output Statistics ---")
        _print_stats("velocity (output)", output_stats)

        # Assertions
        assert output.shape == latent.shape, (
            f"Output shape {output.shape} != input shape {latent.shape}"
        )
        assert output_stats["nan_count"] == 0, (
            f"Output contains {output_stats['nan_count']} NaN values"
        )
        assert output_stats["inf_count"] == 0, (
            f"Output contains {output_stats['inf_count']} Inf values"
        )
        # Output should have meaningful magnitude (not near-zero noise)
        assert output_stats["abs_mean"] > 0.001, (
            f"Output abs_mean={output_stats['abs_mean']:.6f} is suspiciously low"
        )
        # Output should not be astronomically large (would indicate broken scaling)
        assert output_stats["abs_mean"] < 1000.0, (
            f"Output abs_mean={output_stats['abs_mean']:.6f} is suspiciously high"
        )
        # Output should differ from input (model actually did something)
        diff = (output - latent).float().abs().mean().item()
        print(f"\n  |output - input| mean: {diff:.6f}")
        assert diff > 0.01, f"Output nearly identical to input (diff={diff:.6f})"

        print("\n--- PASS: fp8-cast forward produces sane output ---")

        del model, output
        cleanup_gpu()

    def test_load_and_forward_av_mode(self):
        """Load fp8-cast model in AV mode, run forward with both modalities.

        This tests the actual dual-stream path used by generate_video_two_stage.
        """
        from llm_dit.models.ltx2 import Modality, load_ltx2_transformer_fp8_cast
        from llm_dit.quantization.fp8_cast import amend_forward_with_upcast

        tf_path = MODEL_PATH / TRANSFORMER_FILE

        print("\n--- Loading fp8-cast transformer (AV mode) ---")
        model = load_ltx2_transformer_fp8_cast(
            tf_path, dtype=torch.bfloat16, device="cpu", video_only=False,
        )
        patched = amend_forward_with_upcast(model)
        print(f"Patched {patched} linear layers")
        model = model.to("cuda")

        mem_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory: {mem_gb:.2f} GB")

        # Video: 9 frames, 384x256 -> 192 tokens
        # Audio: ~4 tokens for 0.375 seconds of audio
        batch_size = 1
        video_tokens = 192
        audio_tokens = 4
        video_dim = 128
        audio_dim = 128  # 8 channels * 16 mel_bins
        video_context_dim = 4096
        audio_context_dim = 2048
        context_len = 50
        sigma_value = 0.8

        with torch.no_grad():
            # Video modality
            v_latent = torch.randn(batch_size, video_tokens, video_dim,
                                   dtype=torch.bfloat16, device="cuda")
            v_timesteps = torch.full((batch_size, video_tokens), sigma_value,
                                    dtype=torch.bfloat16, device="cuda")
            v_positions = torch.zeros(batch_size, 3, video_tokens, 2,
                                     dtype=torch.bfloat16, device="cuda")
            v_context = torch.randn(batch_size, context_len, video_context_dim,
                                    dtype=torch.bfloat16, device="cuda") * 0.1
            v_sigma = torch.tensor([sigma_value], dtype=torch.bfloat16, device="cuda")

            video = Modality(
                latent=v_latent, sigma=v_sigma, timesteps=v_timesteps,
                positions=v_positions, context=v_context, enabled=True,
            )

            # Audio modality
            a_latent = torch.randn(batch_size, audio_tokens, audio_dim,
                                   dtype=torch.bfloat16, device="cuda")
            a_timesteps = torch.full((batch_size, audio_tokens), sigma_value,
                                    dtype=torch.bfloat16, device="cuda")
            a_positions = torch.zeros(batch_size, 1, audio_tokens, 2,
                                     dtype=torch.bfloat16, device="cuda")
            a_context = torch.randn(batch_size, context_len, audio_context_dim,
                                    dtype=torch.bfloat16, device="cuda") * 0.1
            a_sigma = torch.tensor([sigma_value], dtype=torch.bfloat16, device="cuda")

            audio = Modality(
                latent=a_latent, sigma=a_sigma, timesteps=a_timesteps,
                positions=a_positions, context=a_context, enabled=True,
            )

            print("\n--- Running AV forward pass ---")
            v_out, a_out = model(video=video, audio=audio)

        # Diagnostics
        v_stats = _tensor_stats(v_out)
        a_stats = _tensor_stats(a_out)

        print("\n--- Video Output Statistics ---")
        _print_stats("video velocity", v_stats)
        print("\n--- Audio Output Statistics ---")
        _print_stats("audio velocity", a_stats)

        # Assertions
        assert v_out.shape == v_latent.shape
        assert a_out.shape == a_latent.shape

        for name, stats in [("video", v_stats), ("audio", a_stats)]:
            assert stats["nan_count"] == 0, f"{name} output has NaN"
            assert stats["inf_count"] == 0, f"{name} output has Inf"
            assert stats["abs_mean"] > 0.001, f"{name} output near zero"
            assert stats["abs_mean"] < 1000.0, f"{name} output too large"

        print("\n--- PASS: AV forward produces sane output ---")

        del model, v_out, a_out
        cleanup_gpu()


@pytest.mark.skipif(not models_available(), reason="LTX-2.3 models not found")
class TestCacheReconstructionForward:
    """Test the full cache->reconstruct->forward path used in production."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        cleanup_gpu()
        yield
        cleanup_gpu()

    def test_reconstruct_from_cache_and_forward(self):
        """Reconstruct model from cached state dict, run forward, check output.

        This is the exact code path that generate_video_two_stage uses:
        1. load_ltx2_transformer_fp8_cast -> extract state_dict -> pin
        2. create_model_from_config + load_state_dict(assign=True)
        3. amend_forward_with_upcast
        4. forward pass
        """
        from llm_dit.models.ltx2 import Modality, load_ltx2_transformer_fp8_cast
        from llm_dit.models.ltx2.loader import LTXModelType, create_model_from_config, load_config
        from llm_dit.quantization.fp8_cast import amend_forward_with_upcast
        from llm_dit.utils.meta_init import meta_init

        tf_path = MODEL_PATH / TRANSFORMER_FILE

        # Step 1: Load model, extract state dict, pin memory (same as _preload_ltx2_transformer)
        print("\n--- Step 1: Loading and caching state dict ---")
        model = load_ltx2_transformer_fp8_cast(
            tf_path, dtype=torch.bfloat16, device="cpu", video_only=True,
        )
        config = load_config(tf_path)

        sd = {}
        for k, v in model.state_dict().items():
            sd[k] = v.pin_memory()
        del model

        cache = {"config": config, "state_dict": sd, "video_only": True, "fp8_cast": True}

        fp8_tensors = sum(1 for v in sd.values() if v.dtype == torch.float8_e4m3fn)
        bf16_tensors = sum(1 for v in sd.values() if v.dtype == torch.bfloat16)
        print(f"Cached: {len(sd)} tensors ({fp8_tensors} fp8, {bf16_tensors} bf16)")

        # Step 2: Reconstruct from cache (same as _reconstruct_transformer_from_cache)
        print("\n--- Step 2: Reconstructing from cache ---")
        with meta_init():
            reconstructed = create_model_from_config(
                cache["config"], torch.bfloat16, model_type=LTXModelType.VideoOnly,
            )
        reconstructed.load_state_dict(cache["state_dict"], assign=True)

        # Step 3: Patch forwards
        patched = amend_forward_with_upcast(reconstructed)
        print(f"Patched {patched} linear layers")

        # Verify fp8 weights survived load_state_dict + assign
        fp8_params = sum(1 for p in reconstructed.parameters() if p.dtype == torch.float8_e4m3fn)
        print(f"fp8 parameters after reconstruction: {fp8_params}")
        assert fp8_params > 0, "fp8 parameters lost during reconstruction"

        reconstructed = reconstructed.to("cuda")
        mem_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory after reconstruction: {mem_gb:.2f} GB")

        # Step 4: Forward pass
        print("\n--- Step 3: Forward pass ---")
        batch_size = 1
        num_tokens = 96  # Small: 1 temporal * 12 * 8
        sigma_value = 0.7

        with torch.no_grad():
            latent = torch.randn(batch_size, num_tokens, 128,
                                 dtype=torch.bfloat16, device="cuda")
            timesteps = torch.full((batch_size, num_tokens), sigma_value,
                                   dtype=torch.bfloat16, device="cuda")
            positions = torch.zeros(batch_size, 3, num_tokens, 2,
                                    dtype=torch.bfloat16, device="cuda")
            context = torch.randn(batch_size, 50, 4096,
                                  dtype=torch.bfloat16, device="cuda") * 0.1
            sigma = torch.tensor([sigma_value], dtype=torch.bfloat16, device="cuda")

            video = Modality(
                latent=latent, sigma=sigma, timesteps=timesteps,
                positions=positions, context=context, enabled=True,
            )
            output, _ = reconstructed(video=video)

        stats = _tensor_stats(output)
        print("\n--- Reconstructed Model Output ---")
        _print_stats("velocity", stats)

        assert stats["nan_count"] == 0, "NaN in output"
        assert stats["inf_count"] == 0, "Inf in output"
        assert stats["abs_mean"] > 0.001, "Output near zero"
        assert stats["abs_mean"] < 1000.0, "Output too large"

        # Compare: direct-loaded vs cache-reconstructed should give same output
        # for identical inputs (deterministic fp8 upcast)
        print("\n--- PASS: Cache reconstruction produces sane output ---")

        del reconstructed, output, cache
        cleanup_gpu()


@pytest.mark.skipif(not models_available(), reason="LTX-2.3 models not found")
class TestEncoderDiagnostic:
    """Load the Gemma3 encoder with connectors, encode a prompt, check output."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        cleanup_gpu()
        yield
        cleanup_gpu()

    def test_encode_prompt_with_connectors(self):
        """Encode a prompt through Gemma3 + FeatureExtractorV2 + connectors.

        Verifies the full text encoding path produces embeddings with:
        - Correct shapes (video: 4096-dim, audio: 2048-dim)
        - No NaN/Inf
        - Reasonable magnitudes
        """
        encoder_path = MODEL_PATH / "text_encoder"
        connectors_path = MODEL_PATH / CONNECTORS_FILE

        if not encoder_path.exists():
            pytest.skip(f"Text encoder not found at {encoder_path}")
        if not connectors_path.exists():
            pytest.skip(f"Connectors not found at {connectors_path}")

        from llm_dit.encoders.gemma3_variants import create_gemma3_encoder

        print("\n--- Loading Gemma3 encoder with connectors ---")
        encoder = create_gemma3_encoder(
            variant="q4-qat",  # Default variant
            model_path=str(MODEL_PATH),
            text_encoder_path=str(encoder_path),
            device="cuda",
            dtype=torch.bfloat16,
            connectors_file=CONNECTORS_FILE,
        )

        prompt = "A golden retriever playing fetch in a sunny park"

        print(f"\n--- Encoding: '{prompt}' ---")
        # encoder.encode() takes a list of strings, returns EncodingOutput
        result = encoder.encode([prompt])

        print(f"Encoder returned type: {type(result).__name__}")

        # EncodingOutput has: embeddings, attention_masks, audio_embeddings, etc.
        # Video embeddings: [seq_len, 4096]
        assert hasattr(result, "embeddings"), "EncodingOutput missing .embeddings"
        assert len(result.embeddings) == 1, f"Expected 1 embedding, got {len(result.embeddings)}"

        video_embed = result.embeddings[0]
        video_stats = _tensor_stats(video_embed)
        print("\n--- Video Embeddings ---")
        _print_stats("video_embeds", video_stats)
        assert video_stats["nan_count"] == 0, "Video embeddings have NaN"
        assert video_stats["inf_count"] == 0, "Video embeddings have Inf"
        assert video_stats["abs_mean"] > 0.001, "Video embeddings near zero"
        assert video_embed.shape[-1] == 4096, (
            f"Expected 4096-dim video embeddings, got {video_embed.shape[-1]}"
        )

        # Attention mask
        assert hasattr(result, "attention_masks")
        mask = result.attention_masks[0]
        print(f"\n  attention_mask: shape={mask.shape}, sum={mask.sum().item()}")

        # Audio embeddings (from FeatureExtractorV2 dual projection)
        if hasattr(result, "audio_embeddings") and result.audio_embeddings is not None:
            audio_embed = result.audio_embeddings[0]
            audio_stats = _tensor_stats(audio_embed)
            print("\n--- Audio Embeddings ---")
            _print_stats("audio_embeds", audio_stats)
            assert audio_stats["nan_count"] == 0, "Audio embeddings have NaN"
            assert audio_stats["inf_count"] == 0, "Audio embeddings have Inf"
            assert audio_embed.shape[-1] == 2048, (
                f"Expected 2048-dim audio embeddings, got {audio_embed.shape[-1]}"
            )
        else:
            print("\n  (no audio embeddings returned)")

        # Also encode the negative prompt to verify it works
        from llm_dit.models.ltx2.constants import LTX2_DEFAULT_NEGATIVE_PROMPT
        neg_result = encoder.encode([LTX2_DEFAULT_NEGATIVE_PROMPT])
        neg_embed = neg_result.embeddings[0]
        neg_stats = _tensor_stats(neg_embed)
        print("\n--- Negative Prompt Embeddings ---")
        _print_stats("neg_embeds", neg_stats)
        assert neg_stats["nan_count"] == 0, "Neg embeddings have NaN"

        # Pos and neg should differ (different prompts -> different embeddings)
        if video_embed.shape == neg_embed.shape:
            diff = (video_embed - neg_embed).float().abs().mean().item()
            print(f"\n  |pos - neg| mean: {diff:.6f}")
            assert diff > 0.01, "Positive and negative embeddings are too similar"

        print("\n--- PASS: Encoder produces sane output ---")

        del encoder
        cleanup_gpu()


@pytest.mark.skipif(not models_available(), reason="LTX-2.3 models not found")
class TestDenoiseLoopDiagnostic:
    """Run a minimal 2-step denoising loop to check if latents converge or diverge."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        cleanup_gpu()
        yield
        cleanup_gpu()

    def test_two_step_denoise_convergence(self):
        """Run 2 Euler steps and verify latents don't diverge.

        Uses the distilled sigma schedule [1.0, ..., 0.0] and checks that
        latent magnitudes stay bounded. If they explode, it indicates a
        scaling/timestep bug.
        """
        from llm_dit.models.ltx2 import Modality, load_ltx2_transformer_fp8_cast
        from llm_dit.quantization.fp8_cast import amend_forward_with_upcast

        tf_path = MODEL_PATH / TRANSFORMER_FILE

        print("\n--- Loading transformer for denoising test ---")
        model = load_ltx2_transformer_fp8_cast(
            tf_path, dtype=torch.bfloat16, device="cpu", video_only=True,
        )
        amend_forward_with_upcast(model)
        model = model.to("cuda")

        # Use first 3 sigma values from distilled schedule: [1.0, 0.99375, 0.9875]
        # This gives 2 Euler steps
        sigmas = [1.0, 0.99375, 0.9875]

        batch_size = 1
        num_tokens = 96  # Small
        latent_dim = 128

        # Seed the initial noise
        generator = torch.Generator(device="cuda").manual_seed(42)
        latent = torch.randn(batch_size, num_tokens, latent_dim,
                             dtype=torch.bfloat16, device="cuda", generator=generator)
        context = torch.randn(batch_size, 50, 4096,
                              dtype=torch.bfloat16, device="cuda") * 0.1
        positions = torch.zeros(batch_size, 3, num_tokens, 2,
                                dtype=torch.bfloat16, device="cuda")

        print(f"\n--- Initial latent ---")
        _print_stats("latent_0", _tensor_stats(latent))

        with torch.no_grad():
            for step_idx in range(len(sigmas) - 1):
                sigma = sigmas[step_idx]
                sigma_next = sigmas[step_idx + 1]
                dt = sigma_next - sigma

                timesteps = torch.full((batch_size, num_tokens), sigma,
                                       dtype=torch.bfloat16, device="cuda")
                sigma_t = torch.tensor([sigma], dtype=torch.bfloat16, device="cuda")

                video = Modality(
                    latent=latent, sigma=sigma_t, timesteps=timesteps,
                    positions=positions, context=context, enabled=True,
                )
                velocity, _ = model(video=video)

                # Euler step
                latent = (latent.float() + velocity.float() * dt).to(latent.dtype)

                v_stats = _tensor_stats(velocity)
                l_stats = _tensor_stats(latent)

                print(f"\n--- Step {step_idx}: sigma {sigma:.4f} -> {sigma_next:.4f} ---")
                _print_stats(f"velocity_{step_idx}", v_stats)
                _print_stats(f"latent_{step_idx + 1}", l_stats)

                assert v_stats["nan_count"] == 0, f"Step {step_idx}: velocity has NaN"
                assert v_stats["inf_count"] == 0, f"Step {step_idx}: velocity has Inf"
                assert l_stats["nan_count"] == 0, f"Step {step_idx}: latent has NaN"
                assert l_stats["inf_count"] == 0, f"Step {step_idx}: latent has Inf"

                # Check latent magnitude stays bounded (not diverging)
                assert l_stats["abs_mean"] < 100.0, (
                    f"Step {step_idx}: latent diverging (abs_mean={l_stats['abs_mean']:.2f})"
                )

        print("\n--- PASS: Denoising loop produces stable output ---")

        del model, latent
        cleanup_gpu()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
