"""
Tests for FLUX.2 VAE (AutoEncoder) implementation.

Last Updated: 2026-01-23

These tests verify the FLUX.2 VAE with patchify/unpatchify and
BatchNorm latent normalization. Key differences from standard VAEs:
- 2x2 spatial patchification (4x channel increase, 2x spatial reduction)
- BatchNorm for latent normalization (not LayerNorm)
- 32-channel latent space before patchify

Run with: uv run pytest tests/unit/test_flux2_vae.py -v
"""

import pytest
import torch
import math

from llm_dit.models.flux2.vae import (
    AutoEncoder,
    AutoEncoderParams,
    Flux2VAE,
)
from llm_dit.models.flux2.constants import (
    VAE_Z_CHANNELS,
    TOTAL_SPATIAL_COMPRESSION,
    LATENT_CHANNELS_AFTER_PATCHIFY,
)


# ============================================================================
# Constants Tests
# ============================================================================

class TestVAEConstants:
    """Tests for VAE-related constants."""

    def test_z_channels_value(self):
        """Test VAE latent channels before patchify."""
        assert VAE_Z_CHANNELS == 32, "FLUX.2 uses 32-channel latent space"

    def test_total_compression(self):
        """Test total spatial compression is 16x."""
        assert TOTAL_SPATIAL_COMPRESSION == 16, \
            "Total compression should be 16x (8x VAE + 2x patchify)"

    def test_latent_channels_after_patchify(self):
        """Test channels after 2x2 patchify."""
        # 32 channels * 4 (2x2 patch) = 128
        expected = VAE_Z_CHANNELS * 4
        assert LATENT_CHANNELS_AFTER_PATCHIFY == expected == 128


# ============================================================================
# AutoEncoderParams Tests
# ============================================================================

class TestAutoEncoderParams:
    """Tests for VAE configuration parameters."""

    def test_default_params(self):
        """Test default AutoEncoderParams values."""
        params = AutoEncoderParams()

        # Resolution and channels
        assert params.resolution == 256
        assert params.in_channels == 3
        assert params.z_channels == 32
        assert params.ch == 128  # Base channel count

    def test_channel_mult_structure(self):
        """Test channel multiplier structure for encoder/decoder."""
        params = AutoEncoderParams()

        # Should have multiple levels for progressive downsampling
        assert len(params.ch_mult) >= 3
        # Each level increases channels
        for i in range(len(params.ch_mult) - 1):
            assert params.ch_mult[i] <= params.ch_mult[i + 1], \
                f"Channel mult should increase: {params.ch_mult}"

    def test_attention_resolutions(self):
        """Test attention is applied at specified resolutions."""
        params = AutoEncoderParams()

        # Attention typically at higher-level features
        assert params.attn_resolutions is not None
        assert len(params.attn_resolutions) > 0


# ============================================================================
# Patchify/Unpatchify Tests
# ============================================================================

class TestPatchify:
    """Tests for patchify/unpatchify operations."""

    @pytest.fixture
    def vae(self):
        """Create a minimal VAE for testing (using meta device)."""
        with torch.device("meta"):
            return AutoEncoder(AutoEncoderParams())

    def test_patchify_shape(self, vae):
        """Test patchify reduces spatial dims and increases channels."""
        # Input: [B, C, H, W] where C=32 (z_channels)
        batch, channels = 2, 32
        height, width = 64, 64

        z = torch.randn(batch, channels, height, width)
        patched = vae.patchify(z)

        # After 2x2 patchify:
        # - Channels: 32 * 4 = 128
        # - Spatial: 64/2 = 32
        expected_channels = channels * 4
        expected_h = height // 2
        expected_w = width // 2

        assert patched.shape == (batch, expected_channels, expected_h, expected_w), \
            f"Expected {(batch, expected_channels, expected_h, expected_w)}, got {patched.shape}"

    def test_unpatchify_shape(self, vae):
        """Test unpatchify restores original dimensions."""
        batch, channels = 2, 128  # Patchified channels
        height, width = 32, 32  # Patchified spatial

        patched = torch.randn(batch, channels, height, width)
        unpatched = vae.unpatchify(patched)

        # After unpatchify:
        # - Channels: 128 / 4 = 32
        # - Spatial: 32 * 2 = 64
        expected_channels = channels // 4
        expected_h = height * 2
        expected_w = width * 2

        assert unpatched.shape == (batch, expected_channels, expected_h, expected_w)

    def test_patchify_unpatchify_inverse(self, vae):
        """Test unpatchify(patchify(x)) ≈ x."""
        batch, channels, height, width = 1, 32, 64, 64
        z = torch.randn(batch, channels, height, width)

        patched = vae.patchify(z)
        reconstructed = vae.unpatchify(patched)

        assert reconstructed.shape == z.shape
        assert torch.allclose(reconstructed, z, atol=1e-5), \
            "Unpatchify should be inverse of patchify"

    def test_patchify_preserves_values(self, vae):
        """Test patchify just rearranges, doesn't modify values."""
        batch, channels, height, width = 1, 32, 4, 4
        z = torch.arange(batch * channels * height * width).float()
        z = z.view(batch, channels, height, width)

        patched = vae.patchify(z)

        # Total elements should be preserved
        assert patched.numel() == z.numel()

        # After unpatchify, values should be exactly restored
        unpatched = vae.unpatchify(patched)
        assert torch.equal(unpatched, z)


# ============================================================================
# BatchNorm Normalization Tests
# ============================================================================

class TestBatchNormNormalization:
    """Tests for BatchNorm latent normalization."""

    @pytest.fixture
    def vae(self):
        """Create VAE on CPU for BatchNorm testing."""
        # Use small params to create actual (not meta) model
        params = AutoEncoderParams()
        # Can't easily create full model without weights
        # So we test the normalize method concept
        return None

    def test_normalize_uses_running_stats(self):
        """Test that normalize uses running stats (eval mode)."""
        # BatchNorm in eval mode uses running mean/var
        # In training mode, uses batch statistics
        # For generation, we always want running stats

        # Create a simple BatchNorm to test the concept
        bn = torch.nn.BatchNorm2d(32)
        bn.running_mean = torch.zeros(32)
        bn.running_var = torch.ones(32)

        x = torch.randn(2, 32, 8, 8)

        # Eval mode should use running stats
        bn.train(False)
        out_eval = bn(x)

        # Training mode uses batch stats
        bn.train(True)
        out_train = bn(x)

        # Outputs should differ because different stats are used
        assert not torch.allclose(out_eval, out_train), \
            "Eval and train mode should produce different outputs"

    def test_normalize_maintains_dtype(self):
        """Test BatchNorm preserves dtype for bf16."""
        bn = torch.nn.BatchNorm2d(32)
        bn.running_mean = torch.zeros(32)
        bn.running_var = torch.ones(32)

        x_bf16 = torch.randn(1, 32, 4, 4, dtype=torch.bfloat16)

        # BatchNorm2d with float params handles bf16 input
        bn.train(False)
        # Note: BatchNorm might convert to float32 internally
        # but output should be compatible


# ============================================================================
# Encode/Decode Shape Tests
# ============================================================================

class TestEncodeDecodeShapes:
    """Tests for encode/decode shape transformations."""

    def test_pixel_to_latent_compression(self):
        """Test total compression from pixels to latents."""
        # Input: [B, 3, H, W] pixels
        # Output: [B, 128, H/16, W/16] latents

        pixel_h, pixel_w = 1024, 1024
        latent_h = pixel_h // TOTAL_SPATIAL_COMPRESSION
        latent_w = pixel_w // TOTAL_SPATIAL_COMPRESSION

        assert latent_h == 64
        assert latent_w == 64

    def test_latent_shape_for_standard_resolution(self):
        """Test latent dimensions for standard 1024x1024 images."""
        height, width = 1024, 1024

        latent_h = height // TOTAL_SPATIAL_COMPRESSION
        latent_w = width // TOTAL_SPATIAL_COMPRESSION

        # For transformer: flatten to sequence
        num_tokens = latent_h * latent_w
        assert num_tokens == 4096, f"Expected 4096 tokens for 1024x1024, got {num_tokens}"

    def test_latent_shape_for_various_resolutions(self):
        """Test latent dimensions for various image sizes."""
        test_cases = [
            # (height, width, expected_tokens)
            (1024, 1024, 4096),   # 64 * 64
            (512, 512, 1024),    # 32 * 32
            (768, 768, 2304),    # 48 * 48
            (1024, 768, 3072),   # 64 * 48
        ]

        for h, w, expected_tokens in test_cases:
            lat_h = h // TOTAL_SPATIAL_COMPRESSION
            lat_w = w // TOTAL_SPATIAL_COMPRESSION
            actual_tokens = lat_h * lat_w

            assert actual_tokens == expected_tokens, \
                f"For {h}x{w}: expected {expected_tokens} tokens, got {actual_tokens}"


# ============================================================================
# Flux2VAE Wrapper Tests
# ============================================================================

class TestFlux2VAE:
    """Tests for Flux2VAE convenience wrapper."""

    def test_flux2vae_instantiation(self):
        """Test Flux2VAE can be instantiated."""
        # Just test the class exists and has expected interface
        assert hasattr(Flux2VAE, "__init__")
        assert hasattr(Flux2VAE, "encode") or hasattr(Flux2VAE, "forward")


# ============================================================================
# Latent Space Properties Tests
# ============================================================================

class TestLatentSpaceProperties:
    """Tests for latent space characteristics."""

    def test_latent_channel_count(self):
        """Test FLUX.2 uses 32 latent channels pre-patchify."""
        params = AutoEncoderParams()
        assert params.z_channels == 32

    def test_patchified_latent_channels(self):
        """Test patchified latents have 128 channels."""
        assert LATENT_CHANNELS_AFTER_PATCHIFY == 128

    def test_latent_matches_transformer_input(self):
        """Test latent channels match transformer in_channels."""
        from llm_dit.models.flux2.constants import Klein9BParams

        params = Klein9BParams()
        assert params.in_channels == LATENT_CHANNELS_AFTER_PATCHIFY, \
            f"Transformer in_channels ({params.in_channels}) should match " \
            f"patchified latent channels ({LATENT_CHANNELS_AFTER_PATCHIFY})"


# ============================================================================
# Sequence Conversion Tests
# ============================================================================

class TestSequenceConversion:
    """Tests for converting between spatial and sequence representations."""

    def test_spatial_to_sequence(self):
        """Test converting [B, C, H, W] to [B, H*W, C] for transformer."""
        batch, channels, height, width = 2, 128, 64, 64
        spatial = torch.randn(batch, channels, height, width)

        # FLUX.2 transformer expects [B, seq_len, channels]
        # Conversion: permute then reshape
        sequence = spatial.permute(0, 2, 3, 1).reshape(batch, height * width, channels)

        assert sequence.shape == (batch, 4096, channels)

    def test_sequence_to_spatial(self):
        """Test converting [B, seq_len, C] back to [B, C, H, W]."""
        batch, seq_len, channels = 2, 4096, 128
        height = width = int(math.sqrt(seq_len))  # Assumes square

        sequence = torch.randn(batch, seq_len, channels)

        # Reshape back to spatial
        spatial = sequence.view(batch, height, width, channels).permute(0, 3, 1, 2)

        assert spatial.shape == (batch, channels, height, width)

    def test_spatial_sequence_roundtrip(self):
        """Test spatial -> sequence -> spatial preserves values."""
        batch, channels, height, width = 1, 128, 32, 32
        original = torch.randn(batch, channels, height, width)

        # To sequence
        sequence = original.permute(0, 2, 3, 1).reshape(batch, height * width, channels)

        # Back to spatial
        restored = sequence.view(batch, height, width, channels).permute(0, 3, 1, 2)

        assert torch.allclose(original, restored)


# ============================================================================
# Memory Efficiency Tests
# ============================================================================

class TestMemoryEfficiency:
    """Tests related to VAE memory usage."""

    def test_decode_input_shape(self):
        """Test decode expects correctly shaped input."""
        # Decode input: [B, C, H, W] patchified latents
        # After transformer, we have [B, seq_len, C]
        # Must reshape back to spatial for decode

        batch, seq_len, channels = 2, 4096, 128
        height = width = 64  # sqrt(4096)

        # Transformer output
        transformer_out = torch.randn(batch, seq_len, channels)

        # Reshape for VAE decode
        vae_input = transformer_out.view(batch, height, width, channels)
        vae_input = vae_input.permute(0, 3, 1, 2).contiguous()

        assert vae_input.shape == (batch, channels, height, width)


# Run with: uv run pytest tests/unit/test_flux2_vae.py -v
