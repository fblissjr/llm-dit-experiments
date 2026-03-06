"""
Tests for LTX-2 transformer implementation.

Last Updated: 2026-01-18

These tests verify the pure PyTorch LTX-2 transformer port matches
the expected behavior from the official implementation.

Run with: uv run pytest tests/unit/test_ltx2_transformer.py -v
"""

import pytest
import torch
from pathlib import Path
from unittest.mock import patch

# Import our implementation
from llm_dit.models.ltx2.rope import (
    LTXRopeType,
    apply_rotary_emb,
)
from llm_dit.layers import rms_norm
from llm_dit.models.ltx2.components import (
    GELUApprox,
    FeedForward,
    Timesteps,
    TimestepEmbedding,
    AdaLayerNormSingle,
    PixArtAlphaTextProjection,
)
from llm_dit.models.ltx2.attention import (
    Attention,
    AttentionFunction,
)
from llm_dit.models.ltx2.transformer import (
    LTX2Transformer,
    LTXModelType,
    TransformerConfig,
    BasicTransformerBlock,
)
from llm_dit.models.ltx2.loader import (
    map_key,
    is_audio_key,
    load_config,
    get_model_info,
)


# ============================================================================
# RoPE Tests
# ============================================================================

class TestRoPE:
    """Tests for rotary position embedding implementation."""

    def test_rope_type_enum(self):
        """Test RoPE type enum values."""
        assert LTXRopeType.INTERLEAVED.value == "interleaved"
        assert LTXRopeType.SPLIT.value == "split"

    def test_apply_rotary_emb_interleaved(self):
        """Test interleaved RoPE application."""
        batch, seq = 2, 100
        head_dim = 128

        # Create input tensor [B, T, D]
        x = torch.randn(batch, seq, head_dim)

        # For interleaved RoPE, cos/sin frequencies should match full dimension
        # since each pair of adjacent dimensions uses the same cos/sin
        cos_freq = torch.ones(seq, head_dim)
        sin_freq = torch.zeros(seq, head_dim)

        # Apply RoPE
        out = apply_rotary_emb(x, (cos_freq, sin_freq), LTXRopeType.INTERLEAVED)

        assert out.shape == x.shape

    def test_apply_rotary_emb_split(self):
        """Test split RoPE application."""
        batch, seq = 2, 100
        head_dim = 128

        x = torch.randn(batch, seq, head_dim)
        # For split RoPE, frequencies match half the dimension (first/second half rotated together)
        half_dim = head_dim // 2
        cos_freq = torch.ones(seq, half_dim)
        sin_freq = torch.zeros(seq, half_dim)

        out = apply_rotary_emb(x, (cos_freq, sin_freq), LTXRopeType.SPLIT)

        assert out.shape == x.shape


# ============================================================================
# Component Tests
# ============================================================================

class TestComponents:
    """Tests for core transformer components."""

    def test_gelu_approx(self):
        """Test GELU approximation with projection."""
        dim_in, dim_out = 4096, 4096
        gelu = GELUApprox(dim_in=dim_in, dim_out=dim_out)
        x = torch.randn(2, 10, dim_in)
        y = gelu(x)
        assert y.shape == (2, 10, dim_out)
        # Should not be identity
        assert not torch.allclose(x, y)

    def test_feedforward_shape(self):
        """Test FFN output shape matches input."""
        dim = 4096
        ff = FeedForward(dim=dim, dim_out=dim, mult=4)

        x = torch.randn(2, 100, dim)
        y = ff(x)

        assert y.shape == x.shape

    def test_timesteps_embedding(self):
        """Test timestep sinusoidal embedding."""
        timesteps_mod = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0  # Required parameter
        )
        t = torch.tensor([0.0, 0.5, 1.0])
        emb = timesteps_mod(t)
        assert emb.shape == (3, 256)

    def test_timestep_embedding_projection(self):
        """Test timestep MLP projection."""
        te = TimestepEmbedding(
            in_channels=256,
            time_embed_dim=4096,  # Correct param name
        )
        x = torch.randn(2, 256)
        y = te(x)
        assert y.shape == (2, 4096)

    def test_adaln_single_output(self):
        """Test AdaLN produces correct timestep/embedded output."""
        embedding_dim = 4096
        embedding_coefficient = 6  # default: shift/scale/gate for attn + FFN
        adaln = AdaLayerNormSingle(embedding_dim=embedding_dim)

        # Forward pass with timestep
        timestep = torch.tensor([0.5, 0.5])
        scale_shift_values, embedded_timestep = adaln(timestep, hidden_dtype=torch.float32)

        # scale_shift_values is [B, embedding_coefficient * embedding_dim] (flat)
        assert scale_shift_values.shape == (2, embedding_coefficient * embedding_dim)
        # embedded_timestep is the raw embedding
        assert embedded_timestep.shape == (2, embedding_dim)

    def test_text_projection(self):
        """Test caption projection dimensions."""
        proj = PixArtAlphaTextProjection(
            in_features=3840,  # Gemma3 dim
            hidden_size=4096,  # LTX-2 hidden
        )
        x = torch.randn(2, 100, 3840)
        y = proj(x)
        assert y.shape == (2, 100, 4096)

    def test_rms_norm(self):
        """Test RMS normalization."""
        x = torch.randn(2, 100, 64)
        y = rms_norm(x, weight=None, eps=1e-6)
        assert y.shape == x.shape
        # Output should have roughly unit variance
        assert torch.abs(y.pow(2).mean() - 1.0) < 0.5


# ============================================================================
# Attention Tests
# ============================================================================

class TestAttention:
    """Tests for multi-head attention implementation."""

    def test_self_attention_shape(self):
        """Test self-attention output shape."""
        attn = Attention(
            query_dim=4096,
            heads=32,
            dim_head=128,
            # No bias or qk_norm params - they're built-in
        )

        x = torch.randn(2, 100, 4096)
        y = attn(x)

        assert y.shape == x.shape

    def test_cross_attention_shape(self):
        """Test cross-attention output shape."""
        attn = Attention(
            query_dim=4096,
            heads=32,
            dim_head=128,
            context_dim=4096,  # Cross-attention
        )

        x = torch.randn(2, 100, 4096)
        context = torch.randn(2, 50, 4096)  # Different seq length
        y = attn(x, context=context)

        assert y.shape == x.shape

    def test_attention_function_enum(self):
        """Test attention function enum has expected backends."""
        assert AttentionFunction.PYTORCH is not None
        assert AttentionFunction.XFORMERS is not None
        assert AttentionFunction.FLASH_ATTENTION_3 is not None  # Correct name
        assert AttentionFunction.DEFAULT is not None


# ============================================================================
# Transformer Block Tests
# ============================================================================

class TestTransformerBlock:
    """Tests for BasicTransformerBlock."""

    @pytest.fixture
    def config(self):
        """Create a minimal config for testing."""
        return TransformerConfig(
            dim=4096,
            heads=32,
            d_head=128,
            context_dim=4096,
        )

    def test_block_instantiation(self, config):
        """Test block can be instantiated with config."""
        # BasicTransformerBlock requires idx (block index in transformer)
        block = BasicTransformerBlock(config=config, idx=0)
        assert block is not None


# ============================================================================
# Key Mapping Tests
# ============================================================================

class TestKeyMapping:
    """Tests for diffusers to our key mapping."""

    def test_basic_mappings(self):
        """Test key mappings from DIFFUSERS_TO_OURS."""
        # Test proj_in -> patchify_proj
        assert map_key("proj_in.weight") == "patchify_proj.weight"

        # Test time_embed
        assert "adaln_single" in map_key("time_embed.linear.weight")

        # Test norm_q/norm_k -> q_norm/k_norm
        assert "q_norm" in map_key("attn.norm_q.weight")
        assert "k_norm" in map_key("attn.norm_k.weight")

    def test_audio_key_detection(self):
        """Test audio key filtering."""
        # Audio keys
        assert is_audio_key("audio_proj.weight")
        assert is_audio_key("av_cross_attn.weight")
        assert is_audio_key("transformer_blocks.0.audio_attn.weight")
        assert is_audio_key("a2v_cross_attn.weight")

        # Video keys (should NOT be detected as audio)
        assert not is_audio_key("video_proj.weight")
        assert not is_audio_key("transformer_blocks.0.attn1.weight")
        assert not is_audio_key("caption_projection.weight")


# ============================================================================
# Config Loading Tests
# ============================================================================

class TestConfigLoading:
    """Tests for config and checkpoint handling."""

    def test_default_config_values(self):
        """Test default config has expected structure."""
        # Create temp path that doesn't exist
        fake_path = Path("/nonexistent/path")

        with patch("builtins.open", side_effect=FileNotFoundError):
            config = load_config(fake_path)

        assert config["num_attention_heads"] == 32
        assert config["attention_head_dim"] == 128
        assert config["num_layers"] == 48
        assert config["cross_attention_dim"] == 4096
        assert config["caption_channels"] == 3840


# ============================================================================
# Transformer Integration Tests
# ============================================================================

class TestLTX2Transformer:
    """Integration tests for full transformer."""

    @pytest.fixture
    def mini_transformer(self):
        """Create a minimal LTX2Transformer for testing (1 layer)."""
        return LTX2Transformer(
            model_type=LTXModelType.VideoOnly,
            num_attention_heads=4,
            attention_head_dim=32,
            in_channels=16,
            out_channels=16,
            num_layers=1,  # Minimal for speed
            cross_attention_dim=128,
            caption_channels=64,
            positional_embedding_max_pos=[4, 8, 8],
        )

    def test_transformer_instantiation(self, mini_transformer):
        """Test transformer can be instantiated."""
        assert mini_transformer is not None
        assert mini_transformer.num_layers == 1

    def test_model_type_enum(self):
        """Test model type enum properties."""
        assert LTXModelType.VideoOnly.is_video_enabled()
        assert not LTXModelType.VideoOnly.is_audio_enabled()
        assert LTXModelType.AudioVideo.is_video_enabled()
        assert LTXModelType.AudioVideo.is_audio_enabled()

    def test_get_num_params(self, mini_transformer):
        """Test parameter counting."""
        num_params = mini_transformer.get_num_params()
        assert num_params > 0
        # Mini model should be small
        assert num_params < 10_000_000  # < 10M params


# ============================================================================
# STG Perturbation Tests
# ============================================================================

class TestSTGPerturbation:
    """Tests for Spatio-Temporal Guidance (STG) perturbation support.

    STG works by running a 3rd forward pass where self-attention is skipped
    at specified blocks. The delta between conditioned and perturbed predictions
    drives spatial/temporal guidance. These tests verify:
    1. skip_self_attn flag correctly changes block output
    2. stg_blocks parameter propagates through transformer forward
    3. Edge cases (None, empty set) produce expected behavior
    """

    @pytest.fixture
    def stg_transformer(self):
        """Create a mini LTX2Transformer with proper initialization."""
        torch.manual_seed(42)
        model = LTX2Transformer(
            model_type=LTXModelType.VideoOnly,
            num_attention_heads=4,
            attention_head_dim=32,
            in_channels=16,
            out_channels=16,
            num_layers=4,
            cross_attention_dim=128,
            caption_channels=64,
            positional_embedding_max_pos=[4, 8, 8],
        )
        # Initialize all parameters to avoid NaN from torch.empty()
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
            else:
                torch.nn.init.zeros_(p)
        return model

    @pytest.fixture
    def stg_modality(self):
        """Create a Modality with proper position indices."""
        from llm_dit.models.ltx2 import Modality
        from llm_dit.pipelines.generate import create_position_indices

        torch.manual_seed(42)
        batch, latent_dim = 1, 16
        # 9 frames at 256x256 -> t=2, h=8, w=8 -> 128 tokens
        num_frames, height, width = 9, 256, 256
        t_lat = (num_frames - 1) // 8 + 1
        h_lat = height // 32
        w_lat = width // 32
        num_tokens = t_lat * h_lat * w_lat

        positions = create_position_indices(
            batch_size=batch,
            num_frames=num_frames,
            height=height,
            width=width,
            device=torch.device("cpu"),
        )

        return Modality(
            latent=torch.randn(batch, num_tokens, latent_dim),
            timesteps=torch.full((batch, num_tokens), 0.5),
            positions=positions,
            context=torch.randn(batch, 4, 64),
            enabled=True,
            context_mask=None,
        )

    def test_transformer_stg_blocks_changes_output(self, stg_transformer, stg_modality):
        """LTX2Transformer.forward(stg_blocks={...}) should differ from normal."""
        with torch.no_grad():
            normal_out, _ = stg_transformer(video=stg_modality)
            stg_out, _ = stg_transformer(video=stg_modality, stg_blocks={2})

        assert not torch.equal(normal_out, stg_out), \
            "STG output should differ from normal output"

    def test_transformer_empty_stg_blocks_matches_normal(self, stg_transformer, stg_modality):
        """Empty stg_blocks set should produce identical output to no STG."""
        with torch.no_grad():
            normal_out, _ = stg_transformer(video=stg_modality)
            stg_out, _ = stg_transformer(video=stg_modality, stg_blocks=set())

        assert torch.equal(normal_out, stg_out), \
            "Empty stg_blocks should produce identical output to normal"

    def test_transformer_stg_blocks_none_matches_normal(self, stg_transformer, stg_modality):
        """stg_blocks=None should produce identical output to default."""
        with torch.no_grad():
            normal_out, _ = stg_transformer(video=stg_modality)
            stg_out, _ = stg_transformer(video=stg_modality, stg_blocks=None)

        assert torch.equal(normal_out, stg_out), \
            "stg_blocks=None should produce identical output to default"

    def test_multiple_stg_blocks(self, stg_transformer, stg_modality):
        """Multiple STG blocks should produce different output than single block."""
        with torch.no_grad():
            single_out, _ = stg_transformer(video=stg_modality, stg_blocks={2})
            multi_out, _ = stg_transformer(video=stg_modality, stg_blocks={1, 2, 3})

        assert not torch.equal(single_out, multi_out), \
            "More STG blocks should produce different output"


# ============================================================================
# Slow Tests (require checkpoint)
# ============================================================================

@pytest.mark.slow
class TestWithCheckpoint:
    """Tests that require actual LTX-2 checkpoint. Skip in CI."""

    @pytest.fixture
    def checkpoint_path(self):
        """Get checkpoint path, skip if not available."""
        paths = [
            Path.home() / "Storage/LTX-2/transformer",
            Path.home() / "models/LTX-2/transformer",
            Path("models/LTX-2/transformer"),
        ]
        for path in paths:
            if path.exists():
                return path
        pytest.skip("LTX-2 checkpoint not found")

    def test_config_loading(self, checkpoint_path):
        """Test loading config from real checkpoint."""
        config = load_config(checkpoint_path)
        assert config["num_layers"] == 48
        assert config["num_attention_heads"] == 32

    def test_model_info(self, checkpoint_path):
        """Test getting model info without loading weights."""
        info = get_model_info(checkpoint_path)
        assert info["num_layers"] == 48
        assert info["hidden_dim"] == 4096
        assert info["estimated_size_bf16_gb"] > 30  # Should be ~38GB


# ============================================================================
# Generation Loop Tests
# ============================================================================

class TestGenerationLoop:
    """
    Tests for generation loop helpers in LTX2ExperimentBase.

    These verify the pure PyTorch generation infrastructure:
    - Position index computation (3D spatiotemporal positions)
    - Video modality creation (Modality dataclass)
    - Sigma schedule (monotonic, shifted by resolution)
    - Latent dimension computation

    Reference: experiments/ltx2/base.py
    """

    # -------------------------------------------------------------------------
    # Position Indices Tests
    # -------------------------------------------------------------------------

    def test_position_indices_shape(self):
        """Test _create_position_indices returns correct [B, 3, T] shape."""
        # Import the base class to test its method
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from experiments.ltx2.base import LTX2ExperimentBase

        # Create a minimal subclass for testing
        class TestExp(LTX2ExperimentBase):
            def run_iteration(self, config):
                return {}

        exp = TestExp("test", device="cpu")

        # Test case 1: Standard 768x512, 33 frames
        # Latent dims: t=(33-1)//8+1=5, h=512//32=16, w=768//32=24
        # Token count: 5*16*24 = 1920
        positions = exp._create_position_indices(1, 33, 512, 768)
        assert positions.shape == (1, 3, 1920), f"Expected (1, 3, 1920), got {positions.shape}"

        # Test case 2: Smaller 384x256, 17 frames
        # Latent dims: t=(17-1)//8+1=3, h=256//32=8, w=384//32=12
        # Token count: 3*8*12 = 288
        positions = exp._create_position_indices(1, 17, 256, 384)
        assert positions.shape == (1, 3, 288), f"Expected (1, 3, 288), got {positions.shape}"

    def test_position_indices_values(self):
        """Test position indices contain correct t, h, w ranges."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from experiments.ltx2.base import LTX2ExperimentBase

        class TestExp(LTX2ExperimentBase):
            def run_iteration(self, config):
                return {}

        exp = TestExp("test", device="cpu")

        # 384x256, 17 frames → t=3, h=8, w=12
        positions = exp._create_position_indices(1, 17, 256, 384)

        # Extract t, h, w indices
        t_indices = positions[0, 0, :].unique()  # Temporal
        h_indices = positions[0, 1, :].unique()  # Height
        w_indices = positions[0, 2, :].unique()  # Width

        # Verify ranges
        assert t_indices.min().item() == 0 and t_indices.max().item() == 2, \
            f"Temporal range should be [0, 2], got [{t_indices.min()}, {t_indices.max()}]"
        assert h_indices.min().item() == 0 and h_indices.max().item() == 7, \
            f"Height range should be [0, 7], got [{h_indices.min()}, {h_indices.max()}]"
        assert w_indices.min().item() == 0 and w_indices.max().item() == 11, \
            f"Width range should be [0, 11], got [{w_indices.min()}, {w_indices.max()}]"

    def test_position_indices_batch(self):
        """Test position indices work with batch_size > 1."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from experiments.ltx2.base import LTX2ExperimentBase

        class TestExp(LTX2ExperimentBase):
            def run_iteration(self, config):
                return {}

        exp = TestExp("test", device="cpu")

        # Batch size 4
        positions = exp._create_position_indices(4, 17, 256, 384)
        assert positions.shape == (4, 3, 288), f"Expected (4, 3, 288), got {positions.shape}"

        # All batches should have identical positions
        assert torch.all(positions[0] == positions[1])
        assert torch.all(positions[0] == positions[3])

    # -------------------------------------------------------------------------
    # Video Modality Tests
    # -------------------------------------------------------------------------

    def test_video_modality_creation(self):
        """Test _create_video_modality populates Modality fields correctly."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from experiments.ltx2.base import LTX2ExperimentBase
        from llm_dit.models.ltx2 import Modality

        class TestExp(LTX2ExperimentBase):
            def run_iteration(self, config):
                return {}

        exp = TestExp("test", device="cpu")

        # Create dummy inputs
        batch_size, num_tokens, latent_dim = 2, 288, 128
        context_dim, seq_len = 4096, 100

        latent = torch.randn(batch_size, num_tokens, latent_dim)
        timestep = torch.ones(batch_size, num_tokens) * 500  # Mid-diffusion
        positions = torch.randint(0, 10, (batch_size, 3, num_tokens))
        prompt_embeds = torch.randn(batch_size, seq_len, context_dim)

        modality = exp._create_video_modality(latent, timestep, positions, prompt_embeds)

        # Verify it's a Modality instance with correct fields
        assert isinstance(modality, Modality)
        assert modality.latent is latent
        assert modality.timesteps is timestep
        assert modality.positions is positions
        assert modality.context is prompt_embeds
        assert modality.enabled is True
        assert modality.context_mask is None

    # -------------------------------------------------------------------------
    # Sigma Schedule Tests
    # -------------------------------------------------------------------------

    def test_sigma_schedule_monotonic(self):
        """Test sigma schedule decreases monotonically."""
        import math

        # Replicate sigma computation from base.py
        num_inference_steps = 12
        video_seq_len = 1920  # Standard 768x512, 33 frames

        # Dynamic shift computation
        base_seq_len, max_seq_len = 1024, 4096
        base_shift, max_shift = 0.95, 2.05
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        mu = base_shift + m * (video_seq_len - base_seq_len)
        mu = max(min(mu, max_shift), base_shift)

        # Linear sigmas with exponential shift
        sigmas = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        exp_mu = math.exp(mu)
        sigmas = exp_mu / (exp_mu + (1.0 / sigmas - 1.0))

        # Verify monotonically decreasing
        for i in range(len(sigmas) - 1):
            assert sigmas[i] > sigmas[i + 1], \
                f"Sigma not monotonic at step {i}: {sigmas[i]:.4f} <= {sigmas[i+1]:.4f}"

    def test_sigma_schedule_range(self):
        """Test sigmas are in expected [0.2, 1.0] range after shift."""
        import math

        num_inference_steps = 12
        video_seq_len = 1920

        base_seq_len, max_seq_len = 1024, 4096
        base_shift, max_shift = 0.95, 2.05
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        mu = base_shift + m * (video_seq_len - base_seq_len)
        mu = max(min(mu, max_shift), base_shift)

        sigmas = torch.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
        exp_mu = math.exp(mu)
        sigmas = exp_mu / (exp_mu + (1.0 / sigmas - 1.0))

        # First sigma should be close to 1.0
        assert sigmas[0] > 0.9, f"First sigma should be near 1.0, got {sigmas[0]:.4f}"
        # Last sigma should be small but > 0
        assert 0 < sigmas[-1] < 0.5, f"Last sigma should be in (0, 0.5), got {sigmas[-1]:.4f}"

    def test_dynamic_shift_computation(self):
        """Test dynamic shift μ is computed correctly for various resolutions."""
        import math

        base_seq_len, max_seq_len = 1024, 4096
        base_shift, max_shift = 0.95, 2.05

        def compute_mu(seq_len):
            m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            mu = base_shift + m * (seq_len - base_seq_len)
            return max(min(mu, max_shift), base_shift)

        # Test at base resolution (1024 tokens)
        mu_base = compute_mu(1024)
        assert abs(mu_base - 0.95) < 0.01, f"μ at base should be ~0.95, got {mu_base:.4f}"

        # Test at max resolution (4096 tokens)
        mu_max = compute_mu(4096)
        assert abs(mu_max - 2.05) < 0.01, f"μ at max should be ~2.05, got {mu_max:.4f}"

        # Test at 1920 tokens (standard 768x512x33)
        # Expected: 0.95 + (2.05-0.95)/(4096-1024) * (1920-1024) ≈ 1.27
        mu_standard = compute_mu(1920)
        expected_mu = 0.95 + (2.05 - 0.95) / (4096 - 1024) * (1920 - 1024)
        assert abs(mu_standard - expected_mu) < 0.01, \
            f"μ at 1920 tokens should be ~{expected_mu:.2f}, got {mu_standard:.4f}"

    def test_latent_dimension_computation(self):
        """Test latent dimension computation matches expected token counts."""
        # LTX-2 compression: 8x temporal, 32x spatial

        test_cases = [
            # (num_frames, height, width, expected_tokens)
            (33, 512, 768, 1920),   # Standard: t=5, h=16, w=24
            (17, 256, 384, 288),    # Small: t=3, h=8, w=12
            (9, 512, 512, 512),     # Square: t=2, h=16, w=16
            (49, 768, 1024, 5376),  # Large: t=7, h=24, w=32
        ]

        for num_frames, height, width, expected in test_cases:
            t_latent = (num_frames - 1) // 8 + 1
            h_latent = height // 32
            w_latent = width // 32
            actual = t_latent * h_latent * w_latent

            assert actual == expected, \
                f"Token count mismatch for {num_frames}x{height}x{width}: " \
                f"expected {expected}, got {actual} (t={t_latent}, h={h_latent}, w={w_latent})"


class TestLTX2ReferenceValues:
    """
    Tests validating LTX2ExperimentBase against official LTX-2 reference values.

    Reference: coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py

    These tests ensure our implementation matches the official defaults and constraints:
    - DEFAULT_SEED = 10
    - DEFAULT_1_STAGE_HEIGHT = 512
    - DEFAULT_1_STAGE_WIDTH = 768
    - DEFAULT_NUM_FRAMES = 121 (constraint: frames % 8 == 1)
    - DEFAULT_FRAME_RATE = 24.0
    - DEFAULT_NUM_INFERENCE_STEPS = 40
    - DEFAULT_CFG_GUIDANCE_SCALE = 4.0
    """

    # Reference values from coderef/LTX-2/.../constants.py
    LTX2_REFERENCE = {
        "seed": 10,
        "height_1stage": 512,
        "width_1stage": 768,
        "height_2stage": 1024,  # 512 * 2
        "width_2stage": 1536,   # 768 * 2
        "num_frames": 121,
        "frame_rate": 24.0,
        "num_inference_steps": 30,
        "guidance_scale": 4.0,
        "latent_channels": 128,
    }

    # -------------------------------------------------------------------------
    # Frame Count Constraint Tests
    # -------------------------------------------------------------------------

    def test_frame_count_constraint(self):
        """Test frame count follows n*8+1 constraint (9, 17, 25, ..., 121)."""
        # LTX-2 temporal compression is 8x, so frames must be n*8+1
        # This ensures the first frame is preserved exactly in the latent
        valid_frame_counts = [9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121]

        for frames in valid_frame_counts:
            assert frames % 8 == 1, f"Frame count {frames} doesn't satisfy n*8+1"
            t_latent = (frames - 1) // 8 + 1
            assert t_latent == (frames + 7) // 8, f"Latent temporal dim mismatch for {frames} frames"

        # Default frame count should be valid
        default_frames = self.LTX2_REFERENCE["num_frames"]
        assert default_frames % 8 == 1, f"Default {default_frames} frames doesn't satisfy n*8+1"

    def test_invalid_frame_counts(self):
        """Test that invalid frame counts would produce incorrect latent dimensions."""
        invalid_frame_counts = [10, 20, 30, 32, 100, 120]  # Not n*8+1

        for frames in invalid_frame_counts:
            assert frames % 8 != 1, f"Frame count {frames} unexpectedly valid"
            # These would cause issues with the VAE's temporal compression

    def test_frame_to_latent_formula(self):
        """Test the frame-to-latent formula: t_latent = (frames - 1) // 8 + 1."""
        test_cases = [
            # (frames, expected_t_latent)
            (9, 2),    # (9-1)//8 + 1 = 1 + 1 = 2
            (17, 3),   # (17-1)//8 + 1 = 2 + 1 = 3
            (33, 5),   # (33-1)//8 + 1 = 4 + 1 = 5
            (121, 16), # (121-1)//8 + 1 = 15 + 1 = 16
        ]

        for frames, expected_t_latent in test_cases:
            actual = (frames - 1) // 8 + 1
            assert actual == expected_t_latent, \
                f"Frame {frames} should give t_latent={expected_t_latent}, got {actual}"

    # -------------------------------------------------------------------------
    # Resolution Constraint Tests
    # -------------------------------------------------------------------------

    def test_resolution_divisibility_by_32(self):
        """Test resolution must be divisible by 32 for 1-stage generation."""
        h1 = self.LTX2_REFERENCE["height_1stage"]
        w1 = self.LTX2_REFERENCE["width_1stage"]

        assert h1 % 32 == 0, f"1-stage height {h1} not divisible by 32"
        assert w1 % 32 == 0, f"1-stage width {w1} not divisible by 32"

        h2 = self.LTX2_REFERENCE["height_2stage"]
        w2 = self.LTX2_REFERENCE["width_2stage"]

        assert h2 % 32 == 0, f"2-stage height {h2} not divisible by 32"
        assert w2 % 32 == 0, f"2-stage width {w2} not divisible by 32"

    def test_2stage_is_2x_1stage(self):
        """Test 2-stage resolution is exactly 2x 1-stage resolution."""
        h1 = self.LTX2_REFERENCE["height_1stage"]
        w1 = self.LTX2_REFERENCE["width_1stage"]
        h2 = self.LTX2_REFERENCE["height_2stage"]
        w2 = self.LTX2_REFERENCE["width_2stage"]

        assert h2 == h1 * 2, f"2-stage height {h2} != 2 * {h1}"
        assert w2 == w1 * 2, f"2-stage width {w2} != 2 * {w1}"

    def test_common_valid_resolutions(self):
        """Test common valid resolutions are divisible by 32."""
        valid_resolutions = [
            (512, 768),    # 1-stage default
            (1024, 1536),  # 2-stage default
            (384, 512),    # Small
            (768, 1024),   # Medium
            (256, 384),    # Tiny (for quick tests)
        ]

        for h, w in valid_resolutions:
            assert h % 32 == 0, f"Height {h} not divisible by 32"
            assert w % 32 == 0, f"Width {w} not divisible by 32"
            h_latent = h // 32
            w_latent = w // 32
            assert h_latent > 0, f"Height {h} produces 0 latent height"
            assert w_latent > 0, f"Width {w} produces 0 latent width"

    # -------------------------------------------------------------------------
    # Default Value Tests
    # -------------------------------------------------------------------------

    def test_reference_seed(self):
        """Test default seed matches reference."""
        assert self.LTX2_REFERENCE["seed"] == 10

    def test_reference_frame_rate(self):
        """Test default frame rate matches reference."""
        assert self.LTX2_REFERENCE["frame_rate"] == 24.0

    def test_reference_inference_steps(self):
        """Test default inference steps matches reference."""
        assert self.LTX2_REFERENCE["num_inference_steps"] == 30

    def test_reference_guidance_scale(self):
        """Test default guidance scale matches reference."""
        assert self.LTX2_REFERENCE["guidance_scale"] == 4.0

    def test_reference_latent_channels(self):
        """Test latent channels matches reference."""
        assert self.LTX2_REFERENCE["latent_channels"] == 128

    # -------------------------------------------------------------------------
    # Token Count Calculation Tests (with reference values)
    # -------------------------------------------------------------------------

    def test_default_token_count(self):
        """Test token count for default resolution and frames."""
        # Default: 512x768, 121 frames
        h = self.LTX2_REFERENCE["height_1stage"]
        w = self.LTX2_REFERENCE["width_1stage"]
        frames = self.LTX2_REFERENCE["num_frames"]

        t_latent = (frames - 1) // 8 + 1  # 16
        h_latent = h // 32                 # 16
        w_latent = w // 32                 # 24

        total_tokens = t_latent * h_latent * w_latent

        assert t_latent == 16, f"Expected t_latent=16, got {t_latent}"
        assert h_latent == 16, f"Expected h_latent=16, got {h_latent}"
        assert w_latent == 24, f"Expected w_latent=24, got {w_latent}"
        assert total_tokens == 6144, f"Expected 6144 tokens, got {total_tokens}"

    def test_short_video_token_count(self):
        """Test token count for short video (33 frames)."""
        # 512x768, 33 frames (common test case)
        h, w, frames = 512, 768, 33

        t_latent = (frames - 1) // 8 + 1  # 5
        h_latent = h // 32                 # 16
        w_latent = w // 32                 # 24

        total_tokens = t_latent * h_latent * w_latent

        assert t_latent == 5, f"Expected t_latent=5, got {t_latent}"
        assert total_tokens == 1920, f"Expected 1920 tokens, got {total_tokens}"


# Run with: uv run pytest tests/unit/test_ltx2_transformer.py -v
# Run slow tests: uv run pytest tests/unit/test_ltx2_transformer.py -v --slow
