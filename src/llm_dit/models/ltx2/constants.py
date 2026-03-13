"""
LTX-2 Official Reference Constants.

Last Updated: 2026-03-09

Reference values from the official LTX-2 repository for 1:1 comparison testing.
These match the defaults in coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py

Use these constants when:
- Running reference comparison tests
- Validating numerical equivalence with official implementation
- Generating baseline videos for quality assessment

For experiments that intentionally vary parameters, import these as a baseline
and override as needed.
"""

# =============================================================================
# Default Generation Parameters
# =============================================================================
# Source: coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py

# Video dimensions (must be divisible by 32 for VAE)
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 768

# Frame count (must follow formula: 8K + 1 for temporal VAE)
DEFAULT_NUM_FRAMES = 121  # 8 * 15 + 1 = 121

# Frame rate
DEFAULT_FRAME_RATE = 24.0

# Diffusion parameters
DEFAULT_NUM_INFERENCE_STEPS = 30  # T2V default (official V2.3)
DEFAULT_GUIDANCE_SCALE = 3.0  # CFG strength (V2.3 default)

# Reproducibility
DEFAULT_SEED = 10

# Negative prompt (used across config, schema, and pipeline)
LTX2_DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)


# =============================================================================
# Scheduler Parameters (LTX2Scheduler)
# =============================================================================
# Source: coderef/LTX-2/packages/ltx-core/src/ltx_core/components/schedulers.py

# Shift anchors (token counts)
SCHEDULER_SHIFT_ANCHOR_LOW = 1024   # Tokens at which base_shift applies
SCHEDULER_SHIFT_ANCHOR_HIGH = 4096  # Tokens at which max_shift applies

# Shift values
SCHEDULER_BASE_SHIFT = 0.95  # Shift at 1024 tokens
SCHEDULER_MAX_SHIFT = 2.05   # Shift at 4096 tokens

# Terminal sigma (for stretch mode)
SCHEDULER_TERMINAL = 0.1
SCHEDULER_STRETCH = True


# =============================================================================
# Two-Stage (High Quality) Parameters
# =============================================================================
# For upsampled generation workflow (reference: TI2VidTwoStagesPipeline)

TWO_STAGE_HEIGHT_STAGE1 = 512
TWO_STAGE_WIDTH_STAGE1 = 768
TWO_STAGE_STEPS_STAGE1 = 30  # V2.3 default (V2.0 was 40)

TWO_STAGE_HEIGHT_STAGE2 = 1024  # 2x upsampled
TWO_STAGE_WIDTH_STAGE2 = 1536   # 2x upsampled
# Stage 2 refinement: pre-computed noise schedule from the official reference repo.
# 4 sigma values define 3 denoising steps. Stage 2 re-noises the upsampled
# stage 1 output to sigma[0]=0.909375, then denoises through each step to 0.
# Used unconditionally by the two-stage pipeline (not configurable per-request).
# Source: coderef/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/constants.py
STAGE_2_SIGMA_SCHEDULE = [0.909375, 0.725, 0.421875, 0.0]

# Number of denoising steps in stage 2 = len(STAGE_2_SIGMA_SCHEDULE) - 1.
STAGE_2_STEPS = 3


# =============================================================================
# VAE Compression Ratios
# =============================================================================

VAE_TEMPORAL_COMPRESSION = 8   # frames -> latent frames
VAE_SPATIAL_COMPRESSION = 32   # pixels -> latent spatial
VAE_LATENT_CHANNELS = 128


# =============================================================================
# Model Architecture
# =============================================================================

TRANSFORMER_NUM_LAYERS = 48
TRANSFORMER_HIDDEN_DIM = 4096
TRANSFORMER_NUM_HEADS = 32

TEXT_ENCODER_HIDDEN_DIM = 3840  # Gemma3 projected dimension
TEXT_ENCODER_NUM_LAYERS = 49    # Gemma3 layer count


# =============================================================================
# Convenience Functions
# =============================================================================

def get_reference_config() -> dict:
    """Get a dictionary of all reference generation parameters.

    Returns:
        dict with all default generation parameters for 1:1 comparison.
    """
    return {
        "height": DEFAULT_HEIGHT,
        "width": DEFAULT_WIDTH,
        "num_frames": DEFAULT_NUM_FRAMES,
        "frame_rate": DEFAULT_FRAME_RATE,
        "num_inference_steps": DEFAULT_NUM_INFERENCE_STEPS,
        "guidance_scale": DEFAULT_GUIDANCE_SCALE,
        "seed": DEFAULT_SEED,
        # Scheduler
        "base_shift": SCHEDULER_BASE_SHIFT,
        "max_shift": SCHEDULER_MAX_SHIFT,
        "terminal": SCHEDULER_TERMINAL,
        "stretch": SCHEDULER_STRETCH,
    }


def get_quick_test_config() -> dict:
    """Get parameters for quick smoke tests (faster, lower quality).

    Returns:
        dict with reduced parameters for fast testing.
    """
    return {
        "height": DEFAULT_HEIGHT,
        "width": DEFAULT_WIDTH,
        "num_frames": 33,  # 8 * 4 + 1 = 33 (shorter video)
        "frame_rate": DEFAULT_FRAME_RATE,
        "num_inference_steps": 8,  # Fast
        "guidance_scale": DEFAULT_GUIDANCE_SCALE,
        "seed": DEFAULT_SEED,
        # Scheduler
        "base_shift": SCHEDULER_BASE_SHIFT,
        "max_shift": SCHEDULER_MAX_SHIFT,
        "terminal": SCHEDULER_TERMINAL,
        "stretch": SCHEDULER_STRETCH,
    }


def calculate_latent_tokens(
    num_frames: int = DEFAULT_NUM_FRAMES,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> int:
    """Calculate number of latent tokens for given video dimensions.

    Args:
        num_frames: Number of video frames
        height: Video height in pixels
        width: Video width in pixels

    Returns:
        Total number of latent tokens (t_latent * h_latent * w_latent)
    """
    t_latent = (num_frames - 1) // VAE_TEMPORAL_COMPRESSION + 1
    h_latent = height // VAE_SPATIAL_COMPRESSION
    w_latent = width // VAE_SPATIAL_COMPRESSION
    return t_latent * h_latent * w_latent
