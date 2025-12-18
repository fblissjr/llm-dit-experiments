"""
Qwen3 constants and token IDs.

These values are consistent across Qwen3-4B and Qwen3-VL-4B-Instruct models.
Verified against tokenizer_config.json and special_tokens_map.json.

Model Variants:
    - Qwen3-4B: Base text model, supports thinking via enable_thinking parameter
    - Qwen3-VL-4B-Instruct: Vision-language model, NON-THINKING (Instruct variant)
    - Qwen3-VL-4B-Thinking: Vision-language model with thinking (separate model)

For VL embedding extraction, we use the Instruct variant which does NOT use
<think>...</think> blocks. The Thinking variant is a separate model.
"""

# =============================================================================
# SPECIAL TOKEN IDS
# =============================================================================
# These are defined in tokenizer_config.json added_tokens_decoder

# Core chat tokens
ENDOFTEXT_TOKEN_ID = 151643  # <|endoftext|> - BOS/PAD token
IM_START_TOKEN_ID = 151644   # <|im_start|> - Message start
IM_END_TOKEN_ID = 151645     # <|im_end|> - Message end / EOS

# Object/box/quad markers (not commonly used)
OBJECT_REF_START_TOKEN_ID = 151646  # <|object_ref_start|>
OBJECT_REF_END_TOKEN_ID = 151647    # <|object_ref_end|>
BOX_START_TOKEN_ID = 151648         # <|box_start|>
BOX_END_TOKEN_ID = 151649           # <|box_end|>
QUAD_START_TOKEN_ID = 151650        # <|quad_start|>
QUAD_END_TOKEN_ID = 151651          # <|quad_end|>

# Vision tokens (used by Qwen3-VL only)
VISION_START_TOKEN_ID = 151652  # <|vision_start|>
VISION_END_TOKEN_ID = 151653    # <|vision_end|>
VISION_PAD_TOKEN_ID = 151654    # <|vision_pad|>
IMAGE_PAD_TOKEN_ID = 151655     # <|image_pad|>
VIDEO_PAD_TOKEN_ID = 151656     # <|video_pad|>

# Tool/function calling tokens
TOOL_CALL_START_TOKEN_ID = 151657     # <tool_call>
TOOL_CALL_END_TOKEN_ID = 151658       # </tool_call>
TOOL_RESPONSE_START_TOKEN_ID = 151665 # <tool_response>
TOOL_RESPONSE_END_TOKEN_ID = 151666   # </tool_response>

# Fill-in-middle tokens (code completion)
FIM_PREFIX_TOKEN_ID = 151659  # <|fim_prefix|>
FIM_MIDDLE_TOKEN_ID = 151660  # <|fim_middle|>
FIM_SUFFIX_TOKEN_ID = 151661  # <|fim_suffix|>
FIM_PAD_TOKEN_ID = 151662     # <|fim_pad|>

# Repository/file tokens
REPO_NAME_TOKEN_ID = 151663  # <|repo_name|>
FILE_SEP_TOKEN_ID = 151664   # <|file_sep|>

# Thinking tokens (used by Qwen3-4B and Qwen3-VL-Thinking, NOT by VL-Instruct)
THINK_START_TOKEN_ID = 151667  # <think>
THINK_END_TOKEN_ID = 151668    # </think>

# Common text tokens
DOUBLE_NEWLINE_TOKEN_ID = 271  # \n\n
SINGLE_NEWLINE_TOKEN_ID = 198  # \n


# =============================================================================
# TOKEN SEQUENCES
# =============================================================================

# Empty think block token sequence: <think>\n\n</think>\n\n
# NOTE: Only used for Qwen3-4B and Qwen3-VL-Thinking, NOT for VL-Instruct
EMPTY_THINK_BLOCK_TOKENS = [
    THINK_START_TOKEN_ID,   # <think>
    DOUBLE_NEWLINE_TOKEN_ID,  # \n\n
    THINK_END_TOKEN_ID,     # </think>
    DOUBLE_NEWLINE_TOKEN_ID,  # \n\n
]

# Chat message start: <|im_start|>
CHAT_START_TOKENS = [IM_START_TOKEN_ID]

# Chat message end: <|im_end|>\n
CHAT_END_TOKENS = [IM_END_TOKEN_ID, SINGLE_NEWLINE_TOKEN_ID]


# =============================================================================
# TOKENIZER CONFIGURATION
# =============================================================================

# Padding configuration (from tokenizer_config.json)
PAD_TOKEN = "<|endoftext|>"
PAD_TOKEN_ID = ENDOFTEXT_TOKEN_ID
PADDING_SIDE = "left"  # Qwen3 convention

# EOS configuration
EOS_TOKEN = "<|im_end|>"
EOS_TOKEN_ID = IM_END_TOKEN_ID
EOS_TOKEN_IDS = [IM_END_TOKEN_ID, ENDOFTEXT_TOKEN_ID]  # Multiple valid EOS

# BOS configuration (Qwen3 doesn't use BOS)
BOS_TOKEN = None
BOS_TOKEN_ID = None
ADD_BOS_TOKEN = False

# Other tokenizer settings
CLEAN_UP_TOKENIZATION_SPACES = False
SPLIT_SPECIAL_TOKENS = False


# =============================================================================
# MODEL-SPECIFIC CONFIGURATION
# =============================================================================

# Qwen3-4B configuration
QWEN3_4B_CONFIG = {
    "hidden_size": 2560,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 9728,
    "max_position_embeddings": 40960,
    "rope_theta": 1_000_000,
    "rope_scaling": None,
    "vocab_size": 151936,
    # Chat template supports enable_thinking parameter
    "supports_enable_thinking": True,
}

# Qwen3-Embedding-4B configuration
# Specifically trained for embedding/retrieval tasks, based on Qwen3-4B-Base
QWEN3_EMBEDDING_4B_CONFIG = {
    "hidden_size": 2560,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 9728,
    "max_position_embeddings": 40960,
    "rope_theta": 1_000_000,
    "rope_scaling": None,
    "vocab_size": 151665,  # Smaller vocab than standard Qwen3 (151936)
    # Embedding model characteristics
    "is_embedding_model": True,
    "max_embedding_dim": 2560,
    "min_embedding_dim": 32,
    "mrl_support": True,  # Matryoshka Representation Learning
    "instruction_aware": True,
    # Uses last-token pooling for single-vector embeddings
    "default_pooling": "last_token",
    # Chat template present but optimized for embedding extraction
    "supports_enable_thinking": False,
}

# Qwen3-VL-4B-Instruct configuration
# NOTE: This is the NON-THINKING variant. Qwen3-VL-4B-Thinking is a separate model.
QWEN3_VL_4B_CONFIG = {
    "hidden_size": 2560,  # Text component
    "num_hidden_layers": 36,  # Text layers
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 9728,
    "max_position_embeddings": 262144,  # Much higher than 4B
    "rope_theta": 5_000_000,  # 5x higher than 4B
    "rope_scaling": {
        "mrope_interleaved": True,
        "mrope_section": [24, 20, 20],
        "rope_type": "default",
    },
    "vocab_size": 151936,
    # Vision component
    "vision_hidden_size": 1024,
    "vision_output_size": 2560,  # Projects to text hidden size
    "vision_depth": 24,
    "vision_patch_size": 16,
    # Instruct variant does NOT use thinking (separate Thinking model exists)
    "supports_enable_thinking": False,
    "is_thinking_model": False,
}


# =============================================================================
# GENERATION DEFAULTS
# =============================================================================

# Qwen3-4B generation defaults (from generation_config.json)
QWEN3_4B_GENERATION_DEFAULTS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "do_sample": True,
    "repetition_penalty": 1.0,
}

# Qwen3-VL-4B-Instruct generation defaults (from model card README.md)
# NOTE: This is for the Instruct variant. Thinking variant has different defaults.
QWEN3_VL_GENERATION_DEFAULTS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "do_sample": True,
    "repetition_penalty": 1.0,
    "presence_penalty": 1.5,  # From model card - helps reduce repetitions
}


# =============================================================================
# Z-IMAGE SPECIFIC LIMITS
# =============================================================================

# Maximum text sequence length for Z-Image DiT
# This is a DiT RoPE limit, not a Qwen3 limit
MAX_TEXT_SEQ_LEN = 1504

# Default hidden layer for embedding extraction
# -2 = penultimate layer (layer 35 of 36)
DEFAULT_HIDDEN_LAYER = -2

# Recommended VL hidden layer (cleaner than -2 for VL embeddings)
RECOMMENDED_VL_HIDDEN_LAYER = -6

# =============================================================================
# Z-IMAGE RESOLUTION CONSTRAINTS
# =============================================================================

# VAE scale factor (latent to pixel ratio)
# For Z-Image: latent_dim = image_dim / VAE_SCALE_FACTOR
VAE_SCALE_FACTOR = 8

# Required divisibility for image dimensions
# Z-Image requires dimensions divisible by 16 (VAE_SCALE_FACTOR * 2)
# This is due to the latent grid structure: latent_height = 2 * (height // 16)
VAE_MULTIPLE = 16

# Resolution limits (practical limits, not hard constraints)
MIN_RESOLUTION = 256   # Below this, quality degrades significantly
MAX_RESOLUTION = 4096  # Above this, VRAM becomes prohibitive
DEFAULT_RESOLUTION = 1024

# Common aspect ratios with their names
ASPECT_RATIOS = {
    "1:1": (1, 1),      # Square
    "4:3": (4, 3),      # Standard landscape
    "3:4": (3, 4),      # Standard portrait
    "16:9": (16, 9),    # Widescreen landscape
    "9:16": (9, 16),    # Phone portrait
    "3:2": (3, 2),      # Classic photo landscape
    "2:3": (2, 3),      # Classic photo portrait
    "21:9": (21, 9),    # Ultrawide
}


def snap_to_multiple(value: int, multiple: int = VAE_MULTIPLE) -> int:
    """Round value to nearest multiple.

    Args:
        value: The value to snap
        multiple: The multiple to snap to (default: VAE_MULTIPLE=16)

    Returns:
        Value rounded to nearest multiple
    """
    return round(value / multiple) * multiple


def validate_resolution(width: int, height: int) -> tuple[bool, str]:
    """Validate image resolution for Z-Image.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Tuple of (is_valid, error_message)
    """
    errors = []

    if width % VAE_MULTIPLE != 0:
        errors.append(f"Width must be divisible by {VAE_MULTIPLE} (got {width})")
    if height % VAE_MULTIPLE != 0:
        errors.append(f"Height must be divisible by {VAE_MULTIPLE} (got {height})")
    if width < MIN_RESOLUTION:
        errors.append(f"Width must be at least {MIN_RESOLUTION} (got {width})")
    if height < MIN_RESOLUTION:
        errors.append(f"Height must be at least {MIN_RESOLUTION} (got {height})")
    if width > MAX_RESOLUTION:
        errors.append(f"Width exceeds max {MAX_RESOLUTION} (got {width})")
    if height > MAX_RESOLUTION:
        errors.append(f"Height exceeds max {MAX_RESOLUTION} (got {height})")

    if errors:
        return False, "; ".join(errors)
    return True, ""


def calculate_latent_size(width: int, height: int) -> tuple[int, int]:
    """Calculate latent dimensions for given image size.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Tuple of (latent_width, latent_height)
    """
    # Z-Image uses: latent_dim = 2 * (image_dim // 16)
    # Which simplifies to: latent_dim = image_dim // 8
    return width // VAE_SCALE_FACTOR, height // VAE_SCALE_FACTOR


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_generation_defaults(model_type: str) -> dict:
    """Get generation defaults for a specific model type.

    Args:
        model_type: One of "qwen3_4b", "qwen3_vl"

    Returns:
        Dictionary of generation parameters
    """
    if model_type in ("qwen3_4b", "qwen3-4b", "text"):
        return QWEN3_4B_GENERATION_DEFAULTS.copy()
    elif model_type in ("qwen3_vl", "qwen3-vl", "vl", "vision"):
        return QWEN3_VL_GENERATION_DEFAULTS.copy()
    else:
        # Default to Qwen3-4B settings
        return QWEN3_4B_GENERATION_DEFAULTS.copy()


def get_model_config(model_type: str) -> dict:
    """Get model configuration for a specific model type.

    Args:
        model_type: One of "qwen3_4b", "qwen3_vl", "qwen3_embedding"

    Returns:
        Dictionary of model configuration
    """
    if model_type in ("qwen3_4b", "qwen3-4b", "text"):
        return QWEN3_4B_CONFIG.copy()
    elif model_type in ("qwen3_vl", "qwen3-vl", "vl", "vision"):
        return QWEN3_VL_4B_CONFIG.copy()
    elif model_type in ("qwen3_embedding", "qwen3-embedding", "embedding"):
        return QWEN3_EMBEDDING_4B_CONFIG.copy()
    else:
        return QWEN3_4B_CONFIG.copy()


def supports_enable_thinking(model_type: str) -> bool:
    """Check if a model supports thinking mode.

    For Qwen3-4B: True (supports enable_thinking parameter in chat template)
    For Qwen3-VL-4B-Instruct: False (non-thinking model, Thinking variant is separate)

    Args:
        model_type: One of "qwen3_4b", "qwen3_vl"

    Returns:
        True if model supports thinking mode
    """
    config = get_model_config(model_type)
    return config.get("supports_enable_thinking", False)
