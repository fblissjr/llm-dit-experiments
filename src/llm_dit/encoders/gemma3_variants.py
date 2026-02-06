"""
Gemma3 Variant Loaders for LTX-2 Text Encoding.

Last Updated: 2026-01-23

IMPORTANT: This module is PURE PYTORCH only.
Do NOT import or use any diffusers components.

Provides factory functions to load different Gemma3 variants:
- bf16: Full precision bfloat16 (~24GB VRAM)
- 8bit: TorchAO int8 quantization (~12GB VRAM)
- q4-qat: Pre-quantized Q4 QAT model (~3GB VRAM)

All variants use:
- LTX-2's custom tokenizer (special tokens for video conditioning)
- LTX-2's connector weights (jointly-trained feature extractor + embeddings connector)

This separation allows using any Gemma3 backbone variant while preserving
the LTX-2 specific components that are critical for video generation quality.

Memory Savings on RTX 4090:
- bf16: ~24GB (may OOM during generation)
- 8bit: ~12GB (safe, good quality, torchao int8)
- q4-qat: ~3GB (maximum headroom for transformer)

Usage:
    # Load 8-bit encoder for memory efficiency
    encoder = create_gemma3_encoder(
        variant="8bit",
        model_path="models/LTX-2",
        device="cuda",
    )
    embeddings = encoder.encode("A cat walking through a garden")

    # Load Q4 QAT for maximum memory efficiency
    encoder = create_gemma3_encoder(
        variant="q4-qat",
        model_path="models/LTX-2",
        text_encoder_path="~/Storage/gemma-3-12b-it-qat-q4_0-unquantized",
    )
"""

import logging
from pathlib import Path
from typing import Literal, Optional

import torch

logger = logging.getLogger(__name__)

# Valid Gemma3 variants
Gemma3Variant = Literal["bf16", "8bit", "q4-qat"]

# Default paths for LTX-2 components
DEFAULT_LTX2_PATH = "models/LTX-2"
DEFAULT_TOKENIZER_SUBPATH = "tokenizer"
DEFAULT_TEXT_ENCODER_SUBPATH = "text_encoder"
DEFAULT_CONNECTOR_SHARD = "diffusion_pytorch_model-00011-of-00012.safetensors"


def create_gemma3_encoder(
    variant: Gemma3Variant = "bf16",
    model_path: str = DEFAULT_LTX2_PATH,
    text_encoder_path: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_sequence_length: int = 256,
    use_connector: bool = True,
) -> "Gemma3Encoder":
    """
    Factory function to create Gemma3 encoder with specified variant.

    This is the recommended way to load Gemma3 for LTX-2 text encoding.
    It automatically handles:
    - Loading the correct Gemma3 backbone variant
    - Using LTX-2's custom tokenizer
    - Loading LTX-2's jointly-trained connector weights

    Args:
        variant: Which Gemma3 variant to load:
            - "bf16": Standard bfloat16 precision (~24GB)
            - "8bit": TorchAO int8 quantization (~12GB)
            - "q4-qat": Pre-quantized Q4 QAT model (~3GB)
        model_path: Path to LTX-2 model directory (contains tokenizer/, text_encoder/)
        text_encoder_path: Override path for Gemma model weights.
            - For bf16/8bit: defaults to model_path/text_encoder/
            - For q4-qat: specify path to Q4 QAT model
        device: Device to load on ("cuda", "cpu", "auto")
        dtype: Model dtype (bfloat16 recommended)
        max_sequence_length: Maximum sequence length (256 for LTX-2)
        use_connector: Whether to use Embeddings1DConnector (True for production)

    Returns:
        Initialized Gemma3Encoder ready for encoding.

    Example:
        # Standard bf16 (high quality, high memory)
        encoder = create_gemma3_encoder("bf16", "models/LTX-2")

        # 8-bit (good quality, moderate memory)
        encoder = create_gemma3_encoder("8bit", "models/LTX-2")

        # Q4 QAT (good quality, minimum memory)
        encoder = create_gemma3_encoder(
            "q4-qat",
            "models/LTX-2",
            text_encoder_path="~/Storage/gemma-3-12b-it-qat-q4_0-unquantized"
        )
    """
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    # Resolve paths
    model_path_obj = Path(model_path).expanduser()
    tokenizer_path = str(model_path_obj / DEFAULT_TOKENIZER_SUBPATH)
    connectors_path = str(model_path_obj / DEFAULT_TEXT_ENCODER_SUBPATH / DEFAULT_CONNECTOR_SHARD)

    # Determine text encoder path
    if text_encoder_path:
        encoder_path = str(Path(text_encoder_path).expanduser())
    else:
        encoder_path = str(model_path_obj / DEFAULT_TEXT_ENCODER_SUBPATH)

    logger.info(f"Creating Gemma3 encoder with variant={variant}")
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  Text encoder: {encoder_path}")
    logger.info(f"  Tokenizer: {tokenizer_path}")
    logger.info(f"  Connectors: {connectors_path}")

    # Load based on variant
    if variant == "bf16":
        return _load_bf16_encoder(
            encoder_path=encoder_path,
            tokenizer_path=tokenizer_path,
            connectors_path=connectors_path,
            device=device,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
            use_connector=use_connector,
        )
    elif variant == "8bit":
        return _load_8bit_encoder(
            encoder_path=encoder_path,
            tokenizer_path=tokenizer_path,
            connectors_path=connectors_path,
            device=device,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
            use_connector=use_connector,
        )
    elif variant == "q4-qat":
        return _load_q4_qat_encoder(
            encoder_path=encoder_path,
            tokenizer_path=tokenizer_path,
            connectors_path=connectors_path,
            device=device,
            dtype=dtype,
            max_sequence_length=max_sequence_length,
            use_connector=use_connector,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}. Valid: bf16, 8bit, q4-qat")


def _load_bf16_encoder(
    encoder_path: str,
    tokenizer_path: str,
    connectors_path: str,
    device: str,
    dtype: torch.dtype,
    max_sequence_length: int,
    use_connector: bool,
) -> "Gemma3Encoder":
    """Load standard bf16 Gemma3 encoder."""
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    logger.info("Loading bf16 Gemma3 encoder...")
    encoder = Gemma3Encoder(
        model_id=encoder_path,
        device=device,
        dtype=dtype,
        max_sequence_length=max_sequence_length,
        connectors_path=connectors_path,
        tokenizer_path=tokenizer_path,
        use_connector=use_connector,
    )
    encoder._load_model()

    _log_memory_usage("bf16 encoder loaded")
    return encoder


def _load_8bit_encoder(
    encoder_path: str,
    tokenizer_path: str,
    connectors_path: str,
    device: str,
    dtype: torch.dtype,
    max_sequence_length: int,
    use_connector: bool,
) -> "Gemma3Encoder":
    """
    Load 8-bit quantized Gemma3 encoder using torchao.

    Memory savings: ~24GB -> ~12GB for Gemma3-12B.

    Strategy:
    1. Load Gemma model to CPU in bf16 (~24GB system RAM)
    2. Apply torchao int8_weight_only() quantization
    3. Move quantized model to GPU (~12GB VRAM)
    4. Load LTX-2 connector weights (full precision)
    5. Assemble into encoder

    Note: Uses torchao for all quantization. LTX-2's sharded checkpoint
    format requires manual weight loading.
    """
    try:
        from transformers import AutoTokenizer, Gemma3ForCausalLM
    except ImportError:
        raise ImportError("8-bit loading requires transformers>=4.44.0")

    from llm_dit.encoders.gemma3 import (
        Embeddings1DConnector,
        FeatureExtractorLinear,
        Gemma3Encoder,
        load_connector_weights,
    )

    logger.info("Loading 8-bit Gemma3 encoder with torchao quantization...")

    # Check if encoder_path contains LTX-2 sharded weights or standard HF weights
    encoder_path_obj = Path(encoder_path)
    index_file = encoder_path_obj / "diffusion_pytorch_model.safetensors.index.json"

    if index_file.exists():
        # LTX-2 sharded format - use torchao int8 quantization (same approach as q4-qat)
        logger.info("LTX-2 checkpoint format detected. Using torchao int8 quantization...")

        from llm_dit.utils.availability import is_torchao_available

        has_torchao = is_torchao_available()
        if not has_torchao:
            logger.warning("torchao not available, falling back to bf16 (will use ~24GB)")

        # Step 1: Load model on CPU using the existing Gemma3Encoder loader
        logger.info("Loading model on CPU for quantization...")
        encoder = Gemma3Encoder(
            model_id=encoder_path,
            device="cpu",  # Load on CPU first
            dtype=dtype,
            max_sequence_length=max_sequence_length,
            connectors_path=connectors_path,
            tokenizer_path=tokenizer_path,
            use_connector=use_connector,
        )
        encoder._load_model()

        # Step 2: Apply int8 quantization via unified system
        if has_torchao and encoder._model is not None:
            from llm_dit.quantization import quantize_component

            logger.info("Applying int8 quantization via unified system...")
            encoder._model, stats = quantize_component(
                encoder._model, method="int8", component_type="encoder"
            )
            logger.info(
                f"Quantized: {stats['quantized_layers']}/{stats['total_layers']} layers"
            )
            _log_memory_usage("After int8 quantization (CPU)")

        # Step 3: Move to target device
        target_device = device if device != "auto" else "cuda"
        logger.info(f"Moving quantized model to {target_device}...")
        if encoder._model is not None:
            encoder._model = encoder._model.to(torch.device(target_device))
        if encoder._feature_extractor is not None:
            encoder._feature_extractor = encoder._feature_extractor.to(torch.device(target_device))
        if encoder._embeddings_connector is not None:
            encoder._embeddings_connector = encoder._embeddings_connector.to(
                torch.device(target_device), dtype=dtype
            )
        encoder._device_str = target_device
        _log_memory_usage(f"Model on {target_device}")

        return encoder
    else:
        # Standard HuggingFace format - also use torchao for consistency
        logger.info(f"Loading 8-bit Gemma from: {encoder_path}")

        from llm_dit.utils.availability import is_torchao_available

        has_torchao = is_torchao_available()
        if not has_torchao:
            logger.warning("torchao not available, loading in bf16")

        # Load model on CPU first for quantization
        model = Gemma3ForCausalLM.from_pretrained(
            encoder_path,
            torch_dtype=dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        model.requires_grad_(False)

        # Apply int8 quantization via unified system
        if has_torchao:
            from llm_dit.quantization import quantize_component

            logger.info("Applying int8 quantization via unified system...")
            model, stats = quantize_component(model, method="int8", component_type="encoder")
            logger.info(
                f"Quantized: {stats['quantized_layers']}/{stats['total_layers']} layers"
            )

        # Move to target device
        target_device = device if device != "auto" else "cuda"
        model = model.to(torch.device(target_device))

        # Load tokenizer from LTX-2 (CRITICAL: must use LTX-2's tokenizer)
        logger.info(f"Loading tokenizer from: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            model_max_length=max_sequence_length,
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load connector weights
        logger.info(f"Loading connector weights from: {connectors_path}")
        feature_extractor = FeatureExtractorLinear(dtype=dtype)
        embeddings_connector = None

        if Path(connectors_path).exists():
            from safetensors import safe_open

            with safe_open(connectors_path, framework="pt") as f:
                if "text_proj_in.weight" in f.keys():
                    fe_weight = f.get_tensor("text_proj_in.weight")
                    feature_extractor.aggregate_embed.weight.data = fe_weight.to(dtype)

                if use_connector:
                    import json

                    from llm_dit.encoders.gemma3 import DEFAULT_CONNECTORS_CONFIG

                    config_path = Path(DEFAULT_CONNECTORS_CONFIG)
                    if config_path.exists():
                        with open(config_path) as cfg:
                            config = json.load(cfg)
                    else:
                        config = {
                            "video_connector_attention_head_dim": 128,
                            "video_connector_num_attention_heads": 30,
                            "video_connector_num_layers": 2,
                            "video_connector_num_learnable_registers": 128,
                            "rope_type": "interleaved",
                            "rope_theta": 10000.0,
                        }

                    embeddings_connector = Embeddings1DConnector.from_config(config)
                    load_connector_weights(
                        embeddings_connector, Path(connectors_path), prefix="video_connector."
                    )

        # Move to device
        target_device = torch.device(device if device != "auto" else "cuda")
        feature_extractor = feature_extractor.to(target_device)
        if embeddings_connector is not None:
            embeddings_connector = embeddings_connector.to(target_device, dtype=dtype)

        # Create encoder wrapper
        encoder = Gemma3Encoder.__new__(Gemma3Encoder)
        encoder._model_id = encoder_path
        encoder._device_str = device
        encoder._dtype = dtype
        encoder._max_sequence_length = max_sequence_length
        encoder._quantization_variant = "int8"
        encoder._max_memory = None
        encoder._connectors_path = connectors_path
        encoder._text_encoder_path = encoder_path
        encoder._tokenizer_path = tokenizer_path
        encoder._use_connector = use_connector
        encoder._model = model
        encoder._tokenizer = tokenizer
        encoder._feature_extractor = feature_extractor
        encoder._embeddings_connector = embeddings_connector
        encoder._is_loaded = True
        encoder._is_offloaded = False

    _log_memory_usage("8-bit encoder loaded")
    return encoder


def _load_q4_qat_encoder(
    encoder_path: str,
    tokenizer_path: str,
    connectors_path: str,
    device: str,
    dtype: torch.dtype,
    max_sequence_length: int,
    use_connector: bool,
) -> "Gemma3Encoder":
    """
    Load Q4 QAT pre-quantized Gemma3 model.

    This loads a Gemma model that was quantized using QAT (Quantization Aware Training)
    to 4-bit precision. The model maintains good quality while using ~3GB VRAM.

    The Q4 QAT model must be downloaded separately from HuggingFace:
    - Model: google/gemma-3-12b-it-qat-q4_0-unquantized (or similar)

    Note: Q4 QAT models are pre-quantized and load directly without
    runtime quantization overhead.
    """
    try:
        from transformers import AutoTokenizer, Gemma3ForCausalLM
    except ImportError:
        raise ImportError("Q4 QAT loading requires transformers>=4.44.0")

    from llm_dit.encoders.gemma3 import (
        Embeddings1DConnector,
        FeatureExtractorLinear,
        Gemma3Encoder,
        load_connector_weights,
    )

    logger.info("Loading Q4 QAT Gemma3 encoder with torchao quantization...")
    logger.info(f"  Model path: {encoder_path}")

    # The "unquantized" QAT models store weights in bf16 format.
    # We need to apply quantization at load time using torchao.
    # Strategy: Load on CPU in bf16 -> Apply int4 quantization -> Move to GPU

    from llm_dit.utils.availability import is_torchao_available

    has_torchao = is_torchao_available()
    if not has_torchao:
        logger.warning("torchao not available, falling back to bf16 (will use ~24GB)")

    # Step 1: Load model on CPU first to avoid OOM during quantization
    logger.info("Loading model on CPU for quantization...")
    model = Gemma3ForCausalLM.from_pretrained(
        encoder_path,
        torch_dtype=dtype,
        device_map="cpu",  # Load on CPU first
        low_cpu_mem_usage=True,
    )
    model.requires_grad_(False)

    # Step 2: Apply int4 quantization (reduces ~24GB -> ~3GB) via unified system
    if has_torchao:
        from llm_dit.quantization import quantize_component

        logger.info("Applying int4 quantization via unified system...")
        model, stats = quantize_component(model, method="int4", component_type="encoder")
        logger.info(
            f"Quantized: {stats['quantized_layers']}/{stats['total_layers']} layers"
        )
        _log_memory_usage("After int4 quantization (CPU)")

    # Step 3: Move to target device
    target_device = device if device != "auto" else "cuda"
    target_device_obj = torch.device(target_device)
    logger.info(f"Moving quantized model to {target_device}...")
    model = model.to(target_device_obj)
    _log_memory_usage(f"Model on {target_device}")

    # Load tokenizer from LTX-2 (CRITICAL)
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        model_max_length=max_sequence_length,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load connector weights
    logger.info(f"Loading connector weights from: {connectors_path}")
    feature_extractor = FeatureExtractorLinear(dtype=dtype)
    embeddings_connector = None

    if Path(connectors_path).exists():
        from safetensors import safe_open

        with safe_open(connectors_path, framework="pt") as f:
            if "text_proj_in.weight" in f.keys():
                fe_weight = f.get_tensor("text_proj_in.weight")
                feature_extractor.aggregate_embed.weight.data = fe_weight.to(dtype)

            if use_connector:
                import json

                from llm_dit.encoders.gemma3 import DEFAULT_CONNECTORS_CONFIG

                config_path = Path(DEFAULT_CONNECTORS_CONFIG)
                if config_path.exists():
                    with open(config_path) as cfg:
                        config = json.load(cfg)
                else:
                    config = {
                        "video_connector_attention_head_dim": 128,
                        "video_connector_num_attention_heads": 30,
                        "video_connector_num_layers": 2,
                        "video_connector_num_learnable_registers": 128,
                        "rope_type": "interleaved",
                        "rope_theta": 10000.0,
                    }

                embeddings_connector = Embeddings1DConnector.from_config(config)
                load_connector_weights(
                    embeddings_connector, Path(connectors_path), prefix="video_connector."
                )
    else:
        logger.warning(f"Connector weights not found at {connectors_path}")

    # Move connectors to same device as model (target_device already set above)
    device_obj = torch.device(target_device)
    feature_extractor = feature_extractor.to(device_obj)
    if embeddings_connector is not None:
        embeddings_connector = embeddings_connector.to(device_obj, dtype=dtype)

    # Create encoder wrapper
    encoder = Gemma3Encoder.__new__(Gemma3Encoder)
    encoder._model_id = encoder_path
    encoder._device_str = device
    encoder._dtype = dtype
    encoder._max_sequence_length = max_sequence_length
    encoder._quantization_variant = "q4_0"
    encoder._max_memory = None
    encoder._connectors_path = connectors_path
    encoder._text_encoder_path = encoder_path
    encoder._tokenizer_path = tokenizer_path
    encoder._use_connector = use_connector
    encoder._model = model
    encoder._tokenizer = tokenizer
    encoder._feature_extractor = feature_extractor
    encoder._embeddings_connector = embeddings_connector
    encoder._is_loaded = True
    encoder._is_offloaded = False

    _log_memory_usage("Q4 QAT encoder loaded")
    return encoder


def _log_memory_usage(context: str) -> None:
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"[Memory] {context}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")


def estimate_encoder_memory(variant: Gemma3Variant) -> dict:
    """
    Estimate memory usage for a Gemma3 encoder variant.

    Returns:
        Dict with estimated memory in GB.
    """
    estimates = {
        "bf16": {
            "model_gb": 24.0,
            "connectors_gb": 0.5,
            "activations_gb": 1.0,
            "peak_gb": 26.0,
            "description": "Full bfloat16 precision",
        },
        "8bit": {
            "model_gb": 12.0,
            "connectors_gb": 0.5,
            "activations_gb": 1.0,
            "peak_gb": 14.0,
            "description": "TorchAO int8 quantization",
        },
        "q4-qat": {
            "model_gb": 3.0,
            "connectors_gb": 0.5,
            "activations_gb": 1.0,
            "peak_gb": 5.0,
            "description": "Q4 QAT pre-quantized model",
        },
    }
    return estimates.get(variant, estimates["bf16"])
