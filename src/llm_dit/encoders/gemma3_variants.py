"""
Gemma3 Variant Loaders for LTX-2 Text Encoding.

Last Updated: 2026-02-17

IMPORTANT: This module is PURE PYTORCH only.
Do NOT import or use any diffusers components.

Provides factory functions to load different Gemma3 variants:
- bf16: Full precision bfloat16 (~24GB VRAM)
- fp8: Native PyTorch fp8 layerwise casting (~12GB VRAM, no torchao)
- fp8-safetensors: Pre-converted fp8 weights (~12GB, fastest load)
- 8bit: TorchAO int8 quantization (~12GB VRAM)
- q4-qat: Pre-quantized Q4 QAT model (~3GB VRAM)

All variants use:
- LTX-2's custom tokenizer (special tokens for video conditioning)
- LTX-2's connector weights (jointly-trained feature extractor + embeddings connector)

This separation allows using any Gemma3 backbone variant while preserving
the LTX-2 specific components that are critical for video generation quality.

Memory Savings on RTX 4090:
- bf16: ~24GB (may OOM during generation)
- fp8: ~12GB (same footprint as int8, zero quantization overhead at load)
- fp8-safetensors: ~12GB (pre-converted, fastest load -- skips bf16 entirely)
- 8bit: ~12GB (safe, good quality, torchao int8)
- q4-qat: ~3GB (maximum headroom for transformer)

Usage:
    # Load fp8 encoder (recommended -- fast load, no torchao dependency)
    encoder = create_gemma3_encoder(
        variant="fp8",
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
Gemma3Variant = Literal["bf16", "fp8", "fp8-safetensors", "8bit", "q4-qat"]

# Default paths for LTX-2.3 components
DEFAULT_LTX2_PATH = "models/LTX-2.3"
DEFAULT_TOKENIZER_SUBPATH = "tokenizer"
DEFAULT_TEXT_ENCODER_SUBPATH = "text_encoder"
DEFAULT_CONNECTORS_FILE = "ltx-2.3-connectors.safetensors"


def create_gemma3_encoder(
    variant: Gemma3Variant = "bf16",
    model_path: str = DEFAULT_LTX2_PATH,
    text_encoder_path: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_sequence_length: int = 512,
    use_connector: bool = True,
    connectors_file: str = DEFAULT_CONNECTORS_FILE,
    model_version: str = "auto",  # kept for signature compat; ignored
) -> "Gemma3Encoder":
    """Factory function to create Gemma3 encoder for LTX-2.3 (V2.3 only)."""
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    model_path_obj = Path(model_path).expanduser()
    connectors_path = str(model_path_obj / connectors_file)

    if text_encoder_path:
        encoder_path = str(Path(text_encoder_path).expanduser())
    else:
        encoder_path = str(model_path_obj / DEFAULT_TEXT_ENCODER_SUBPATH)

    # Tokenizer lives inside the encoder directory (HF model layout).
    # Fall back to legacy separate tokenizer/ subpath if it exists.
    legacy_tokenizer = model_path_obj / DEFAULT_TOKENIZER_SUBPATH
    tokenizer_path = str(legacy_tokenizer) if legacy_tokenizer.exists() else encoder_path

    logger.info(f"Creating Gemma3 encoder with variant={variant}")
    logger.info(f"  Model path: {model_path}")
    logger.info(f"  Text encoder: {encoder_path}")
    logger.info(f"  Tokenizer: {tokenizer_path}")
    logger.info(f"  Connectors: {connectors_path}")

    kwargs = dict(
        encoder_path=encoder_path,
        tokenizer_path=tokenizer_path,
        connectors_path=connectors_path,
        device=device,
        dtype=dtype,
        max_sequence_length=max_sequence_length,
        use_connector=use_connector,
    )

    if variant == "bf16":
        return _load_bf16_encoder(**kwargs)
    elif variant == "fp8":
        return _load_fp8_encoder(**kwargs)
    elif variant == "fp8-safetensors":
        return _load_fp8_safetensors_encoder(model_path=model_path, **kwargs)
    elif variant == "8bit":
        return _load_8bit_encoder(**kwargs)
    elif variant == "q4-qat":
        return _load_q4_qat_encoder(**kwargs)
    else:
        raise ValueError(f"Unknown variant: {variant}. Valid: bf16, fp8, fp8-safetensors, 8bit, q4-qat")


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


def _load_fp8_encoder(
    encoder_path: str,
    tokenizer_path: str,
    connectors_path: str,
    device: str,
    dtype: torch.dtype,
    max_sequence_length: int,
    use_connector: bool,
) -> "Gemma3Encoder":
    """Load Gemma3 encoder with native fp8 layerwise casting (~12GB VRAM)."""
    from llm_dit.encoders.gemma3 import Gemma3Encoder
    from llm_dit.quantization.layerwise_fp8 import apply_fp8_layerwise_casting

    logger.info("Loading fp8 Gemma3 encoder with native layerwise casting...")

    encoder = Gemma3Encoder(
        model_id=encoder_path,
        device="cpu",
        dtype=dtype,
        max_sequence_length=max_sequence_length,
        quantization_variant="fp8",
        connectors_path=connectors_path,
        tokenizer_path=tokenizer_path,
        use_connector=use_connector,
    )
    encoder._load_model()
    _log_memory_usage("After bf16 load on CPU")

    if encoder._model is not None:
        converted = apply_fp8_layerwise_casting(encoder._model)
        logger.info(f"Applied fp8 layerwise casting: {converted} linear layers converted")

    target_device = device if device != "auto" else "cuda"
    logger.info(f"Moving fp8 model to {target_device}...")
    if encoder._model is not None:
        encoder._model = encoder._model.to(torch.device(target_device))
    if encoder._feature_extractor_v2 is not None:
        encoder._feature_extractor_v2 = encoder._feature_extractor_v2.to(torch.device(target_device))
    if encoder._embeddings_connector is not None:
        encoder._embeddings_connector = encoder._embeddings_connector.to(
            torch.device(target_device), dtype=dtype
        )
    if encoder._audio_connector is not None:
        encoder._audio_connector = encoder._audio_connector.to(
            torch.device(target_device), dtype=dtype
        )
    encoder._device_str = target_device

    _log_memory_usage("fp8 encoder loaded")
    return encoder


def _load_fp8_safetensors_encoder(
    encoder_path: str,
    tokenizer_path: str,
    connectors_path: str,
    model_path: str,
    device: str,
    dtype: torch.dtype,
    max_sequence_length: int,
    use_connector: bool,
) -> "Gemma3Encoder":
    """Load Gemma3 encoder from pre-converted fp8 safetensors checkpoint.

    Fastest load path: weights already in float8_e4m3fn. Produced by
    scripts/convert_gemma3_fp8.py. V2.3 connectors loaded from separate file.
    """
    from safetensors.torch import load_file as load_safetensors

    from llm_dit.encoders.gemma3 import Gemma3Encoder

    # Find the fp8 safetensors file
    model_path_obj = Path(model_path).expanduser()
    fp8_path = model_path_obj / "text_encoder_fp8.safetensors"
    if not fp8_path.exists():
        raise FileNotFoundError(
            f"Pre-converted fp8 checkpoint not found at {fp8_path}. "
            "Run: uv run python scripts/convert_gemma3_fp8.py "
            f"{model_path_obj}"
        )

    logger.info(f"Loading fp8-safetensors Gemma3 encoder from {fp8_path}...")

    # Step 1: Load the pre-converted state dict (Gemma model weights only)
    state_dict = load_safetensors(str(fp8_path), device="cpu")
    logger.info(f"Loaded {len(state_dict)} tensors from fp8 checkpoint")

    # Filter out old V1 connector weights embedded in the checkpoint
    # V2.3 connectors come from the separate connectors file instead
    model_sd: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key.startswith(("feature_extractor.", "embeddings_connector.")):
            continue  # Skip V1 connector weights
        model_sd[key] = tensor

    fp8_count = sum(1 for t in model_sd.values() if t.dtype == torch.float8_e4m3fn)
    logger.info(f"Model: {len(model_sd)} tensors ({fp8_count} fp8, {len(model_sd) - fp8_count} bf16)")

    # Step 2: Create empty model architecture and load weights
    try:
        from transformers import AutoTokenizer, Gemma3Config, Gemma3ForCausalLM
    except ImportError:
        raise ImportError("fp8-safetensors loading requires transformers>=4.44.0")

    logger.info(f"Creating Gemma3ForCausalLM architecture from {encoder_path}...")
    full_config = Gemma3Config.from_pretrained(encoder_path, local_files_only=True)
    text_config = full_config.text_config

    with torch.device("cpu"):
        if hasattr(Gemma3ForCausalLM, "_from_config"):
            model = Gemma3ForCausalLM._from_config(text_config)
        else:
            model = Gemma3ForCausalLM(text_config)

    missing, unexpected = model.load_state_dict(model_sd, strict=False, assign=True)
    if missing:
        expected_missing = {"lm_head.weight"}
        real_missing = [k for k in missing if k not in expected_missing]
        if real_missing:
            logger.warning(f"Missing keys: {real_missing[:10]}")
        if "lm_head.weight" in missing and hasattr(model, "tie_weights"):
            model.tie_weights()

    model.requires_grad_(False)

    # Step 3: Install fp8 layerwise casting hooks
    hook_count = 0
    from llm_dit.quantization.layerwise_fp8 import _post_forward_hook, _pre_forward_hook
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if module.weight.dtype == torch.float8_e4m3fn:
            module.register_forward_pre_hook(_pre_forward_hook)
            module.register_forward_hook(_post_forward_hook)
            hook_count += 1
    logger.info(f"Installed fp8 layerwise hooks on {hook_count} layers")

    # Step 4: Move to target device
    target_device = device if device != "auto" else "cuda"
    logger.info(f"Moving fp8 model to {target_device}...")
    model = model.to(torch.device(target_device))

    # Step 5: Load tokenizer
    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, model_max_length=max_sequence_length,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 6: Assemble encoder wrapper via __new__
    encoder = Gemma3Encoder.__new__(Gemma3Encoder)
    encoder._model_id = encoder_path
    encoder._device_str = target_device
    encoder._dtype = dtype
    encoder._max_sequence_length = max_sequence_length
    encoder._quantization_variant = "fp8"
    encoder._max_memory = None
    encoder._connectors_path = connectors_path
    encoder._text_encoder_path = encoder_path
    encoder._tokenizer_path = tokenizer_path
    encoder._use_connector = use_connector
    encoder._model = model
    encoder._tokenizer = tokenizer
    encoder._feature_extractor_v2 = None
    encoder._embeddings_connector = None
    encoder._audio_connector = None
    encoder._is_loaded = False  # Will be set True after connector loading
    encoder._is_offloaded = False
    encoder._is_pinned = False
    encoder._pinned_shadows = {}

    # Step 7: Load V2.3 connectors from separate file
    encoder._load_connector_weights()

    # Move connectors to target device
    if encoder._feature_extractor_v2 is not None:
        encoder._feature_extractor_v2 = encoder._feature_extractor_v2.to(torch.device(target_device))
    if encoder._embeddings_connector is not None:
        encoder._embeddings_connector = encoder._embeddings_connector.to(
            torch.device(target_device), dtype=dtype
        )
    if encoder._audio_connector is not None:
        encoder._audio_connector = encoder._audio_connector.to(
            torch.device(target_device), dtype=dtype
        )

    encoder._is_loaded = True
    _log_memory_usage("fp8-safetensors encoder loaded")
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
    """Load 8-bit quantized Gemma3 encoder using torchao (~12GB VRAM)."""
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    logger.info("Loading 8-bit Gemma3 encoder with torchao quantization...")

    # Goes through __init__ -> _load_model() -> _load_connector_weights() (V2.3 path)
    encoder = Gemma3Encoder(
        model_id=encoder_path,
        device="cpu",
        dtype=dtype,
        max_sequence_length=max_sequence_length,
        connectors_path=connectors_path,
        tokenizer_path=tokenizer_path,
        use_connector=use_connector,
    )
    encoder._load_model()

    # Apply int8 quantization to Gemma model
    from llm_dit.utils.availability import is_torchao_available
    has_torchao = is_torchao_available()
    if has_torchao and encoder._model is not None:
        from llm_dit.quantization import quantize_component
        logger.info("Applying int8 quantization via unified system...")
        encoder._model, stats = quantize_component(
            encoder._model, method="int8", component_type="encoder"
        )
        logger.info(f"Quantized: {stats['quantized_layers']}/{stats['total_layers']} layers")
        _log_memory_usage("After int8 quantization (CPU)")
    elif not has_torchao:
        logger.warning("torchao not available, falling back to bf16 (will use ~24GB)")

    # Move to target device
    target_device = device if device != "auto" else "cuda"
    logger.info(f"Moving quantized model to {target_device}...")
    if encoder._model is not None:
        encoder._model = encoder._model.to(torch.device(target_device))
    if encoder._feature_extractor_v2 is not None:
        encoder._feature_extractor_v2 = encoder._feature_extractor_v2.to(torch.device(target_device))
    if encoder._embeddings_connector is not None:
        encoder._embeddings_connector = encoder._embeddings_connector.to(
            torch.device(target_device), dtype=dtype
        )
    if encoder._audio_connector is not None:
        encoder._audio_connector = encoder._audio_connector.to(
            torch.device(target_device), dtype=dtype
        )
    encoder._device_str = target_device

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
    """Load QAT-trained Gemma3 model with int8 quantization (~6GB VRAM).

    Uses the same _load_model() path as bf16/fp8 variants, which handles
    both V1 bundled and standard HF checkpoint formats. Applies int8
    quantization post-load (int4 requires mslk which is not yet public).
    """
    from llm_dit.encoders.gemma3 import Gemma3Encoder

    logger.info("Loading QAT Gemma3 encoder with int8 quantization...")
    logger.info(f"  Encoder path: {encoder_path}")

    encoder = Gemma3Encoder(
        model_id=encoder_path,
        device="cpu",
        dtype=dtype,
        max_sequence_length=max_sequence_length,
        connectors_path=connectors_path,
        tokenizer_path=tokenizer_path,
        use_connector=use_connector,
    )
    encoder._load_model()

    # Apply int8 quantization (int4 needs unreleased mslk package)
    from llm_dit.utils.availability import is_torchao_available
    if is_torchao_available() and encoder._model is not None:
        from llm_dit.quantization import quantize_component
        logger.info("Applying int8 quantization (int4 unavailable: mslk not public)...")
        encoder._model, stats = quantize_component(
            encoder._model, method="int8", component_type="encoder"
        )
        logger.info(f"Quantized: {stats['quantized_layers']}/{stats['total_layers']} layers")
        _log_memory_usage("After int8 quantization (CPU)")
    else:
        logger.warning("torchao not available, keeping bf16 (~24GB)")

    # Move to target device
    target_device = device if device != "auto" else "cuda"
    logger.info(f"Moving quantized model to {target_device}...")
    if encoder._model is not None:
        encoder._model = encoder._model.to(torch.device(target_device))
    if encoder._feature_extractor_v2 is not None:
        encoder._feature_extractor_v2 = encoder._feature_extractor_v2.to(torch.device(target_device))
    if encoder._embeddings_connector is not None:
        encoder._embeddings_connector = encoder._embeddings_connector.to(
            torch.device(target_device), dtype=dtype
        )
    if encoder._audio_connector is not None:
        encoder._audio_connector = encoder._audio_connector.to(
            torch.device(target_device), dtype=dtype
        )
    encoder._device_str = target_device

    _log_memory_usage("QAT encoder loaded (int8)")
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
        "fp8": {
            "model_gb": 12.0,
            "connectors_gb": 0.5,
            "activations_gb": 1.0,
            "peak_gb": 14.0,
            "description": "Native fp8 layerwise casting (no torchao)",
        },
        "fp8-safetensors": {
            "model_gb": 12.0,
            "connectors_gb": 0.5,
            "activations_gb": 1.0,
            "peak_gb": 14.0,
            "description": "Pre-converted fp8 checkpoint (fastest load)",
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
