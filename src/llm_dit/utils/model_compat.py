"""
Model compatibility validation for Z-Image text encoders.

Validates that a Qwen3-family model has the required architecture
for Z-Image embedding extraction. This allows experimentation with
different Qwen3 variants (e.g., Qwen3-4B-Instruct-2507).

Critical requirements for Z-Image:
- hidden_size == 2560 (embedding dimension)
- num_hidden_layers >= 36 (we extract hidden_states[-2])

Non-critical differences (won't affect embeddings):
- max_position_embeddings (only affects context length)
- rope_theta (affects positional encoding, not semantic content)
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


# Z-Image Qwen3-4B reference configuration
ZIMAGE_REFERENCE = {
    "hidden_size": 2560,
    "num_hidden_layers": 36,
    "intermediate_size": 9216,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 40960,
    "rope_theta": 1000000.0,
    "vocab_size": 151936,
}


@dataclass
class CompatibilityResult:
    """Result of model compatibility check."""

    compatible: bool
    model_name: str
    hidden_size: int
    num_hidden_layers: int
    warnings: list[str]
    errors: list[str]
    config_diff: Dict[str, tuple]  # key -> (reference, actual)

    def __str__(self) -> str:
        status = "COMPATIBLE" if self.compatible else "INCOMPATIBLE"
        lines = [f"Model: {self.model_name}", f"Status: {status}"]

        if self.errors:
            lines.append("\nErrors (will not work):")
            for err in self.errors:
                lines.append(f"  - {err}")

        if self.warnings:
            lines.append("\nWarnings (may affect quality):")
            for warn in self.warnings:
                lines.append(f"  - {warn}")

        if self.config_diff:
            lines.append("\nConfiguration differences:")
            for key, (ref, actual) in self.config_diff.items():
                lines.append(f"  {key}: {ref} -> {actual}")

        return "\n".join(lines)


def validate_model_config(
    config: Dict[str, Any] | "PretrainedConfig",
    model_name: str = "unknown",
) -> CompatibilityResult:
    """
    Validate a model configuration for Z-Image compatibility.

    Args:
        config: Model configuration dict or PretrainedConfig object
        model_name: Name/path for reporting

    Returns:
        CompatibilityResult with compatibility status and details

    Example:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
        result = validate_model_config(config.to_dict(), "Qwen3-4B-Instruct-2507")
        print(result)
    """
    # Convert PretrainedConfig to dict if needed
    if hasattr(config, "to_dict"):
        config = config.to_dict()

    errors = []
    warnings = []
    config_diff = {}

    # Critical: hidden_size must match exactly
    hidden_size = config.get("hidden_size", 0)
    if hidden_size != ZIMAGE_REFERENCE["hidden_size"]:
        errors.append(
            f"hidden_size mismatch: expected {ZIMAGE_REFERENCE['hidden_size']}, "
            f"got {hidden_size}. Embeddings will be wrong dimension."
        )

    # Critical: must have at least 36 layers (we extract layer -2)
    num_layers = config.get("num_hidden_layers", 0)
    if num_layers < ZIMAGE_REFERENCE["num_hidden_layers"]:
        errors.append(
            f"num_hidden_layers too low: expected >= {ZIMAGE_REFERENCE['num_hidden_layers']}, "
            f"got {num_layers}. Cannot extract penultimate layer."
        )

    # Non-critical differences
    rope_theta = config.get("rope_theta", ZIMAGE_REFERENCE["rope_theta"])
    if rope_theta != ZIMAGE_REFERENCE["rope_theta"]:
        config_diff["rope_theta"] = (ZIMAGE_REFERENCE["rope_theta"], rope_theta)
        warnings.append(
            f"rope_theta differs ({rope_theta} vs {ZIMAGE_REFERENCE['rope_theta']}). "
            "Positional encoding may vary but semantic content should be similar."
        )

    max_pos = config.get("max_position_embeddings", ZIMAGE_REFERENCE["max_position_embeddings"])
    if max_pos != ZIMAGE_REFERENCE["max_position_embeddings"]:
        config_diff["max_position_embeddings"] = (
            ZIMAGE_REFERENCE["max_position_embeddings"],
            max_pos,
        )
        # This is usually fine - just allows longer context
        logger.debug(f"max_position_embeddings differs: {max_pos}")

    vocab_size = config.get("vocab_size", ZIMAGE_REFERENCE["vocab_size"])
    if vocab_size != ZIMAGE_REFERENCE["vocab_size"]:
        config_diff["vocab_size"] = (ZIMAGE_REFERENCE["vocab_size"], vocab_size)
        warnings.append(
            f"vocab_size differs ({vocab_size} vs {ZIMAGE_REFERENCE['vocab_size']}). "
            "Tokenization may vary slightly."
        )

    intermediate_size = config.get("intermediate_size", ZIMAGE_REFERENCE["intermediate_size"])
    if intermediate_size != ZIMAGE_REFERENCE["intermediate_size"]:
        config_diff["intermediate_size"] = (ZIMAGE_REFERENCE["intermediate_size"], intermediate_size)
        warnings.append(
            f"intermediate_size differs ({intermediate_size} vs {ZIMAGE_REFERENCE['intermediate_size']}). "
            "Model may have been trained differently."
        )

    compatible = len(errors) == 0

    return CompatibilityResult(
        compatible=compatible,
        model_name=model_name,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        warnings=warnings,
        errors=errors,
        config_diff=config_diff,
    )


def validate_model_path(
    model_path: str | Path,
    subfolder: str = "text_encoder",
) -> CompatibilityResult:
    """
    Validate a model from a local path or HuggingFace ID.

    Args:
        model_path: Path to model directory or HuggingFace model ID
        subfolder: Subfolder containing the text encoder config

    Returns:
        CompatibilityResult

    Example:
        result = validate_model_path("/path/to/z-image-turbo")
        if not result.compatible:
            print(f"Model not compatible: {result.errors}")
    """
    from transformers import AutoConfig

    model_path = Path(model_path) if not isinstance(model_path, Path) else model_path

    # Determine if local or HuggingFace
    is_local = model_path.exists()

    try:
        if is_local:
            config_path = model_path / subfolder if subfolder else model_path
            config = AutoConfig.from_pretrained(str(config_path), trust_remote_code=True)
            model_name = str(model_path.name)
        else:
            config = AutoConfig.from_pretrained(
                str(model_path),
                subfolder=subfolder if subfolder else None,
                trust_remote_code=True,
            )
            model_name = str(model_path)

        return validate_model_config(config.to_dict(), model_name)

    except Exception as e:
        return CompatibilityResult(
            compatible=False,
            model_name=str(model_path),
            hidden_size=0,
            num_hidden_layers=0,
            warnings=[],
            errors=[f"Failed to load config: {e}"],
            config_diff={},
        )


def check_compatibility(model_path: str | Path, subfolder: str = "text_encoder") -> bool:
    """
    Quick check if model is compatible with Z-Image.

    Args:
        model_path: Path to model or HuggingFace ID
        subfolder: Subfolder containing text encoder

    Returns:
        True if compatible, False otherwise

    Example:
        if check_compatibility("/path/to/model"):
            backend = TransformersBackend.from_pretrained("/path/to/model")
    """
    result = validate_model_path(model_path, subfolder)

    if not result.compatible:
        logger.error(f"Model incompatible: {result.errors}")
    elif result.warnings:
        for warn in result.warnings:
            logger.warning(warn)

    return result.compatible
