"""
LoRA loading utilities for DiT models (pipeline-agnostic).

Supports loading LoRA weights and fusing them into any DiT transformer.
Based on DiffSynth-Studio's LoRA implementation for compatibility.

Usage:
    from llm_dit.utils.lora import load_lora, fuse_lora

    # Simple loading (fuses into model)
    load_lora(pipeline.transformer, "/path/to/lora.safetensors", scale=0.8)

    # Multiple LoRAs
    load_lora(pipeline.transformer, "/path/to/lora1.safetensors", scale=0.5)
    load_lora(pipeline.transformer, "/path/to/lora2.safetensors", scale=0.3)

    # Check what's already fused on a persistent model
    from llm_dit.utils.lora import get_fused_state
    state = get_fused_state(pipeline.transformer)
    if state.matches([("/path/to/lora.safetensors", 0.8)]):
        print("Already fused, skipping")
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors

logger = logging.getLogger(__name__)

# Storage dtypes that indicate quantization -- the model's compute dtype
# is bfloat16 (or float16/float32), not the storage format.
_QUANTIZED_STORAGE_DTYPES = frozenset({
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.int8,
})


# =============================================================================
# LoRA Fusion Tracking
# =============================================================================


@dataclass(frozen=True)
class LoRAFusionRecord:
    """Immutable record of a single LoRA fusion applied to a model."""
    path: str           # Resolved absolute path
    scale: float
    layers_updated: int


@dataclass
class FusedLoRAState:
    """Tracks which LoRAs have been fused into a model.

    Attached to model objects via get_fused_state() so that tracking
    is pipeline-agnostic -- it works regardless of how the pipeline
    stores its model (dict, attribute, local variable).
    """
    records: list[LoRAFusionRecord] = field(default_factory=list)

    def is_fused(self, path: str, scale: float) -> bool:
        """Check if a specific LoRA (path + scale) is already fused."""
        resolved = str(Path(path).resolve())
        return any(r.path == resolved and r.scale == scale for r in self.records)

    def add(self, path: str, scale: float, layers_updated: int) -> None:
        """Record a successful fusion."""
        resolved = str(Path(path).resolve())
        self.records.append(LoRAFusionRecord(resolved, scale, layers_updated))

    def matches(self, requested: list[tuple[str, float]]) -> bool:
        """Check if the fused state exactly matches a list of (path, scale) specs.

        Order-independent comparison using resolved absolute paths.
        """
        if len(requested) != len(self.records):
            return False
        fused_set = {(r.path, r.scale) for r in self.records}
        requested_set = {(str(Path(p).resolve()), s) for p, s in requested}
        return fused_set == requested_set

    @property
    def is_empty(self) -> bool:
        return len(self.records) == 0

    def summary(self) -> str:
        """Human-readable summary for logging."""
        if not self.records:
            return "no LoRAs fused"
        parts = [f"{Path(r.path).name}@{r.scale}" for r in self.records]
        return ", ".join(parts)


def get_fused_state(model: nn.Module) -> FusedLoRAState:
    """Get or create the FusedLoRAState attached to a model.

    The state is stored as model._fused_lora_state so it travels with the
    model object regardless of how the pipeline holds it.
    """
    if not hasattr(model, "_fused_lora_state"):
        model._fused_lora_state = FusedLoRAState()  # type: ignore[attr-defined]
    return model._fused_lora_state  # type: ignore[attr-defined]


class LoRALoader:
    """
    LoRA loader for DiT transformer models.

    Converts various LoRA state dict formats to a standardized format
    and fuses the LoRA weights into the base model.
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the LoRA loader.

        Args:
            device: Device to load tensors onto
            dtype: Data type for tensors
        """
        self.device = device
        self.dtype = dtype

    def get_name_dict(self, lora_state_dict: Dict[str, torch.Tensor]) -> Dict:
        """
        Extract LoRA layer name mappings from state dict.

        Handles different LoRA naming conventions:
        - lora_up/lora_down (Kohya format)
        - lora_A/lora_B (PEFT/diffusers format)

        Args:
            lora_state_dict: Raw LoRA state dict

        Returns:
            Dict mapping target layer name -> (lora_B_key, lora_A_key)
        """
        lora_name_dict = {}

        for key in lora_state_dict:
            # Determine naming convention
            if ".lora_up." in key:
                lora_A_key = "lora_down"
                lora_B_key = "lora_up"
            else:
                lora_A_key = "lora_A"
                lora_B_key = "lora_B"

            if lora_B_key not in key:
                continue

            # Parse layer name
            keys = key.split(".")

            # Handle nested structure (e.g., lora_B.weight)
            if len(keys) > keys.index(lora_B_key) + 2:
                keys.pop(keys.index(lora_B_key) + 1)

            keys.pop(keys.index(lora_B_key))

            # Remove common prefixes
            if keys[0] == "diffusion_model":
                keys.pop(0)
            if keys[0] == "transformer":
                keys.pop(0)

            # Remove weight suffix
            keys.pop(-1)

            target_name = ".".join(keys)
            lora_name_dict[target_name] = (key, key.replace(lora_B_key, lora_A_key))

        return lora_name_dict

    def convert_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        suffix: str = ".weight",
    ) -> Dict[str, torch.Tensor]:
        """
        Convert LoRA state dict to standardized format.

        Args:
            state_dict: Raw LoRA state dict
            suffix: Weight key suffix

        Returns:
            Standardized state dict with .lora_A.weight and .lora_B.weight keys
        """
        name_dict = self.get_name_dict(state_dict)
        state_dict_ = {}

        for name in name_dict:
            lora_B_key, lora_A_key = name_dict[name]
            weight_up = state_dict[lora_B_key]
            weight_down = state_dict[lora_A_key]
            state_dict_[name + f".lora_B{suffix}"] = weight_up
            state_dict_[name + f".lora_A{suffix}"] = weight_down

        return state_dict_

    def fuse_lora_to_base_model(
        self,
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
        alpha: float = 1.0,
    ) -> int:
        """
        Fuse LoRA weights into the base model.

        Computes: weight = weight + alpha * (lora_B @ lora_A)

        Args:
            model: Target model (e.g., transformer)
            state_dict: LoRA state dict (raw format)
            alpha: LoRA scale factor

        Returns:
            Number of layers updated
        """
        updated_num = 0
        dequantized_modules: list[nn.Module] = []
        state_dict = self.convert_state_dict(state_dict)

        # Get unique layer names
        lora_layer_names = set(
            [i.replace(".lora_B.weight", "") for i in state_dict if i.endswith(".lora_B.weight")]
        )

        logger.debug(f"Found {len(lora_layer_names)} LoRA layers to fuse")

        for name, module in model.named_modules():
            if name in lora_layer_names:
                lora_B_key = name + ".lora_B.weight"
                lora_A_key = name + ".lora_A.weight"

                weight_up = state_dict[lora_B_key].to(device=self.device, dtype=self.dtype)
                weight_down = state_dict[lora_A_key].to(device=self.device, dtype=self.dtype)

                # Handle conv2d LoRA (4D tensors)
                if len(weight_up.shape) == 4:
                    weight_up = weight_up.squeeze(3).squeeze(2)
                    weight_down = weight_down.squeeze(3).squeeze(2)
                    weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                else:
                    weight_lora = alpha * torch.mm(weight_up, weight_down)

                # Fuse into base model
                state_dict_base = module.state_dict()
                base_weight = state_dict_base["weight"]

                # Dequantize quantized tensors (e.g., torchao Float8Tensor)
                # before LoRA merge. Float8Tensor.to(dtype=bf16) returns another
                # Float8Tensor, and aten.add is not implemented for that type.
                is_quantized = type(base_weight) is not torch.Tensor
                if is_quantized:
                    if hasattr(base_weight, "dequantize"):
                        base_weight = base_weight.dequantize()
                    else:
                        base_weight = base_weight.float()

                merged_weight = (
                    base_weight.to(device=self.device, dtype=self.dtype) + weight_lora
                )

                if is_quantized:
                    # Cannot use load_state_dict on quantized modules --
                    # Float8Tensor.copy_() expects qdata on the source tensor.
                    # Directly replace the parameter instead.
                    module.weight = nn.Parameter(
                        merged_weight,
                        requires_grad=module.weight.requires_grad,
                    )
                    dequantized_modules.append(module)
                else:
                    state_dict_base["weight"] = merged_weight
                    module.load_state_dict(state_dict_base)

                updated_num += 1

        # Re-quantize dequantized layers to reclaim VRAM.
        # Without this, fused layers stay at bf16 (~17GB for the whole
        # transformer) instead of fp8 (~9GB), causing OOM on the next
        # request when the encoder shuttle needs GPU space.
        if dequantized_modules:
            from torchao.quantization import float8_weight_only, quantize_

            logger.info(
                f"Re-quantizing {len(dequantized_modules)} LoRA-affected "
                f"layers to restore fp8..."
            )
            for mod in dequantized_modules:
                quantize_(mod, float8_weight_only())
            logger.info("Re-quantization complete, VRAM reclaimed")

        logger.info(f"Fused {updated_num} LoRA layers (alpha={alpha})")
        return updated_num


def _infer_model_device_dtype(
    model: nn.Module,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[Union[str, torch.device], torch.dtype]:
    """Infer device and dtype from model parameters if not specified.

    For quantized models (torchao Float8Tensor, int8, etc.), the parameter's
    .dtype returns the storage format (e.g., uint8, float8_e4m3fn) rather than
    the compute dtype. We detect these and return bfloat16 as the compute dtype
    so that LoRA math happens in the correct precision.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"

    if dtype is None:
        try:
            param = next(model.parameters())
            inferred = param.dtype
            # Quantized parameters report storage dtype (uint8, float8_e4m3fn, etc.)
            # but LoRA computation needs the compute dtype.
            # Note: nn.Parameter is a torch.Tensor subclass, so we use isinstance()
            # to avoid false positives. Non-Tensor types (e.g., torchao Float8Tensor)
            # are NOT subclasses of torch.Tensor.
            is_quantized_type = not isinstance(param, torch.Tensor)
            is_quantized_dtype = inferred in _QUANTIZED_STORAGE_DTYPES
            if is_quantized_type or is_quantized_dtype:
                dtype = torch.bfloat16
                logger.debug(
                    f"Quantized model detected (param type={type(param).__name__}, "
                    f"storage dtype={inferred}), using compute dtype={dtype}"
                )
            else:
                dtype = inferred
        except StopIteration:
            dtype = torch.float32

    return device, dtype


def load_lora(
    model: nn.Module,
    lora_path: Union[str, Path],
    scale: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> int:
    """
    Load and fuse a LoRA into a model.

    Args:
        model: Target model (typically pipeline.transformer)
        lora_path: Path to LoRA weights (.safetensors or .bin)
        scale: LoRA scale factor (alpha)
        device: Device for computation (defaults to model device)
        dtype: Data type (defaults to model dtype)

    Returns:
        Number of layers updated

    Example:
        load_lora(pipeline.transformer, "anime_style.safetensors", scale=0.7)
    """
    lora_path = Path(lora_path)
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    device, dtype = _infer_model_device_dtype(model, device, dtype)

    logger.info(f"Loading LoRA: {lora_path} (scale={scale})")

    # Load state dict (safetensors only)
    if lora_path.suffix != ".safetensors":
        raise ValueError(
            f"Expected .safetensors file, got {lora_path.suffix}. "
            f"Convert with: uv run python scripts/convert_to_safetensors.py {lora_path}"
        )
    state_dict = load_safetensors(str(lora_path))

    # Fuse
    loader = LoRALoader(device=device, dtype=dtype)
    updated = loader.fuse_lora_to_base_model(model, state_dict, alpha=scale)

    # Record the fusion on the model for re-fusion prevention
    fused_state = get_fused_state(model)
    fused_state.add(str(lora_path), scale, updated)
    logger.debug(f"Recorded fusion: {fused_state.summary()}")

    return updated


def fuse_lora(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    scale: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> int:
    """
    Fuse LoRA weights from a state dict into a model.

    Same as load_lora but takes a state dict directly instead of a file path.

    Args:
        model: Target model
        state_dict: LoRA state dict
        scale: LoRA scale factor
        device: Device for computation
        dtype: Data type

    Returns:
        Number of layers updated
    """
    device, dtype = _infer_model_device_dtype(model, device, dtype)
    loader = LoRALoader(device=device, dtype=dtype)
    return loader.fuse_lora_to_base_model(model, state_dict, alpha=scale)


def clear_lora(model: nn.Module) -> None:
    """
    Clear LoRA weights from a model.

    NOTE: Fused LoRAs cannot be cleared - they are permanently merged
    into the base weights. To clear, you must reload the original model.

    This function is provided for API compatibility but will raise an error.
    """
    raise NotImplementedError(
        "Fused LoRAs cannot be cleared. Reload the model to remove LoRA weights. "
        "For unfusable LoRAs, consider using diffusers' PEFT-based LoRA loading."
    )


def parse_lora_spec(spec: str) -> tuple[str, float]:
    """
    Parse a LoRA specification string.

    Format: path[:scale]
    Examples:
        "lora.safetensors" -> ("lora.safetensors", 1.0)
        "lora.safetensors:0.8" -> ("lora.safetensors", 0.8)

    Args:
        spec: LoRA specification string

    Returns:
        Tuple of (path, scale)
    """
    if ":" in spec:
        parts = spec.rsplit(":", 1)
        path = parts[0]
        try:
            scale = float(parts[1])
        except ValueError:
            # Colon was part of path (e.g., Windows drive letter)
            path = spec
            scale = 1.0
    else:
        path = spec
        scale = 1.0

    return path, scale
