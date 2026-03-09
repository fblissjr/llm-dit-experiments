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

    def _detect_format(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Detect LoRA format from state dict keys.

        Returns:
            "lokr" if any key contains .lokr_w1, "lora" otherwise.
        """
        for key in state_dict:
            if ".lokr_w1" in key:
                return "lokr"
        return "lora"

    def _get_lokr_name_dict(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, Tuple[str, str]]:
        """Extract LoKR layer name mappings from state dict.

        Iterates keys ending with .lokr_w1, strips common prefixes
        (diffusion_model., transformer.) and the .lokr_w1 suffix.

        Returns:
            Dict mapping target_name -> (w1_key, w2_key)
        """
        name_dict: Dict[str, Tuple[str, str]] = {}

        for key in state_dict:
            if not key.endswith(".lokr_w1"):
                continue

            w1_key = key
            w2_key = key.replace(".lokr_w1", ".lokr_w2")

            # Strip .lokr_w1 suffix to get module path
            target = key.removesuffix(".lokr_w1")

            # Remove common prefixes
            if target.startswith("diffusion_model."):
                target = target.removeprefix("diffusion_model.")
            if target.startswith("transformer."):
                target = target.removeprefix("transformer.")

            name_dict[target] = (w1_key, w2_key)

        return name_dict

    def _fuse_lokr_to_base_model(
        self,
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
        alpha: float = 1.0,
    ) -> int:
        """Fuse LoKR (Kronecker product) weights into the base model.

        For full-matrix LoKR: weight += alpha * kron(lokr_w1, lokr_w2)
        The stored .alpha tensor is ignored for full-matrix LoKR (scale = 1.0).

        Args:
            model: Target model
            state_dict: Raw LoKR state dict
            alpha: User-specified scale factor

        Returns:
            Number of layers updated
        """
        updated_num = 0
        requant_num = 0
        lokr_name_dict = self._get_lokr_name_dict(state_dict)

        _requant_config = None
        _quantize_fn = None

        logger.debug(f"Found {len(lokr_name_dict)} LoKR layers to fuse")

        for name, module in model.named_modules():
            if name not in lokr_name_dict:
                continue

            w1_key, w2_key = lokr_name_dict[name]
            w1 = state_dict[w1_key].to(device=self.device, dtype=self.dtype)
            w2 = state_dict[w2_key].to(device=self.device, dtype=self.dtype)

            delta_w = alpha * torch.kron(w1, w2)

            # Fuse into base model
            state_dict_base = module.state_dict()
            base_weight = state_dict_base["weight"]

            is_quantized = type(base_weight) is not torch.Tensor
            if is_quantized:
                if hasattr(base_weight, "dequantize"):
                    base_weight = base_weight.dequantize()
                else:
                    base_weight = base_weight.float()

            merged_weight = (
                base_weight.to(device=self.device, dtype=self.dtype) + delta_w
            )

            if is_quantized:
                module.weight = nn.Parameter(
                    merged_weight,
                    requires_grad=module.weight.requires_grad,
                )
                del w1, w2, delta_w, base_weight, state_dict_base

                if _requant_config is None:
                    from torchao.quantization import Float8WeightOnlyConfig, quantize_
                    _requant_config = Float8WeightOnlyConfig()
                    _quantize_fn = quantize_
                _quantize_fn(module, _requant_config)
                requant_num += 1

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                state_dict_base["weight"] = merged_weight
                module.load_state_dict(state_dict_base)

            updated_num += 1

        if requant_num:
            logger.info(f"Re-quantized {requant_num} layers to fp8 during LoKR fusion")

        logger.info(f"Fused {updated_num} LoKR layers (alpha={alpha})")
        return updated_num

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
        # Dispatch by format
        fmt = self._detect_format(state_dict)
        if fmt == "lokr":
            return self._fuse_lokr_to_base_model(model, state_dict, alpha)

        updated_num = 0
        requant_num = 0
        state_dict = self.convert_state_dict(state_dict)

        # Lazy-init: re-quantization config loaded on first quantized layer
        _requant_config = None
        _quantize_fn = None

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

                    # Free transient tensors before re-quantization to reduce
                    # peak memory. merged_weight is still referenced by
                    # module.weight so deleting the local name is safe.
                    del weight_up, weight_down, weight_lora, base_weight, state_dict_base

                    # Re-quantize immediately to prevent VRAM accumulation.
                    # Without per-layer re-quant, fused layers stay at bf16 (2x
                    # memory each), and a large LoRA (e.g., rank-384 distilled)
                    # can balloon the model from ~13GB fp8 to ~26GB bf16 mid-loop.
                    if _requant_config is None:
                        from torchao.quantization import Float8WeightOnlyConfig, quantize_
                        _requant_config = Float8WeightOnlyConfig()
                        _quantize_fn = quantize_
                    _quantize_fn(module, _requant_config)
                    requant_num += 1

                    # Release CUDA cached blocks every layer to prevent
                    # fragmentation. Each layer's dequant/requant cycle creates
                    # ~70-134 MiB of transient bf16 tensors that fragment the
                    # CUDA pool. Without per-layer cleanup, ~20 layers can
                    # exhaust contiguous free memory even though total allocated
                    # has barely grown. The ~1ms overhead per call is negligible
                    # compared to the fusion loop's total runtime.
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    state_dict_base["weight"] = merged_weight
                    module.load_state_dict(state_dict_base)

                updated_num += 1

        if requant_num:
            logger.info(f"Re-quantized {requant_num} layers to fp8 during fusion")

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


def fuse_lora_to_state_dict(
    state_dict: Dict[str, torch.Tensor],
    lora_paths: list[Union[str, Path]],
    lora_scales: list[float],
    device: str = "cpu",
    weight_scales: Optional[Dict[str, torch.Tensor]] = None,
) -> Union[Dict[str, torch.Tensor], tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
    """Fuse LoRA deltas into a state dict (supports fp8 + bf16 weights).

    Matches official LTX-2 fuse_loras.py pattern:
    - Scaled fp8 weights (with weight_scales): dequantize, add delta in f32,
      re-quantize with new per-tensor scale. This is critical for the distilled
      pipeline where raw fp8 values (~[-448, 448]) would dwarf the LoRA delta
      (~0.001) without proper dequantization.
    - Unscaled fp8 weights: upcast to bf16, add delta, downcast back to fp8
    - bf16 weights: direct addition
    - LoRA deltas always computed in bf16

    This operates on the raw state dict BEFORE load_state_dict(), which is
    critical for fp8-cast models where native fp8 tensors can't do arithmetic.

    Args:
        state_dict: Base model state dict (NOT mutated -- cloned internally).
        lora_paths: List of paths to LoRA .safetensors files.
        lora_scales: Scale factor for each LoRA (same length as lora_paths).
        device: Device for delta computation (typically "cpu").
        weight_scales: Optional dict mapping weight keys (e.g. "blocks.0.attn.qkv.weight")
            to per-tensor scale tensors for scaled FP8 dequantization. When provided,
            returns a tuple of (state_dict, updated_weight_scales).

    Returns:
        When weight_scales is None: new state dict with fused weights.
        When weight_scales is provided: (new_state_dict, updated_weight_scales).
    """
    if len(lora_paths) != len(lora_scales):
        raise ValueError(
            f"Number of LoRA paths ({len(lora_paths)}) must match "
            f"number of scales ({len(lora_scales)})"
        )

    # Clone so we don't mutate the cache
    result = {k: v.clone() for k, v in state_dict.items()}
    new_weight_scales: Dict[str, torch.Tensor] = dict(weight_scales) if weight_scales else {}

    loader = LoRALoader(device=device, dtype=torch.bfloat16)

    for lora_path, scale in zip(lora_paths, lora_scales):
        lora_path = Path(lora_path)
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA file not found: {lora_path}")
        if lora_path.suffix != ".safetensors":
            raise ValueError(f"Expected .safetensors file, got {lora_path.suffix}")

        raw_sd = load_safetensors(str(lora_path))

        fmt = loader._detect_format(raw_sd)
        if fmt == "lokr":
            lokr_dict = loader._get_lokr_name_dict(raw_sd)
            for name, (w1_key, w2_key) in lokr_dict.items():
                weight_key = name + ".weight"
                if weight_key not in result:
                    continue
                w1 = raw_sd[w1_key].to(device=device, dtype=torch.bfloat16)
                w2 = raw_sd[w2_key].to(device=device, dtype=torch.bfloat16)
                delta = scale * torch.kron(w1, w2)
                result[weight_key], new_weight_scales = _fuse_delta(
                    result[weight_key], delta, weight_key, new_weight_scales,
                )
        else:
            std_sd = loader.convert_state_dict(raw_sd)
            layer_names = {
                k.replace(".lora_B.weight", "")
                for k in std_sd if k.endswith(".lora_B.weight")
            }
            for name in layer_names:
                weight_key = name + ".weight"
                if weight_key not in result:
                    continue
                lora_B = std_sd[name + ".lora_B.weight"].to(device=device, dtype=torch.bfloat16)
                lora_A = std_sd[name + ".lora_A.weight"].to(device=device, dtype=torch.bfloat16)
                if len(lora_B.shape) == 4:
                    lora_B = lora_B.squeeze(3).squeeze(2)
                    lora_A = lora_A.squeeze(3).squeeze(2)
                    delta = scale * torch.mm(lora_B, lora_A).unsqueeze(2).unsqueeze(3)
                else:
                    delta = scale * torch.mm(lora_B, lora_A)
                result[weight_key], new_weight_scales = _fuse_delta(
                    result[weight_key], delta, weight_key, new_weight_scales,
                )

        logger.info(f"State-dict LoRA fusion: {lora_path.name} (scale={scale})")

    if weight_scales is not None:
        return result, new_weight_scales
    return result


def _fuse_delta(
    weight: torch.Tensor,
    delta: torch.Tensor,
    weight_key: str,
    weight_scales: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Fuse a LoRA delta into a weight tensor, handling scaled fp8/unscaled fp8/bf16.

    For scaled fp8 (weight_key found in weight_scales): dequantize to f32,
    add delta, re-quantize to fp8 with a new per-tensor scale. This matches
    the official LTX-2 `_fuse_delta_with_scaled_fp8` pattern.

    For unscaled fp8: upcast to bf16, add delta, downcast back to fp8.
    For bf16/fp32: direct addition.

    Returns:
        (fused_weight, updated_weight_scales dict).
    """
    is_fp8 = weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    has_scale = weight_key in weight_scales

    if is_fp8 and has_scale:
        # Scaled fp8: dequant -> fuse in f32 -> re-quant with new scale
        from llm_dit.quantization.fp8_cast import quantize_to_fp8_per_tensor

        ws = weight_scales[weight_key]
        real_weight = weight.to(torch.float32) * ws.to(torch.float32)
        merged = real_weight + delta.to(torch.float32)
        new_fp8, new_scale = quantize_to_fp8_per_tensor(merged)
        weight_scales[weight_key] = new_scale
        return new_fp8, weight_scales
    elif is_fp8:
        # Unscaled fp8 (cast-only): simple upcast/downcast
        original_dtype = weight.dtype
        merged = weight.to(torch.bfloat16) + delta.to(torch.bfloat16)
        return merged.to(original_dtype), weight_scales
    else:
        # bf16/fp32: direct addition
        return weight + delta.to(weight.dtype), weight_scales


def _fuse_delta_into_weight(weight: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Legacy: fuse a LoRA delta without weight_scale handling.

    Kept for backward compatibility with tests. New code should use _fuse_delta.
    """
    if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        original_dtype = weight.dtype
        merged = weight.to(torch.bfloat16) + delta.to(torch.bfloat16)
        return merged.to(original_dtype)
    else:
        return weight + delta.to(weight.dtype)


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


