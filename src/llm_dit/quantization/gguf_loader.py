"""GGUF file loader for LLM-DiT transformer models.

Adapted from ComfyUI-GGUF (City96, Apache-2.0), stripped of ComfyUI deps.

Loads GGUF files into a state dict of GGMLTensors. Handles:
- Memory-mapped reading via gguf.GGUFReader
- Key remapping from GGUF namespace to our transformer keys
- Eager dequant of 1D tensors (biases, norms) to float32
"""

import logging
import warnings

import gguf
import torch

from .gguf_dequant import dequantize_tensor, is_quantized
from .gguf_tensor import GGMLTensor

logger = logging.getLogger(__name__)

# GGUF files from Unsloth use "model.diffusion_model." prefix for transformer keys.
DEFAULT_PREFIX = "model.diffusion_model."


def get_orig_shape(reader, tensor_name):
    """Read original shape metadata stored by ComfyUI GGUF converter."""
    field_key = f"comfy.gguf.orig_shape.{tensor_name}"
    field = reader.get_field(field_key)
    if field is None:
        return None
    if len(field.types) != 2 or field.types[0] != gguf.GGUFValueType.ARRAY or field.types[1] != gguf.GGUFValueType.INT32:
        raise TypeError(f"Bad original shape metadata for {field_key}: Expected ARRAY of INT32, got {field.types}")
    return torch.Size(tuple(int(field.parts[part_idx][0]) for part_idx in field.data))


def get_gguf_metadata(reader):
    """Extract simple scalar metadata fields from GGUF file."""
    metadata = {}
    for field_name in reader.fields:
        try:
            field = reader.get_field(field_name)
            if len(field.types) == 1:
                if field.types[0] == gguf.GGUFValueType.STRING:
                    metadata[field_name] = str(field.parts[field.data[-1]], "utf-8")
                elif field.types[0] == gguf.GGUFValueType.INT32:
                    metadata[field_name] = int(field.parts[field.data[-1]])
                elif field.types[0] == gguf.GGUFValueType.F32:
                    metadata[field_name] = float(field.parts[field.data[-1]])
                elif field.types[0] == gguf.GGUFValueType.BOOL:
                    metadata[field_name] = bool(field.parts[field.data[-1]])
        except Exception:
            continue
    return metadata


def gguf_sd_loader(path: str, handle_prefix: str = DEFAULT_PREFIX):
    """Load GGUF file into a state dict of GGMLTensors.

    Args:
        path: Path to the GGUF file.
        handle_prefix: Key prefix to strip (e.g., "model.diffusion_model.").

    Returns:
        Tuple of (state_dict, extra_info) where extra_info contains
        arch_str and metadata.
    """
    reader = gguf.GGUFReader(path)

    # Check if tensors have the expected prefix
    has_prefix = False
    if handle_prefix is not None:
        prefix_len = len(handle_prefix)
        tensor_names = set(tensor.name for tensor in reader.tensors)
        has_prefix = any(s.startswith(handle_prefix) for s in tensor_names)

    tensors = []
    for tensor in reader.tensors:
        sd_key = tensor_name = tensor.name
        if has_prefix:
            if not tensor_name.startswith(handle_prefix):
                continue
            sd_key = tensor_name[prefix_len:]
        tensors.append((sd_key, tensor))

    # Read architecture metadata
    arch_field = reader.get_field("general.architecture")
    arch_str = None
    if arch_field is not None and len(arch_field.types) == 1 and arch_field.types[0] == gguf.GGUFValueType.STRING:
        arch_str = str(arch_field.parts[arch_field.data[-1]], encoding="utf-8")

    # Main loading loop
    state_dict = {}
    qtype_dict = {}
    for sd_key, tensor in tensors:
        tensor_name = tensor.name

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(tensor.data)

        shape = get_orig_shape(reader, tensor_name)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))

        # F32/F16 tensors can be directly viewed
        if tensor.tensor_type in {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}:
            torch_tensor = torch_tensor.view(*shape)

        state_dict[sd_key] = GGMLTensor(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)

        # 1D tensors (biases, norms) shouldn't stay quantized
        if len(shape) <= 1 and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            state_dict[sd_key] = dequantize_tensor(state_dict[sd_key], dtype=torch.float32)

        tensor_type_str = getattr(tensor.tensor_type, "name", repr(tensor.tensor_type))
        qtype_dict[tensor_type_str] = qtype_dict.get(tensor_type_str, 0) + 1

    logger.info("GGUF qtypes: %s", ", ".join(f"{k} ({v})" for k, v in qtype_dict.items()))

    # Mark largest quantized tensor for VRAM estimation
    qsd = {k: v for k, v in state_dict.items() if is_quantized(v)}
    if len(qsd) > 0:
        max_key = max(qsd.keys(), key=lambda k: qsd[k].numel())
        state_dict[max_key].is_largest_weight = True

    extra = {
        "arch_str": arch_str,
        "metadata": get_gguf_metadata(reader),
    }
    return state_dict, extra


def detect_v2_from_state_dict(state_dict: dict) -> bool:
    """Detect LTX-2.3 (V2) model from state dict keys.

    V2 models have prompt_scale_shift_table in transformer blocks and
    cross_attention_adaln support (9-element scale_shift_table per block
    instead of 6).
    """
    for key in state_dict:
        if "prompt_scale_shift_table" in key:
            return True
        if "to_gate_logits" in key:
            return True
    return False
