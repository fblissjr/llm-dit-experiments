"""GGMLLinear: nn.Linear replacement that dequantizes GGML weights on the fly.

Peak VRAM = all quantized weights + ONE layer's bf16 weight + activations.
LoRA deltas are applied to the dequantized weight before matmul.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gguf_dequant import dequantize_tensor, is_quantized
from .gguf_tensor import GGMLTensor

logger = logging.getLogger(__name__)


class GGMLLinear(nn.Linear):
    """Linear layer that stores GGML-quantized weights and dequantizes per-forward.

    Usage:
        1. Build model with GGMLLinear layers (or replace nn.Linear post-hoc)
        2. Load GGUF state dict -- _load_from_state_dict accepts GGMLTensors
        3. Forward: dequantize weight -> F.linear -> free dequantized weight

    LoRA support:
        Attach LoRA deltas via model._ggml_lora_deltas dict mapping
        parameter names to (delta_weight, scale) tuples. The delta is applied
        to the dequantized weight before matmul.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        # Skip nn.Linear's __init__ to avoid allocating full-size weight
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = None
        self.bias = None
        # LoRA per-forward fields (set externally via attach_lora_deltas)
        self.lora_delta: torch.Tensor | None = None
        self.lora_scale: float | None = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Accept GGMLTensors without shape validation."""
        for k, v in state_dict.items():
            suffix = k[len(prefix):]
            if suffix == "weight":
                self.weight = nn.Parameter(v, requires_grad=False)
            elif suffix == "bias" and v is not None:
                self.bias = nn.Parameter(v, requires_grad=False)

        if self.weight is None:
            missing_keys.append(prefix + "weight")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dtype = input.dtype
        weight = self._get_dequantized_weight(dtype)
        return F.linear(input, weight, self._get_bias(dtype))

    def _get_dequantized_weight(self, dtype: torch.dtype) -> torch.Tensor:
        """Dequantize weight to target dtype. Applies LoRA if attached."""
        if not is_quantized(self.weight):
            weight = self.weight.to(dtype)
        else:
            weight = dequantize_tensor(self.weight, dtype=dtype)
            # Prevent GGMLTensor from propagating through matmul
            if isinstance(weight, GGMLTensor):
                weight = torch.Tensor(weight)

        # Apply LoRA delta if attached (per-forward, no weight mutation)
        if self.lora_delta is not None and self.lora_scale:
            weight = weight + self.lora_scale * self.lora_delta.to(dtype)

        return weight

    def _get_bias(self, dtype: torch.dtype):
        if self.bias is None:
            return None
        if is_quantized(self.bias):
            return dequantize_tensor(self.bias, dtype=dtype)
        return self.bias.to(dtype)


def replace_linear_with_ggml(model: nn.Module) -> int:
    """Replace all nn.Linear layers in model with GGMLLinear.

    Returns the number of layers replaced.
    """
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and not isinstance(module, GGMLLinear):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = model if parent_name == "" else dict(model.named_modules())[parent_name]

            ggml_linear = GGMLLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            )
            setattr(parent, child_name, ggml_linear)
            count += 1

    return count
