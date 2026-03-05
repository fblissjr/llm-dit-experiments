"""GGMLTensor: torch.Tensor subclass that preserves GGML quantization metadata.

Extracted from ComfyUI-GGUF (City96, Apache-2.0), stripped of ComfyUI deps.

Key behaviors:
- `.shape` returns logical shape (not byte-layout shape)
- `.to()` preserves quantization metadata
- `.clone()` / `.detach()` are no-ops (prevent accidental dequant copies)
"""

import torch


class GGMLTensor(torch.Tensor):
    """Tensor subclass that carries GGML quantization type and logical shape."""

    def __init__(self, *args, tensor_type, tensor_shape, **kwargs):
        super().__init__()
        self.tensor_type = tensor_type
        self.tensor_shape = tensor_shape

    def __new__(cls, *args, tensor_type, tensor_shape, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.data.shape)
        return new

    def clone(self, *args, **kwargs):
        return self

    def detach(self, *args, **kwargs):
        return self

    def new_empty(self, size, *args, **kwargs):
        new_tensor = super().new_empty(size, *args, **kwargs)
        return GGMLTensor(
            new_tensor,
            tensor_type=getattr(self, "tensor_type", None),
            tensor_shape=size,
        )

    @property
    def shape(self):
        if not hasattr(self, "tensor_shape"):
            self.tensor_shape = self.size()
        return self.tensor_shape
