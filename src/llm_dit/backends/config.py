"""
Backend configuration dataclass.

Provides a unified configuration interface for all LLM backends,
with sensible defaults for Z-Image (Qwen3-4B).
"""

from dataclasses import dataclass, field
from typing import Literal

import torch


def _detect_best_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


@dataclass
class BackendConfig:
    """
    Configuration for LLM text encoder backends.

    Attributes:
        backend_type: Which backend to use ("transformers", "vllm", "sglang", "mlx")
        model_path: Path to model (local or HuggingFace hub ID)
        max_length: Maximum sequence length for tokenization
        torch_dtype: Model dtype as string ("bfloat16", "float16", "float32")
        device: Target device ("cuda", "cpu", "mps", "auto")
        trust_remote_code: Allow loading custom model code
        use_flash_attention: Enable flash attention if available
        tensor_parallel_size: For vLLM/SGLang distributed inference

    Example:
        config = BackendConfig(
            model_path="Tongyi-MAI/Z-Image-Turbo",
            max_length=512,
            torch_dtype="bfloat16",
        )
    """

    backend_type: Literal["transformers", "vllm", "sglang", "mlx"] = "transformers"
    model_path: str = ""
    subfolder: str = "text_encoder"  # Z-Image stores encoder in subfolder
    max_length: int = 512
    torch_dtype: str = "bfloat16"
    device: str = field(default_factory=_detect_best_device)
    trust_remote_code: bool = True
    use_flash_attention: bool = True
    tensor_parallel_size: int = 1  # For vLLM/SGLang

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "float": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.bfloat16)

    def get_device(self) -> torch.device:
        """Get torch device."""
        return torch.device(self.device)

    @classmethod
    def for_z_image(cls, model_path: str, **kwargs) -> "BackendConfig":
        """
        Create config optimized for Z-Image (Qwen3-4B encoder).

        Args:
            model_path: Path to Z-Image model or HuggingFace ID
            **kwargs: Override any default settings

        Returns:
            BackendConfig with Z-Image defaults
        """
        defaults = {
            "model_path": model_path,
            "subfolder": "text_encoder",
            "max_length": 512,
            "torch_dtype": "bfloat16",
        }
        defaults.update(kwargs)
        return cls(**defaults)
