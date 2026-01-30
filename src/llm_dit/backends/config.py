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
        dtype: Model dtype as string ("bfloat16", "float16", "float32")
        device: Target device ("cuda", "cpu", "mps", "auto")
        trust_remote_code: Allow loading custom model code
        use_flash_attention: Enable flash attention if available
        tensor_parallel_size: For vLLM/SGLang distributed inference
        quantization: Quantization mode ("none", "4bit", "8bit", "int8_dynamic")

    Example:
        config = BackendConfig(
            model_path="Tongyi-MAI/Z-Image-Turbo",
            max_length=512,
            dtype="bfloat16",
        )
    """

    backend_type: Literal["transformers", "vllm", "sglang", "mlx"] = "transformers"
    model_path: str = ""
    subfolder: str = "text_encoder"  # Z-Image stores encoder in subfolder
    max_length: int = 2048  # Increased from 512 - allows longer prompts
    dtype: str = "bfloat16"
    device: str = field(default_factory=_detect_best_device)
    trust_remote_code: bool = True
    use_flash_attention: bool = True
    tensor_parallel_size: int = 1  # For vLLM/SGLang
    quantization: str = "none"  # none, 4bit, 8bit, fp8, int8 (TorchAO), int8_dynamic (deprecated)

    def __post_init__(self):
        """Validate and migrate deprecated settings."""
        self._validate_quantization()

    def _validate_quantization(self):
        """Check for deprecated or incompatible quantization settings."""
        import logging

        logger = logging.getLogger(__name__)

        # Check for deprecated int8_dynamic with bfloat16
        if self.quantization == "int8_dynamic":
            dtype = self.get_dtype()
            if dtype != torch.float32:
                logger.warning(
                    f"quantization='int8_dynamic' is incompatible with {self.dtype}. "
                    "PyTorch dynamic quantization requires float32. "
                    "Auto-migrating to '8bit' (bitsandbytes). "
                    "Update your config to use 'fp8', 'int8', or '8bit' instead."
                )
                self.quantization = "8bit"

    def get_dtype(self) -> torch.dtype:
        """Convert string dtype to torch.dtype."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "float": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)

    def get_device(self) -> torch.device:
        """Get torch device."""
        return torch.device(self.device)

    def needs_post_load_quantization(self) -> bool:
        """Check if post-load quantization is needed.

        Returns:
            True if int8_dynamic quantization should be applied after model loading.
        """
        return self.quantization == "int8_dynamic"

    def apply_post_load_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply post-load quantization (int8_dynamic) to the model.

        Uses torch.ao.quantization.quantize_dynamic() to apply int8 dynamic
        quantization to all Linear layers. This provides ~50% VRAM reduction
        with minimal quality impact for LLMs.

        Args:
            model: The loaded model to quantize

        Returns:
            Quantized model (in-place modification)

        Raises:
            ValueError: If quantization mode is not int8_dynamic
        """
        if self.quantization != "int8_dynamic":
            raise ValueError(
                f"apply_post_load_quantization() only valid for int8_dynamic, "
                f"got {self.quantization}"
            )

        import logging

        import torch.ao.quantization as tq

        logger = logging.getLogger(__name__)
        logger.info("Applying int8 dynamic quantization (torchao)...")

        # Quantize all Linear layers to int8
        model = tq.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8,
        )

        logger.info("  int8 dynamic quantization applied successfully")
        return model

    def get_quantization_config(self):
        """Convert string quantization mode to BitsAndBytesConfig if needed.

        This method handles the conversion from human-readable quantization
        strings ("4bit", "8bit", "fp8") to proper BitsAndBytesConfig objects
        that transformers from_pretrained() expects.

        Returns:
            BitsAndBytesConfig for the quantization mode, or None if no
            quantization is needed (e.g., "none" or "int8_dynamic" which
            uses torch.ao post-load quantization instead).
        """
        # No quantization or uses post-load quantization (torch.ao)
        if self.quantization in ("none", "int8_dynamic"):
            return None

        try:
            from transformers import BitsAndBytesConfig
        except ImportError:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"BitsAndBytesConfig not available. Cannot apply quantization={self.quantization}. "
                "Install with: pip install bitsandbytes"
            )
            return None

        import logging

        logger = logging.getLogger(__name__)

        if self.quantization == "4bit":
            logger.info("Creating 4-bit quantization config (bitsandbytes)")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.get_dtype(),
                bnb_4bit_use_double_quant=True,  # Nested quantization for extra savings
            )
        elif self.quantization == "8bit":
            logger.info("Creating 8-bit quantization config (bitsandbytes)")
            return BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization == "fp8":
            # FP8 in bitsandbytes is still experimental, fall back to 8-bit for now
            logger.info("FP8 requested, using 8-bit quantization (bitsandbytes)")
            return BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization == "int8":
            # Alias for 8bit
            logger.info("Creating int8 quantization config (bitsandbytes)")
            return BitsAndBytesConfig(load_in_8bit=True)
        else:
            logger.warning(f"Unknown quantization mode: {self.quantization}, ignoring")
            return None

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
            "max_length": 2048,  # Increased from 512 - allows longer prompts
            "dtype": "bfloat16",
        }
        defaults.update(kwargs)
        return cls(**defaults)
