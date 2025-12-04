"""
LoRA loading utilities for Z-Image DiT.

Supports loading LoRA weights and fusing them into the DiT transformer.
Based on DiffSynth-Studio's LoRA implementation for compatibility.

Usage:
    from llm_dit.utils.lora import load_lora, fuse_lora

    # Simple loading (fuses into model)
    load_lora(pipeline.transformer, "/path/to/lora.safetensors", scale=0.8)

    # Multiple LoRAs
    load_lora(pipeline.transformer, "/path/to/lora1.safetensors", scale=0.5)
    load_lora(pipeline.transformer, "/path/to/lora2.safetensors", scale=0.3)
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors

logger = logging.getLogger(__name__)


class LoRALoader:
    """
    LoRA loader for DiT transformer models.

    Converts various LoRA state dict formats to a standardized format
    and fuses the LoRA weights into the base model.
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cpu",
        torch_dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the LoRA loader.

        Args:
            device: Device to load tensors onto
            torch_dtype: Data type for tensors
        """
        self.device = device
        self.torch_dtype = torch_dtype

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

                weight_up = state_dict[lora_B_key].to(device=self.device, dtype=self.torch_dtype)
                weight_down = state_dict[lora_A_key].to(device=self.device, dtype=self.torch_dtype)

                # Handle conv2d LoRA (4D tensors)
                if len(weight_up.shape) == 4:
                    weight_up = weight_up.squeeze(3).squeeze(2)
                    weight_down = weight_down.squeeze(3).squeeze(2)
                    weight_lora = alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3)
                else:
                    weight_lora = alpha * torch.mm(weight_up, weight_down)

                # Fuse into base model
                state_dict_base = module.state_dict()
                state_dict_base["weight"] = (
                    state_dict_base["weight"].to(device=self.device, dtype=self.torch_dtype)
                    + weight_lora
                )
                module.load_state_dict(state_dict_base)
                updated_num += 1

        logger.info(f"Fused {updated_num} LoRA layers (alpha={alpha})")
        return updated_num


def load_lora(
    model: nn.Module,
    lora_path: Union[str, Path],
    scale: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> int:
    """
    Load and fuse a LoRA into a model.

    Args:
        model: Target model (typically pipeline.transformer)
        lora_path: Path to LoRA weights (.safetensors or .bin)
        scale: LoRA scale factor (alpha)
        device: Device for computation (defaults to model device)
        torch_dtype: Data type (defaults to model dtype)

    Returns:
        Number of layers updated

    Example:
        load_lora(pipeline.transformer, "anime_style.safetensors", scale=0.7)
    """
    lora_path = Path(lora_path)
    if not lora_path.exists():
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    # Infer device/dtype from model if not specified
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"

    if torch_dtype is None:
        try:
            torch_dtype = next(model.parameters()).dtype
        except StopIteration:
            torch_dtype = torch.float32

    logger.info(f"Loading LoRA: {lora_path} (scale={scale})")

    # Load state dict
    if lora_path.suffix == ".safetensors":
        state_dict = load_safetensors(str(lora_path))
    else:
        state_dict = torch.load(str(lora_path), map_location="cpu", weights_only=True)

    # Fuse
    loader = LoRALoader(device=device, torch_dtype=torch_dtype)
    return loader.fuse_lora_to_base_model(model, state_dict, alpha=scale)


def fuse_lora(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    scale: float = 1.0,
    device: Optional[Union[str, torch.device]] = None,
    torch_dtype: Optional[torch.dtype] = None,
) -> int:
    """
    Fuse LoRA weights from a state dict into a model.

    Same as load_lora but takes a state dict directly instead of a file path.

    Args:
        model: Target model
        state_dict: LoRA state dict
        scale: LoRA scale factor
        device: Device for computation
        torch_dtype: Data type

    Returns:
        Number of layers updated
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"

    if torch_dtype is None:
        try:
            torch_dtype = next(model.parameters()).dtype
        except StopIteration:
            torch_dtype = torch.float32

    loader = LoRALoader(device=device, torch_dtype=torch_dtype)
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
