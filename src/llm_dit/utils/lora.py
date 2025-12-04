"""
LoRA loading utilities for Z-Image DiT.

Supports loading LoRA weights with reversible patching (ComfyUI-style).
Weights are backed up before patching and can be restored without model reload.

Usage:
    from llm_dit.utils.lora import LoRAManager, LoRAEntry

    # Create manager for a model
    manager = LoRAManager(pipeline.transformer, loras_dir="/path/to/loras")

    # Add LoRAs (reversible)
    manager.add_lora("anime_style.safetensors", scale=0.8, trigger_words="anime style")
    manager.add_lora("detail.safetensors", scale=0.5)
    manager.apply()

    # Clear all LoRAs (restore original weights)
    manager.clear_all()

    # Legacy API (permanent fusion - still supported)
    load_lora(pipeline.transformer, "/path/to/lora.safetensors", scale=0.8)
"""

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file as load_safetensors

logger = logging.getLogger(__name__)


@dataclass
class LoRAEntry:
    """Configuration for a single LoRA."""

    path: str  # Path to LoRA file
    name: str = ""  # Display name (defaults to filename without extension)
    scale: float = 1.0  # Scale factor (0.0-2.0 typical)
    trigger_words: str = ""  # Trigger words to prepend to prompt
    enabled: bool = True  # Whether this LoRA is active

    def __post_init__(self):
        """Set default name from path if not provided."""
        if not self.name:
            self.name = Path(self.path).stem

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "path": self.path,
            "name": self.name,
            "scale": self.scale,
            "trigger_words": self.trigger_words,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LoRAEntry":
        """Create from dictionary."""
        return cls(
            path=data["path"],
            name=data.get("name", ""),
            scale=data.get("scale", 1.0),
            trigger_words=data.get("trigger_words", ""),
            enabled=data.get("enabled", True),
        )


class LoRAManager:
    """
    Manages LoRA loading with reversible patching.

    Uses ComfyUI-style backup+patch pattern:
    1. Before applying LoRAs, backup original weights to CPU RAM
    2. Apply LoRA patches to model weights
    3. To clear, restore from backup (no model reload needed)

    Thread-safe for concurrent apply/clear operations.
    """

    def __init__(
        self,
        model: nn.Module,
        loras_dir: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize LoRA manager.

        Args:
            model: Target model (typically transformer)
            loras_dir: Directory to scan for LoRA files
            device: Device for computation (defaults to model device)
            torch_dtype: Data type (defaults to model dtype)
        """
        self.model = model
        self.loras_dir = Path(loras_dir) if loras_dir else None

        # Infer device/dtype from model
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

        self.device = device
        self.torch_dtype = torch_dtype

        # State
        self.backup: Dict[str, torch.Tensor] = {}  # Original weights (on CPU)
        self.active_loras: List[LoRAEntry] = []  # Currently applied LoRAs
        self.lora_cache: Dict[str, Dict[str, torch.Tensor]] = {}  # path -> state_dict

        # Thread safety
        self._lock = threading.Lock()

        # Internal loader
        self._loader = LoRALoader(device=device, torch_dtype=torch_dtype)

    def scan_directory(self) -> List[str]:
        """
        Scan loras_dir for LoRA files.

        Returns:
            List of absolute paths to LoRA files (.safetensors, .bin)
        """
        if self.loras_dir is None or not self.loras_dir.exists():
            return []

        lora_files = []
        for ext in ["*.safetensors", "*.bin"]:
            lora_files.extend(self.loras_dir.glob(ext))

        return sorted([str(p.absolute()) for p in lora_files])

    def get_available_loras(self) -> List[LoRAEntry]:
        """
        Get list of available LoRAs from directory.

        Returns:
            List of LoRAEntry objects (not yet applied)
        """
        paths = self.scan_directory()
        return [LoRAEntry(path=p, enabled=False) for p in paths]

    def _load_lora_state_dict(self, path: str) -> Dict[str, torch.Tensor]:
        """Load LoRA state dict, using cache if available."""
        if path in self.lora_cache:
            return self.lora_cache[path]

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"LoRA file not found: {path}")

        if path_obj.suffix == ".safetensors":
            state_dict = load_safetensors(str(path_obj))
        else:
            state_dict = torch.load(str(path_obj), map_location="cpu", weights_only=True)

        self.lora_cache[path] = state_dict
        return state_dict

    def _backup_weights(self) -> None:
        """Backup current model weights to CPU."""
        if self.backup:
            return  # Already backed up

        logger.info("Backing up model weights to CPU...")
        for name, module in self.model.named_modules():
            if hasattr(module, "weight") and module.weight is not None:
                # Clone to CPU to save VRAM
                self.backup[name] = module.weight.data.clone().to("cpu")

        logger.info(f"Backed up {len(self.backup)} layers")

    def _restore_weights(self) -> None:
        """Restore model weights from backup."""
        if not self.backup:
            logger.warning("No backup to restore from")
            return

        logger.info("Restoring model weights from backup...")
        restored = 0
        for name, module in self.model.named_modules():
            if name in self.backup:
                weight = self.backup[name].to(device=self.device, dtype=self.torch_dtype)
                module.weight.data.copy_(weight)
                restored += 1

        logger.info(f"Restored {restored} layers")

    def add_lora(
        self,
        path: str,
        scale: float = 1.0,
        trigger_words: str = "",
        enabled: bool = True,
    ) -> None:
        """
        Add a LoRA to the active list.

        Does not apply immediately - call apply() to patch weights.

        Args:
            path: Path to LoRA file
            scale: Scale factor
            trigger_words: Trigger words to prepend to prompt
            enabled: Whether this LoRA is active
        """
        # Resolve relative paths against loras_dir
        path_obj = Path(path)
        if not path_obj.is_absolute() and self.loras_dir:
            path_obj = self.loras_dir / path_obj

        entry = LoRAEntry(
            path=str(path_obj),
            scale=scale,
            trigger_words=trigger_words,
            enabled=enabled,
        )

        # Remove existing entry with same path
        self.active_loras = [l for l in self.active_loras if l.path != entry.path]
        self.active_loras.append(entry)

    def remove_lora(self, path: str) -> bool:
        """
        Remove a LoRA from the active list.

        Does not apply immediately - call apply() to update weights.

        Args:
            path: Path to LoRA file

        Returns:
            True if removed, False if not found
        """
        original_len = len(self.active_loras)
        self.active_loras = [l for l in self.active_loras if l.path != path]
        return len(self.active_loras) < original_len

    def set_scale(self, path: str, scale: float) -> bool:
        """
        Update scale for a LoRA.

        Does not apply immediately - call apply() to update weights.

        Args:
            path: Path to LoRA file
            scale: New scale factor

        Returns:
            True if updated, False if not found
        """
        for lora in self.active_loras:
            if lora.path == path:
                lora.scale = scale
                return True
        return False

    def set_enabled(self, path: str, enabled: bool) -> bool:
        """
        Enable/disable a LoRA.

        Does not apply immediately - call apply() to update weights.

        Args:
            path: Path to LoRA file
            enabled: New enabled state

        Returns:
            True if updated, False if not found
        """
        for lora in self.active_loras:
            if lora.path == path:
                lora.enabled = enabled
                return True
        return False

    def reorder(self, paths: List[str]) -> None:
        """
        Reorder active LoRAs.

        Does not apply immediately - call apply() to update weights.

        Args:
            paths: List of paths in new order
        """
        # Build lookup
        lora_by_path = {l.path: l for l in self.active_loras}

        # Reorder
        new_order = []
        for path in paths:
            if path in lora_by_path:
                new_order.append(lora_by_path[path])

        # Add any missing (shouldn't happen but defensive)
        for lora in self.active_loras:
            if lora not in new_order:
                new_order.append(lora)

        self.active_loras = new_order

    def apply(self) -> int:
        """
        Apply all active LoRAs to model weights.

        Thread-safe. Backs up weights before first application.

        Returns:
            Number of layers updated
        """
        with self._lock:
            # Backup original weights if not done
            self._backup_weights()

            # Restore to clean state
            self._restore_weights()

            # Apply enabled LoRAs in order
            total_updated = 0
            for lora in self.active_loras:
                if not lora.enabled:
                    continue

                try:
                    state_dict = self._load_lora_state_dict(lora.path)
                    updated = self._loader.fuse_lora_to_base_model(
                        self.model, state_dict, alpha=lora.scale
                    )
                    total_updated += updated
                    logger.info(f"Applied {lora.name} (scale={lora.scale}): {updated} layers")
                except Exception as e:
                    logger.error(f"Failed to apply {lora.name}: {e}")

            return total_updated

    def clear_all(self) -> None:
        """
        Clear all LoRAs and restore original weights.

        Thread-safe.
        """
        with self._lock:
            self._restore_weights()
            self.active_loras.clear()
            logger.info("Cleared all LoRAs")

    def get_trigger_words(self) -> str:
        """
        Get combined trigger words from all enabled LoRAs.

        Returns:
            Space-separated trigger words
        """
        words = []
        for lora in self.active_loras:
            if lora.enabled and lora.trigger_words:
                words.append(lora.trigger_words.strip())
        return " ".join(words)

    def get_active_entries(self) -> List[LoRAEntry]:
        """Get list of active LoRA entries."""
        return list(self.active_loras)

    def set_loras(self, entries: List[LoRAEntry]) -> int:
        """
        Replace all LoRAs with new list and apply.

        Args:
            entries: List of LoRAEntry objects

        Returns:
            Number of layers updated
        """
        self.active_loras = list(entries)
        return self.apply()


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

    NOTE: For permanently fused LoRAs (using load_lora/fuse_lora), this is not possible.
    Use LoRAManager for reversible LoRA application.

    This function is provided for API compatibility but will raise an error
    unless the model has an attached LoRAManager.
    """
    # Check if model has a LoRAManager attached
    if hasattr(model, "_lora_manager") and model._lora_manager is not None:
        model._lora_manager.clear_all()
        return

    raise NotImplementedError(
        "Fused LoRAs cannot be cleared. Reload the model to remove LoRA weights. "
        "For reversible LoRAs, use LoRAManager instead of load_lora()."
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
