"""
Loader for spatial upsampler model from safetensors checkpoint.

The safetensors file stores model config as JSON in the file's metadata
under the "config" key. The loader reads this config, creates the model
via LatentUpsamplerConfigurator, then loads the state dict.
"""

import json
import logging
from pathlib import Path

import safetensors.torch
import torch

from llm_dit.models.ltx2.upsampler.model import LatentUpsampler
from llm_dit.models.ltx2.upsampler.model_configurator import LatentUpsamplerConfigurator

logger = logging.getLogger(__name__)

# Default config matching the released ltx-2-spatial-upscaler-x2-1.0.safetensors
DEFAULT_UPSAMPLER_CONFIG = {
    "in_channels": 128,
    "mid_channels": 512,
    "num_blocks_per_stage": 4,
    "dims": 3,
    "spatial_upsample": True,
    "temporal_upsample": False,
    "spatial_scale": 2.0,
    "rational_resampler": False,
}


def load_spatial_upsampler(
    path: str | Path,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
) -> LatentUpsampler:
    """Load spatial upsampler from safetensors checkpoint.

    Reads model config from safetensors metadata, falling back to defaults
    if metadata is missing (e.g., for older checkpoint formats).

    Args:
        path: Path to .safetensors checkpoint file.
        dtype: Model dtype (default bfloat16).
        device: Initial device for loading (default cpu for sequential offloading).

    Returns:
        Loaded LatentUpsampler model.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Spatial upsampler not found: {path}")

    # Read config from safetensors metadata
    config = DEFAULT_UPSAMPLER_CONFIG.copy()
    try:
        with safetensors.torch.safe_open(str(path), framework="pt") as f:
            metadata = f.metadata()
            if metadata and "config" in metadata:
                file_config = json.loads(metadata["config"])
                config.update(file_config)
                logger.debug(f"Loaded upsampler config from metadata: {config}")
            else:
                logger.debug("No config in safetensors metadata, using defaults")
    except Exception as e:
        logger.warning(f"Failed to read upsampler metadata: {e}, using defaults")

    # Create model from config
    model = LatentUpsamplerConfigurator.from_config(config)

    # Load state dict
    state_dict = safetensors.torch.load_file(str(path), device=device)
    model.load_state_dict(state_dict)
    model = model.to(dtype=dtype)
    model.train(False)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Loaded spatial upsampler: {param_count / 1e6:.1f}M params, "
        f"scale={config.get('spatial_scale', 2.0)}x, dtype={dtype}"
    )

    return model
