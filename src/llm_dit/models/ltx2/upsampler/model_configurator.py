"""
Configuration loader for LatentUpsampler.

Last Updated: 2026-01-18

Provides a configurator class for creating LatentUpsampler from config dicts.
"""

from typing import Any, Dict

from llm_dit.models.ltx2.upsampler.model import LatentUpsampler


class LatentUpsamplerConfigurator:
    """
    Configurator for LatentUpsampler model.

    Used to create a LatentUpsampler model from a configuration dictionary.
    """

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> LatentUpsampler:
        """
        Create a LatentUpsampler from a configuration dictionary.

        Args:
            config: Dictionary with configuration parameters. All keys are optional
                and will use defaults from LatentUpsampler if not specified.

        Returns:
            Configured LatentUpsampler instance.
        """
        return LatentUpsampler(
            in_channels=config.get("in_channels", 128),
            mid_channels=config.get("mid_channels", 512),
            num_blocks_per_stage=config.get("num_blocks_per_stage", 4),
            dims=config.get("dims", 3),
            spatial_upsample=config.get("spatial_upsample", True),
            temporal_upsample=config.get("temporal_upsample", False),
            spatial_scale=config.get("spatial_scale", 2.0),
            rational_resampler=config.get("rational_resampler", False),
        )
