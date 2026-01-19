"""
llm-dit backend implementation for portable tests.

Last Updated: 2026-01-19

Uses our llm_dit package to generate videos.
This backend is used when tests run in the llm-dit-experiments repo.
"""

import gc
import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch

from .protocol import (
    Backend,
    GenerationConfig,
    GenerationInputs,
    GenerationResult,
    GenerationStats,
)

logger = logging.getLogger(__name__)


def _get_peak_memory() -> float:
    """Get current peak GPU memory in GB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0.0


def _reset_peak_memory() -> None:
    """Reset peak memory tracker."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _cleanup_memory() -> None:
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class LLMDitBackend(Backend):
    """Backend using our llm_dit implementation."""

    def __init__(
        self,
        model_path: str = "models/LTX-2",
        text_encoder_path: Optional[str] = None,
    ):
        """Initialize backend.

        Args:
            model_path: Path to LTX-2 model directory
            text_encoder_path: Optional separate path for text encoder
        """
        self.model_path = Path(model_path)
        self.text_encoder_path = (
            Path(text_encoder_path) if text_encoder_path else None
        )

    @property
    def name(self) -> str:
        return "llm_dit"

    def generate_video(
        self,
        prompt: str,
        config: GenerationConfig,
        output_dir: Optional[Path] = None,
        save_video: bool = True,
        save_latents: bool = False,
    ) -> GenerationResult:
        """Generate video using llm_dit with sequential offloading.

        This uses generate_video_with_offloading() which:
        1. Loads text encoder -> encodes -> unloads
        2. Loads transformer -> denoises -> unloads
        3. Loads VAE -> decodes -> unloads

        Enables full 121-frame generation on 24GB GPUs.
        """
        from llm_dit.pipelines.generate import (
            GenerationConfig as LLMDitConfig,
            generate_video_with_offloading,
        )

        # Validate config
        config.validate()

        # Resolve paths
        text_encoder_path = self.text_encoder_path or (self.model_path / "text_encoder")
        transformer_path = self.model_path / "transformer"
        vae_path = self.model_path / "vae"

        # Log all generation inputs at the start
        inputs = GenerationInputs(
            prompt=prompt,
            negative_prompt="",
            num_frames=config.num_frames,
            height=config.height,
            width=config.width,
            frame_rate=config.frame_rate,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            seed=config.seed,
            transformer_path=str(transformer_path),
            text_encoder_path=str(text_encoder_path),
            vae_path=str(vae_path),
            transformer_dtype="bfloat16",
            transformer_quantization="fp8-quanto" if config.fp8 else "bf16",
            text_encoder_dtype="bfloat16",
            text_encoder_quantization="8bit",
            vae_dtype="bfloat16",
            base_shift=0.95,
            max_shift=2.05,
            terminal_sigma=0.1,
        )
        inputs.log_summary(logger)

        # Track stats
        stats = GenerationStats()
        start_time = time.time()

        # Track timing per stage
        stage_times: dict[str, float] = {}
        stage_memories: dict[str, float] = {}
        current_stage_start = time.time()

        def progress_callback(stage: str, step: int, total: int):
            nonlocal current_stage_start
            if step == 0:
                # Stage starting
                _reset_peak_memory()
                current_stage_start = time.time()
            elif step == total:
                # Stage complete
                stage_times[stage] = time.time() - current_stage_start
                stage_memories[stage] = _get_peak_memory()
                logger.info(
                    f"Stage {stage} complete: {stage_times[stage]:.1f}s, "
                    f"{stage_memories[stage]:.1f}GB peak"
                )

        # Convert to llm_dit config
        llm_dit_config = LLMDitConfig(
            num_frames=config.num_frames,
            height=config.height,
            width=config.width,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            seed=config.seed,
        )

        # Generate video with offloading
        video = generate_video_with_offloading(
            prompt=prompt,
            config=llm_dit_config,
            model_path=self.model_path,
            text_encoder_path=text_encoder_path,
            quantize=config.fp8,
            precision="fp8-quanto" if config.fp8 else "bf16",
            dtype=config.dtype,
            callback=progress_callback,
        )

        # Populate stats
        stats.total_time = time.time() - start_time
        stats.text_encoder_time = stage_times.get("text_encoder", 0.0)
        stats.transformer_time = stage_times.get("transformer", 0.0)
        stats.vae_time = stage_times.get("vae", 0.0)
        stats.text_encoder_peak_memory = stage_memories.get("text_encoder", 0.0)
        stats.transformer_peak_memory = stage_memories.get("transformer", 0.0)
        stats.vae_peak_memory = stage_memories.get("vae", 0.0)
        stats.actual_num_frames = video.shape[0]
        stats.actual_height = video.shape[1]
        stats.actual_width = video.shape[2]

        # Create result
        result = GenerationResult(
            video=video,
            prompt=prompt,
            config=config,
            stats=stats,
            backend_name=self.name,
        )

        # Save outputs if requested
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            if save_video:
                video_path = output_dir / "video.mp4"
                result.save_video(video_path, fps=int(config.frame_rate))
                logger.info(f"Saved video to {video_path}")

            # Always save metadata
            metadata_path = output_dir / "metadata.json"
            result.save_metadata(metadata_path)
            logger.info(f"Saved metadata to {metadata_path}")

            # Save full generation inputs
            inputs_path = output_dir / "inputs.json"
            with open(inputs_path, "w") as f:
                json.dump(inputs.to_dict(), f, indent=2)
            logger.info(f"Saved inputs to {inputs_path}")

        return result

    def encode_text(self, prompt: str) -> torch.Tensor:
        """Encode text prompt using Gemma3 encoder.

        Returns:
            Text embeddings [1, seq_len, 3840] from 49-layer extraction
        """
        from llm_dit.encoders import Gemma3Encoder

        encoder = Gemma3Encoder(
            model_id=str(self.text_encoder_path or self.model_path / "text_encoder"),
            load_in_8bit=True,
            device="cuda",
        )

        encoding_output = encoder.encode(prompt)
        embeddings = encoding_output.embeddings[0].unsqueeze(0)

        # Cleanup
        del encoder
        _cleanup_memory()

        return embeddings

    def cleanup(self) -> None:
        """Clean up GPU memory."""
        _cleanup_memory()

    def is_available(self) -> bool:
        """Check if llm_dit is available."""
        try:
            import llm_dit  # noqa: F401
            return True
        except ImportError:
            return False


# Standard configs are defined in protocol.py (single source of truth)
# Import via: from tests.backends import SMOKE_CONFIG, SHORT_CONFIG, REFERENCE_CONFIG
