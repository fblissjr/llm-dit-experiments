"""
Official LTX-2 backend implementation for portable tests.

Last Updated: 2026-01-19

Uses the official LTX-2 pipeline from ltx_pipelines package.
This backend is used when tests run in the LTX-2 repo or when
ltx_pipelines is available via coderef.
"""

import gc
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import torch

from .protocol import (
    Backend,
    GenerationConfig,
    GenerationResult,
    GenerationStats,
)

logger = logging.getLogger(__name__)


def _ensure_ltx2_imports():
    """Ensure ltx_pipelines is importable (add coderef to path if needed)."""
    try:
        import ltx_pipelines  # noqa: F401

        return True
    except ImportError:
        pass

    # Try adding coderef to path
    current = Path(__file__).parent
    for _ in range(10):
        if (current / "CLAUDE.md").exists() or (current / ".git").exists():
            coderef_core = current / "coderef/LTX-2/packages/ltx-core/src"
            coderef_pipelines = current / "coderef/LTX-2/packages/ltx-pipelines/src"
            if coderef_core.exists() and coderef_pipelines.exists():
                if str(coderef_core) not in sys.path:
                    sys.path.insert(0, str(coderef_core))
                if str(coderef_pipelines) not in sys.path:
                    sys.path.insert(0, str(coderef_pipelines))
                return True
            break
        current = current.parent
    return False


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


class LTX2Backend(Backend):
    """Backend using official LTX-2 implementation."""

    def __init__(
        self,
        checkpoint_path: str = "models/LTX-2/transformer/model.safetensors",
        gemma_root: str = "models/LTX-2/text_encoder/",
    ):
        """Initialize backend.

        Args:
            checkpoint_path: Path to transformer checkpoint (safetensors)
            gemma_root: Path to Gemma text encoder directory
        """
        _ensure_ltx2_imports()
        self.checkpoint_path = checkpoint_path
        self.gemma_root = gemma_root
        self._pipeline = None

    @property
    def name(self) -> str:
        return "ltx2"

    def _get_pipeline(self, fp8: bool = True):
        """Get or create the pipeline (lazy initialization)."""
        if self._pipeline is None:
            from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline

            self._pipeline = TI2VidOneStagePipeline(
                checkpoint_path=self.checkpoint_path,
                gemma_root=self.gemma_root,
                loras=[],
                device=torch.device("cuda"),
                fp8transformer=fp8,
            )
        return self._pipeline

    def generate_video(
        self,
        prompt: str,
        config: GenerationConfig,
        output_dir: Optional[Path] = None,
        save_video: bool = True,
        save_latents: bool = False,
    ) -> GenerationResult:
        """Generate video using official LTX-2 pipeline.

        Uses TI2VidOneStagePipeline from ltx_pipelines.
        """
        from ltx_pipelines.utils.media_io import encode_video

        # Validate config
        config.validate()

        # Track stats
        stats = GenerationStats()
        start_time = time.time()
        _reset_peak_memory()

        # Get pipeline
        pipeline = self._get_pipeline(fp8=config.fp8)

        # Generate video
        logger.info(f"Generating with LTX-2 official pipeline: {prompt}")
        logger.info(
            f"Config: {config.num_frames} frames, {config.height}x{config.width}, "
            f"{config.num_inference_steps} steps, CFG {config.guidance_scale}"
        )

        video_iterator, audio = pipeline(
            prompt=prompt,
            negative_prompt="",
            seed=config.seed,
            height=config.height,
            width=config.width,
            num_frames=config.num_frames,
            frame_rate=config.frame_rate,
            num_inference_steps=config.num_inference_steps,
            cfg_guidance_scale=config.guidance_scale,
            images=[],  # No image conditioning for T2V
            enhance_prompt=False,
        )

        # Collect video frames (generator yields intermediate results)
        video_frames = list(video_iterator)
        if not video_frames:
            raise RuntimeError("No video frames generated")

        # Final video is the last yielded result
        video = video_frames[-1]  # [F, H, W, C] or similar
        logger.info(f"Video shape: {video.shape}")

        # Convert to [F, H, W, C] uint8 if needed
        if video.dtype != torch.uint8:
            if video.dim() == 5:  # [B, C, F, H, W]
                video = video[0].permute(1, 2, 3, 0)  # [F, H, W, C]
            video = ((video.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)

        # Populate stats
        stats.total_time = time.time() - start_time
        stats.transformer_peak_memory = _get_peak_memory()
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
                # Use official encode_video
                encode_video(video, str(video_path), frame_rate=config.frame_rate)
                logger.info(f"Saved video to {video_path}")

            # Always save metadata
            metadata_path = output_dir / "metadata.json"
            result.save_metadata(metadata_path)
            logger.info(f"Saved metadata to {metadata_path}")

        return result

    def encode_text(self, prompt: str) -> torch.Tensor:
        """Encode text prompt using official Gemma encoder.

        Returns:
            Text embeddings from official implementation
        """
        from ltx_core.text_encoders.gemma import encode_text

        pipeline = self._get_pipeline()
        text_encoder = pipeline.model_ledger.text_encoder()

        # encode_text returns (video_context, audio_context) tuples for positive/negative
        context_p, _ = encode_text(text_encoder, prompts=[prompt, ""])
        v_context_p, _ = context_p  # Video context only

        # Cleanup
        del text_encoder
        _cleanup_memory()

        return v_context_p

    def cleanup(self) -> None:
        """Clean up pipeline and GPU memory."""
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        _cleanup_memory()

    def is_available(self) -> bool:
        """Check if LTX-2 packages are available."""
        return _ensure_ltx2_imports()


# Convenience function to run comparison between backends
def compare_backends(
    prompt: str,
    config: GenerationConfig,
    output_base: Path,
) -> dict:
    """Run generation with both backends and compare results.

    Args:
        prompt: Text prompt for generation
        config: Generation configuration
        output_base: Base directory for outputs

    Returns:
        Comparison dict with stats from both backends
    """
    from . import get_backend, is_llm_dit_available, is_ltx2_available

    results = {}

    # Run llm_dit backend
    if is_llm_dit_available():
        from .llm_dit_backend import LLMDitBackend

        backend = LLMDitBackend()
        output_dir = output_base / "llm_dit"
        result = backend.generate_video(prompt, config, output_dir)
        results["llm_dit"] = {
            "stats": result.stats.to_dict() if result.stats else {},
            "video_shape": list(result.video.shape),
        }
        backend.cleanup()

    # Run ltx2 backend
    if is_ltx2_available():
        backend = LTX2Backend()
        output_dir = output_base / "ltx2"
        result = backend.generate_video(prompt, config, output_dir)
        results["ltx2"] = {
            "stats": result.stats.to_dict() if result.stats else {},
            "video_shape": list(result.video.shape),
        }
        backend.cleanup()

    return results
