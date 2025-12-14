#!/usr/bin/env python3
"""
Caption Length Study for Z-Image.

Tests whether longer, more detailed prompts improve generation quality,
and whether padding embeddings to a fixed 1504 length affects results.

Workflow:
1. Generate source images from simple prompts
2. Caption each with Qwen3-VL to get detailed descriptions
3. Test various truncation lengths and padding strategies
4. Generate comparison grids and compute SSIM metrics

Usage:
    uv run experiments/caption_length_study.py \\
        --config config.toml \\
        --prompts "A cat" "A mountain landscape" \\
        --seeds 42 \\
        --output-dir experiments/results/caption_length_study
"""

import argparse
import gc
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_dit.cli import RuntimeConfig, load_runtime_config
from llm_dit.utils.long_prompt import compress_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
MAX_TEXT_SEQ_LEN = 1504
DEFAULT_TARGET_LENGTHS = [50, 150, 300, 600, 1000, 1504]
DEFAULT_FILL_MODES = [
    "content_only",     # Just caption truncated to target (variable length, capped)
    "pad_end_zero",     # Content + zero padding to exact target length
    "pad_end_mean",     # Content + mean-embedding padding to exact target length
    "pad_middle_zero",  # Content split with zeros in middle
    "filler_repeat",    # Repeat content tokens to reach target length
]
DEFAULT_COMPRESSION_MODES = ["truncate", "interpolate", "pool", "attention_pool"]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================


def truncate_to_tokens(text: str, tokenizer, target_tokens: int) -> tuple[str, int]:
    """
    Truncate text to approximately N tokens.

    Args:
        text: Input text
        tokenizer: HuggingFace tokenizer
        target_tokens: Target token count

    Returns:
        Tuple of (truncated_text, actual_token_count)
    """
    # Encode without special tokens for accurate counting
    input_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(input_ids) <= target_tokens:
        return text, len(input_ids)

    truncated_ids = input_ids[:target_tokens]
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)

    # Re-encode to get actual count after decode
    actual_count = len(tokenizer.encode(truncated_text, add_special_tokens=False))

    return truncated_text, actual_count


def apply_fill_mode(
    embeddings: torch.Tensor,
    target_length: int,
    mode: str = "content_only",
) -> torch.Tensor:
    """
    Fill embeddings to exactly target_length using specified strategy.

    This is the core function for testing how Z-Image DiT responds to different
    sequence lengths and fill patterns.

    Args:
        embeddings: Input embeddings [seq_len, hidden_dim]
        target_length: Exact target sequence length
        mode: Fill strategy:
            - "content_only": Truncate to target, return variable length (capped)
            - "pad_end_zero": Content + zero padding at end to reach exact target
            - "pad_end_mean": Content + mean-embedding padding at end
            - "pad_middle_zero": Split content, zeros in middle
            - "filler_repeat": Repeat content tokens cyclically to reach target

    Returns:
        Filled embeddings of shape [target_length, hidden_dim]
        (except content_only which returns [min(seq_len, target), hidden_dim])
    """
    seq_len, hidden_dim = embeddings.shape

    if mode == "content_only":
        # Just truncate/cap at target length, don't pad
        return embeddings[:target_length]

    elif mode == "pad_end_zero":
        # Truncate if needed, then pad with zeros
        content = embeddings[:target_length]
        if content.shape[0] >= target_length:
            return content
        pad_length = target_length - content.shape[0]
        padding = torch.zeros(
            pad_length, hidden_dim, dtype=embeddings.dtype, device=embeddings.device
        )
        return torch.cat([content, padding], dim=0)

    elif mode == "pad_end_mean":
        # Truncate if needed, then pad with mean embedding
        content = embeddings[:target_length]
        if content.shape[0] >= target_length:
            return content
        pad_length = target_length - content.shape[0]
        mean_emb = content.mean(dim=0, keepdim=True)
        padding = mean_emb.expand(pad_length, -1)
        return torch.cat([content, padding], dim=0)

    elif mode == "pad_middle_zero":
        # Split content in half, put zeros in middle
        content = embeddings[:target_length]
        if content.shape[0] >= target_length:
            return content
        pad_length = target_length - content.shape[0]
        # Split content: first half, padding, second half
        half = content.shape[0] // 2
        first_half = content[:half]
        second_half = content[half:]
        padding = torch.zeros(
            pad_length, hidden_dim, dtype=embeddings.dtype, device=embeddings.device
        )
        return torch.cat([first_half, padding, second_half], dim=0)

    elif mode == "filler_repeat":
        # Repeat content tokens cyclically to fill target length
        content = embeddings[:target_length]
        if content.shape[0] >= target_length:
            return content
        # Calculate how many times to repeat
        repeats_needed = (target_length + content.shape[0] - 1) // content.shape[0]
        repeated = content.repeat(repeats_needed, 1)
        return repeated[:target_length]

    else:
        raise ValueError(f"Unknown fill mode: {mode}")


def caption_image(image_path: Path, vl_extractor) -> str:
    """
    Generate detailed caption using Qwen3-VL.

    Args:
        image_path: Path to image file
        vl_extractor: VLEmbeddingExtractor instance

    Returns:
        Detailed caption string
    """
    image = Image.open(image_path).convert("RGB")

    # Put instructions in the prompt (not system_prompt) to avoid Qwen3-VL processor issues
    prompt = """Describe this image in extensive detail for use as a text-to-image prompt. Include:
- Main subject(s) and their appearance
- Colors, textures, and materials
- Lighting and atmosphere
- Composition and perspective
- Style and artistic qualities
- Background elements and setting

Be specific and descriptive. Do not use phrases like "the image shows" - just describe directly."""

    return vl_extractor.generate(
        image=image,
        prompt=prompt,
        max_new_tokens=1024,
        temperature=0.6,
    )


def compute_ssim(img1: Image.Image, img2: Image.Image) -> float:
    """
    Compute SSIM between two images.

    Uses skimage if available, falls back to simple MSE-based approximation.
    """
    import numpy as np

    # Convert to numpy
    arr1 = np.array(img1.convert("RGB")).astype(np.float32) / 255.0
    arr2 = np.array(img2.convert("RGB")).astype(np.float32) / 255.0

    # Resize if different sizes
    if arr1.shape != arr2.shape:
        from PIL import Image as PILImage

        img2_resized = img2.resize(img1.size, PILImage.Resampling.LANCZOS)
        arr2 = np.array(img2_resized.convert("RGB")).astype(np.float32) / 255.0

    try:
        from skimage.metrics import structural_similarity as ssim

        # Compute SSIM for each channel and average
        ssim_values = []
        for c in range(3):
            ssim_val = ssim(arr1[:, :, c], arr2[:, :, c], data_range=1.0)
            ssim_values.append(ssim_val)
        return float(np.mean(ssim_values))
    except ImportError:
        # Fallback: simple MSE-based similarity (not true SSIM)
        mse = np.mean((arr1 - arr2) ** 2)
        # Convert to similarity score (1.0 = identical)
        return float(1.0 / (1.0 + mse * 100))


def create_comparison_grid(
    images: list[tuple[str, Image.Image]],
    output_path: Path,
    cols: int = 4,
    cell_size: int = 512,
    label_height: int = 30,
) -> None:
    """
    Create a labeled comparison grid of images.

    Args:
        images: List of (label, image) tuples
        output_path: Path to save grid
        cols: Number of columns
        cell_size: Size of each cell (images scaled to fit)
        label_height: Height for text labels
    """
    from PIL import ImageDraw, ImageFont

    n_images = len(images)
    rows = (n_images + cols - 1) // cols

    grid_width = cols * cell_size
    grid_height = rows * (cell_size + label_height)

    grid = Image.new("RGB", (grid_width, grid_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Try to get a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    for idx, (label, img) in enumerate(images):
        row = idx // cols
        col = idx % cols

        x = col * cell_size
        y = row * (cell_size + label_height)

        # Resize image to fit cell
        img_resized = img.copy()
        img_resized.thumbnail((cell_size, cell_size), Image.Resampling.LANCZOS)

        # Center in cell
        paste_x = x + (cell_size - img_resized.width) // 2
        paste_y = y + label_height + (cell_size - img_resized.height) // 2

        grid.paste(img_resized, (paste_x, paste_y))

        # Draw label
        draw.text((x + 5, y + 5), label, fill=(0, 0, 0), font=font)

    grid.save(output_path)
    logger.info(f"Saved comparison grid: {output_path}")


# ============================================================
# DATA CLASSES
# ============================================================


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment variant."""

    source_prompt: str
    source_image_path: str
    caption: str
    caption_token_count: int
    target_length: int  # Final embedding length to test
    fill_mode: str  # How to reach target length (content_only, pad_end_zero, etc.)
    compression_mode: str | None  # Only for when raw embeddings >1504 tokens
    seed: int
    output_path: str
    # Hidden layer selection
    hidden_layer: int = -2  # For text encoding (Qwen3-4B)
    vl_hidden_layer: int | None = None  # For VL embedding extraction (Qwen3-VL)
    token_mode: str | None = None  # For VL: full, text_only, image_only, image_no_markers
    use_vl_embeddings: bool = False  # Whether to use VL embeddings instead of text
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ExperimentResult:
    """Result of a single experiment."""

    config: ExperimentConfig
    actual_token_count: int
    final_embedding_length: int
    ssim_vs_source: float | None
    generation_time_seconds: float
    error: str | None = None


# ============================================================
# EXPERIMENT RUNNER
# ============================================================


# Token mode type for VL extraction
TokenMode = Literal["full", "text_only", "image_only", "image_no_markers"]


class CaptionLengthStudy:
    """Runs caption length and fill mode experiments."""

    def __init__(
        self,
        model_path: str,
        vl_model_path: str | None,
        output_dir: Path,
        text_encoder_device: str = "cpu",
        dit_device: str = "cuda",
        vae_device: str = "cuda",
        target_lengths: list[int] | None = None,
        fill_modes: list[str] | None = None,
        compression_modes: list[str] | None = None,
        hidden_layers: list[int] | None = None,
        vl_hidden_layers: list[int] | None = None,
        token_modes: list[TokenMode] | None = None,
        use_vl_embeddings: bool = False,
        dry_run: bool = False,
    ):
        self.model_path = model_path
        self.vl_model_path = vl_model_path
        self.output_dir = Path(output_dir)
        self.text_encoder_device = text_encoder_device
        self.dit_device = dit_device
        self.vae_device = vae_device
        self.target_lengths = target_lengths or DEFAULT_TARGET_LENGTHS
        self.fill_modes = fill_modes or DEFAULT_FILL_MODES
        self.compression_modes = compression_modes or DEFAULT_COMPRESSION_MODES
        self.hidden_layers = hidden_layers or [-2]  # Default: penultimate layer
        self.vl_hidden_layers = vl_hidden_layers  # None = don't use VL embeddings
        self.token_modes = token_modes or ["full"]  # Default: all tokens
        self.use_vl_embeddings = use_vl_embeddings
        self.dry_run = dry_run

        self.pipeline = None
        self.vl_extractor = None
        self.tokenizer = None

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "source").mkdir(exist_ok=True)
        (self.output_dir / "regenerated").mkdir(exist_ok=True)
        (self.output_dir / "grids").mkdir(exist_ok=True)

    def load_vl_extractor(self):
        """Load Qwen3-VL for captioning on CUDA."""
        if self.dry_run:
            logger.info("[DRY RUN] Would load VL extractor")
            return

        if self.vl_extractor is not None:
            return

        from llm_dit.vl.qwen3_vl import VLEmbeddingExtractor

        vl_path = self.vl_model_path
        if not vl_path:
            vl_path = VLEmbeddingExtractor.find_model_path()
            if not vl_path:
                raise ValueError(
                    "VL model path not provided and auto-detection failed. "
                    "Use --vl-model-path to specify Qwen3-VL model location."
                )

        # Load VL on CUDA for fast captioning (will unload before pipeline loads)
        vl_device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading VL extractor from {vl_path} on {vl_device}")
        self.vl_extractor = VLEmbeddingExtractor.from_pretrained(
            vl_path,
            device=vl_device,
            torch_dtype=torch.bfloat16,
        )
        logger.info(f"VL extractor loaded on {vl_device}")

    def unload_vl_extractor(self):
        """Unload VL extractor to free memory."""
        if self.vl_extractor is not None:
            logger.info("Unloading VL extractor...")
            self.vl_extractor.unload()
            self.vl_extractor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def load_pipeline(self):
        """Load Z-Image pipeline."""
        if self.dry_run:
            logger.info("[DRY RUN] Would load pipeline")
            return

        if self.pipeline is not None:
            return

        from llm_dit import ZImagePipeline

        logger.info(f"Loading pipeline from {self.model_path}")
        self.pipeline = ZImagePipeline.from_pretrained(
            self.model_path,
            encoder_device=self.text_encoder_device,
            dit_device=self.dit_device,
            vae_device=self.vae_device,
            torch_dtype=torch.bfloat16,
        )

        # Get tokenizer for truncation
        self.tokenizer = self.pipeline.encoder.backend.tokenizer
        logger.info("Pipeline loaded")

    def unload_pipeline(self):
        """Unload pipeline to free memory."""
        if self.pipeline is not None:
            logger.info("Unloading pipeline...")
            del self.pipeline
            self.pipeline = None
            self.tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_source_image(
        self, prompt: str, seed: int, width: int = 1024, height: int = 1024
    ) -> Path:
        """Generate a source image from a simple prompt."""
        self.load_pipeline()

        safe_prompt = prompt.replace(" ", "_")[:30]
        output_path = self.output_dir / "source" / f"{safe_prompt}_seed{seed}.png"

        if self.dry_run:
            logger.info(f"[DRY RUN] Would generate source: {output_path}")
            return output_path

        logger.info(f"Generating source image: {prompt}")
        result = self.pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=9,
            generator=torch.Generator().manual_seed(seed),
        )

        # Handle different result types
        if hasattr(result, "images"):
            image = result.images[0]
        elif hasattr(result, "save"):
            image = result
        else:
            image = result[0]

        image.save(output_path)
        logger.info(f"Saved source: {output_path}")

        return output_path

    def caption_source_image(self, image_path: Path) -> tuple[str, int]:
        """Generate caption for source image."""
        self.load_vl_extractor()

        if self.dry_run:
            # Return dummy caption for dry run
            return "A detailed caption would go here...", 100

        logger.info(f"[Caption] Starting captioning for: {image_path}")
        caption = caption_image(image_path, self.vl_extractor)

        # Strip thinking content if present
        if "<think>" in caption and "</think>" in caption:
            think_start = caption.find("<think>")
            think_end = caption.find("</think>")
            thinking = caption[think_start + len("<think>"):think_end].strip()
            logger.debug(f"[Caption] Thinking content ({len(thinking)} chars): {thinking[:200]}...")
            caption = caption[think_end + len("</think>"):].strip()

        # Count tokens
        token_count = len(self.vl_extractor.processor.tokenizer.encode(caption, add_special_tokens=False))

        # Save caption
        caption_path = image_path.with_suffix(".txt")
        caption_path.write_text(caption)

        token_info_path = image_path.with_name(image_path.stem + "_tokens.txt")
        token_info_path.write_text(f"Token count: {token_count}\n\nCaption:\n{caption}")

        logger.info(f"[Caption] Generated {token_count} tokens")
        logger.debug(f"[Caption] Full caption:\n{caption}")
        return caption, token_count

    def run_variant(
        self,
        source_prompt: str,
        source_image_path: Path,
        caption: str,
        caption_token_count: int,
        target_length: int,
        fill_mode: str,
        compression_mode: str | None,
        seed: int,
        hidden_layer: int = -2,
        vl_hidden_layer: int | None = None,
        token_mode: str | None = None,
        use_vl_embeddings: bool = False,
    ) -> ExperimentResult:
        """Run a single experiment variant.

        Args:
            source_prompt: Original simple prompt used to generate source image
            source_image_path: Path to source image (used for VL extraction if enabled)
            caption: Detailed caption from Qwen3-VL
            caption_token_count: Token count of caption
            target_length: Target embedding sequence length
            fill_mode: How to reach target length
            compression_mode: Compression for >1504 tokens
            seed: Random seed for generation
            hidden_layer: Hidden layer for text encoding (default: -2, penultimate)
            vl_hidden_layer: Hidden layer for VL extraction (if using VL embeddings)
            token_mode: Token mode for VL: full, text_only, image_only, image_no_markers
            use_vl_embeddings: Whether to use VL embeddings instead of text encoding
        """
        import time

        self.load_pipeline()

        # Build output filename with layer info
        layer_str = f"_L{hidden_layer}"
        if use_vl_embeddings:
            layer_str = f"_vlL{vl_hidden_layer}_{token_mode}"
        compression_str = f"_{compression_mode}" if compression_mode else ""
        output_name = f"{source_image_path.stem}_len{target_length}_{fill_mode}{layer_str}{compression_str}.png"
        output_path = self.output_dir / "regenerated" / output_name

        config = ExperimentConfig(
            source_prompt=source_prompt,
            source_image_path=str(source_image_path),
            caption=caption[:200] + "..." if len(caption) > 200 else caption,
            caption_token_count=caption_token_count,
            target_length=target_length,
            fill_mode=fill_mode,
            compression_mode=compression_mode,
            seed=seed,
            output_path=str(output_path),
            hidden_layer=hidden_layer,
            vl_hidden_layer=vl_hidden_layer,
            token_mode=token_mode,
            use_vl_embeddings=use_vl_embeddings,
        )

        if self.dry_run:
            logger.info(f"[DRY RUN] Would generate: {output_path}")
            return ExperimentResult(
                config=config,
                actual_token_count=0,
                final_embedding_length=target_length,
                ssim_vs_source=None,
                generation_time_seconds=0.0,
            )

        start_time = time.time()

        try:
            logger.info(f"[Variant] target_length={target_length}, fill_mode={fill_mode}, compression={compression_mode}")
            logger.info(f"[Variant] hidden_layer={hidden_layer}, use_vl={use_vl_embeddings}, vl_layer={vl_hidden_layer}, token_mode={token_mode}")
            logger.debug(f"[Variant] Caption ({caption_token_count} tokens): {caption[:100]}...")

            if use_vl_embeddings:
                # Use VL embedding extraction from source image
                # This keeps VL extractor loaded during generation phase
                self.load_vl_extractor()
                source_image = Image.open(source_image_path)

                # Map token_mode to VLEmbeddingExtractor parameters
                text_tokens_only = token_mode == "text_only"
                image_tokens_only = token_mode == "image_only"
                image_tokens_no_markers = token_mode == "image_no_markers"

                logger.info(f"[Variant] Extracting VL embeddings from {source_image_path}")
                vl_result = self.vl_extractor.extract(
                    image=source_image,
                    text=caption,  # Use caption as text context
                    hidden_layer=vl_hidden_layer,
                    text_tokens_only=text_tokens_only,
                    image_tokens_only=image_tokens_only,
                    image_tokens_no_markers=image_tokens_no_markers,
                    scale_to_text=True,  # Scale VL embeddings to match text std
                )
                embeddings = vl_result.embeddings
                actual_count = embeddings.shape[0]
                logger.info(f"[Variant] VL embeddings: shape={embeddings.shape}, scaled_std={vl_result.scaled_std:.2f}")
            else:
                # Use text encoding (Qwen3-4B)
                prompt_output = self.pipeline.encoder.encode(
                    caption,
                    force_think_block=True,
                    layer_index=hidden_layer,
                )
                embeddings = prompt_output.embeddings[0]
                actual_count = embeddings.shape[0]

            logger.info(f"[Variant] Raw embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
            logger.debug(f"[Variant] Raw stats: min={embeddings.min():.2f}, max={embeddings.max():.2f}, mean={embeddings.mean():.4f}, std={embeddings.std():.2f}")

            # Apply compression if raw embeddings >1504 and compression mode specified
            if embeddings.shape[0] > MAX_TEXT_SEQ_LEN and compression_mode:
                logger.info(f"[Variant] Compressing {embeddings.shape[0]} -> {MAX_TEXT_SEQ_LEN} using {compression_mode}")
                embeddings = compress_embeddings(
                    embeddings, MAX_TEXT_SEQ_LEN, mode=compression_mode
                )
                logger.info(f"[Variant] After compression: shape={embeddings.shape}")

            # Apply fill mode to reach exact target length
            pre_fill_shape = embeddings.shape
            embeddings = apply_fill_mode(embeddings, target_length, mode=fill_mode)
            final_length = embeddings.shape[0]
            logger.info(f"[Variant] After fill ({fill_mode}): {pre_fill_shape} -> {embeddings.shape}")
            logger.debug(f"[Variant] Final stats: min={embeddings.min():.2f}, max={embeddings.max():.2f}, mean={embeddings.mean():.4f}, std={embeddings.std():.2f}")

            # Move to device
            device = self.pipeline.device
            dtype = self.pipeline.dtype
            embeddings = embeddings.to(device=device, dtype=dtype)

            # Generate with embeddings
            result = self.pipeline(
                prompt_embeds=embeddings,
                width=1024,
                height=1024,
                num_inference_steps=9,
                generator=torch.Generator().manual_seed(seed),
            )

            # Save image
            if hasattr(result, "images"):
                image = result.images[0]
            elif hasattr(result, "save"):
                image = result
            else:
                image = result[0]

            image.save(output_path)

            generation_time = time.time() - start_time

            # Compute SSIM
            source_image = Image.open(source_image_path)
            ssim_score = compute_ssim(source_image, image)

            return ExperimentResult(
                config=config,
                actual_token_count=actual_count,
                final_embedding_length=final_length,
                ssim_vs_source=ssim_score,
                generation_time_seconds=generation_time,
            )

        except Exception as e:
            logger.error(f"Variant failed: {e}")
            return ExperimentResult(
                config=config,
                actual_token_count=0,
                final_embedding_length=0,
                ssim_vs_source=None,
                generation_time_seconds=time.time() - start_time,
                error=str(e),
            )

    def run_study(
        self,
        prompts: list[str],
        seeds: list[int],
    ) -> list[ExperimentResult]:
        """Run the full caption length study."""
        all_results = []

        # Phase 1 & 2: Generate source images and captions
        source_data = []  # List of (prompt, seed, image_path, caption, token_count)

        for prompt in prompts:
            for seed in seeds:
                # Generate source
                source_path = self.generate_source_image(prompt, seed)

                # Caption
                caption, token_count = self.caption_source_image(source_path)
                source_data.append((prompt, seed, source_path, caption, token_count))

        # Only unload VL extractor if not using VL embeddings
        # (if using VL embeddings, we need it during generation phase)
        if not self.use_vl_embeddings:
            self.unload_vl_extractor()

        # Phase 3: Run variants
        for prompt, seed, source_path, caption, token_count in source_data:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {prompt} (seed={seed}, caption_tokens={token_count})")
            logger.info(f"use_vl_embeddings={self.use_vl_embeddings}, hidden_layers={self.hidden_layers}")
            if self.use_vl_embeddings:
                logger.info(f"vl_hidden_layers={self.vl_hidden_layers}, token_modes={self.token_modes}")
            logger.info(f"{'='*60}")

            # Run each target_length x fill_mode x hidden_layer combination
            for target_length in self.target_lengths:
                for fill_mode in self.fill_modes:
                    # Determine if compression is needed (when raw embeddings > target AND > 1504)
                    needs_compression = token_count > MAX_TEXT_SEQ_LEN
                    compression_modes_to_test = self.compression_modes if needs_compression else [None]

                    for compression_mode in compression_modes_to_test:
                        if self.use_vl_embeddings:
                            # VL embedding mode: sweep vl_hidden_layers x token_modes
                            for vl_hidden_layer in self.vl_hidden_layers:
                                for token_mode in self.token_modes:
                                    result = self.run_variant(
                                        source_prompt=prompt,
                                        source_image_path=source_path,
                                        caption=caption,
                                        caption_token_count=token_count,
                                        target_length=target_length,
                                        fill_mode=fill_mode,
                                        compression_mode=compression_mode,
                                        seed=seed,
                                        use_vl_embeddings=True,
                                        vl_hidden_layer=vl_hidden_layer,
                                        token_mode=token_mode,
                                    )
                                    all_results.append(result)
                        else:
                            # Text encoding mode: sweep hidden_layers
                            for hidden_layer in self.hidden_layers:
                                result = self.run_variant(
                                    source_prompt=prompt,
                                    source_image_path=source_path,
                                    caption=caption,
                                    caption_token_count=token_count,
                                    target_length=target_length,
                                    fill_mode=fill_mode,
                                    compression_mode=compression_mode,
                                    seed=seed,
                                    hidden_layer=hidden_layer,
                                    use_vl_embeddings=False,
                                )
                                all_results.append(result)

            # Generate comparison grids for this source
            if not self.dry_run:
                self._generate_grids(source_path, all_results)

        # Save results
        self._save_results(all_results)

        return all_results

    def _generate_grids(self, source_path: Path, results: list[ExperimentResult]):
        """Generate comparison grids for a source image."""
        # Filter results for this source
        source_results = [r for r in results if r.config.source_image_path == str(source_path)]
        if not source_results:
            return

        # Load source image
        source_image = Image.open(source_path)

        # Helper to build label with layer info
        def build_label(r: ExperimentResult, include_length: bool = True, include_fill: bool = True) -> str:
            parts = []
            if include_length:
                parts.append(f"len={r.config.target_length}")
            if include_fill:
                parts.append(r.config.fill_mode[:10])
            if r.config.use_vl_embeddings:
                parts.append(f"vlL{r.config.vl_hidden_layer}")
                parts.append(r.config.token_mode[:8])
            else:
                parts.append(f"L{r.config.hidden_layer}")
            if r.config.compression_mode:
                parts.append(r.config.compression_mode[:8])
            if r.ssim_vs_source:
                parts.append(f"SSIM={r.ssim_vs_source:.3f}")
            return "\n".join(parts)

        # Grid 1: Length comparison (content_only fill mode, first layer)
        length_images = [("Source", source_image)]
        first_layer = source_results[0].config.hidden_layer if source_results else -2
        for r in source_results:
            if r.config.fill_mode == "content_only" and r.error is None:
                # Only include first layer to avoid duplicates
                if r.config.hidden_layer == first_layer or r.config.vl_hidden_layer == self.vl_hidden_layers[0] if self.vl_hidden_layers else True:
                    label = build_label(r, include_fill=False)
                    try:
                        img = Image.open(r.config.output_path)
                        length_images.append((label, img))
                    except Exception:
                        pass

        if len(length_images) > 1:
            grid_path = self.output_dir / "grids" / f"{source_path.stem}_length_grid.png"
            create_comparison_grid(length_images, grid_path, cols=4)

        # Grid 2: Fill mode comparison (for a fixed target length like 300)
        fill_images = [("Source", source_image)]
        target_len = 300  # Pick a representative length
        for r in source_results:
            if r.config.target_length == target_len and r.error is None:
                label = build_label(r, include_length=False)
                try:
                    img = Image.open(r.config.output_path)
                    fill_images.append((label, img))
                except Exception:
                    pass

        if len(fill_images) > 1:
            grid_path = self.output_dir / "grids" / f"{source_path.stem}_fill_mode_grid.png"
            create_comparison_grid(fill_images, grid_path, cols=3)

        # Grid 3: Hidden layer comparison (content_only, fixed length 300)
        layer_images = [("Source", source_image)]
        for r in source_results:
            if r.config.fill_mode == "content_only" and r.config.target_length == 300 and r.error is None:
                if r.config.use_vl_embeddings:
                    label = f"VL L{r.config.vl_hidden_layer}\n{r.config.token_mode}"
                else:
                    label = f"L{r.config.hidden_layer}"
                if r.ssim_vs_source:
                    label += f"\nSSIM={r.ssim_vs_source:.3f}"
                try:
                    img = Image.open(r.config.output_path)
                    layer_images.append((label, img))
                except Exception:
                    pass

        if len(layer_images) > 1:
            grid_path = self.output_dir / "grids" / f"{source_path.stem}_layer_grid.png"
            create_comparison_grid(layer_images, grid_path, cols=4)

    def _save_results(self, results: list[ExperimentResult]):
        """Save experiment results to JSON and CSV."""
        # JSON
        json_path = self.output_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "total_experiments": len(results),
                    "successful": len([r for r in results if r.error is None]),
                    "failed": len([r for r in results if r.error is not None]),
                    "results": [
                        {
                            "config": asdict(r.config),
                            "actual_token_count": r.actual_token_count,
                            "final_embedding_length": r.final_embedding_length,
                            "ssim_vs_source": r.ssim_vs_source,
                            "generation_time_seconds": r.generation_time_seconds,
                            "error": r.error,
                        }
                        for r in results
                    ],
                },
                f,
                indent=2,
            )
        logger.info(f"Results saved to {json_path}")

        # CSV summary
        csv_path = self.output_dir / "summary.csv"
        with open(csv_path, "w") as f:
            f.write(
                "source_prompt,seed,target_length,fill_mode,compression_mode,"
                "actual_tokens,final_length,ssim,time_seconds,error\n"
            )
            for r in results:
                f.write(
                    f'"{r.config.source_prompt}",'
                    f"{r.config.seed},"
                    f"{r.config.target_length},"
                    f"{r.config.fill_mode},"
                    f"{r.config.compression_mode or ''},"
                    f"{r.actual_token_count},"
                    f"{r.final_embedding_length},"
                    f"{r.ssim_vs_source or ''},"
                    f"{r.generation_time_seconds:.2f},"
                    f'"{r.error or ""}"\n'
                )
        logger.info(f"Summary saved to {csv_path}")


# ============================================================
# CLI
# ============================================================


def main():
    parser = argparse.ArgumentParser(
        description="Caption Length Study for Z-Image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Config
    parser.add_argument("--config", type=str, help="Path to TOML config file")
    parser.add_argument("--profile", type=str, default="default", help="Config profile")

    # Model paths
    parser.add_argument("--model-path", type=str, help="Path to Z-Image model")
    parser.add_argument("--vl-model-path", type=str, help="Path to Qwen3-VL model")

    # Devices
    parser.add_argument("--text-encoder-device", type=str, default="cpu")
    parser.add_argument("--dit-device", type=str, default="cuda")
    parser.add_argument("--vae-device", type=str, default="cuda")

    # Experiment parameters
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["A cat", "A mountain landscape", "A woman portrait"],
        help="Simple prompts for source image generation",
    )
    parser.add_argument(
        "--seeds", type=str, default="42", help="Comma-separated seeds"
    )
    parser.add_argument(
        "--target-lengths",
        type=str,
        default="50,150,300,600,1000,1504",
        help="Comma-separated final embedding lengths to test",
    )
    parser.add_argument(
        "--fill-modes",
        type=str,
        default="content_only,pad_end_zero,pad_end_mean,pad_middle_zero,filler_repeat",
        help="Comma-separated fill modes: content_only, pad_end_zero, pad_end_mean, pad_middle_zero, filler_repeat",
    )
    parser.add_argument(
        "--compression-modes",
        type=str,
        default="truncate,interpolate,pool,attention_pool",
        help="Comma-separated compression modes (used when caption >1504 tokens)",
    )

    # Hidden layer selection
    parser.add_argument(
        "--hidden-layers",
        type=str,
        default="-2",
        help="Comma-separated hidden layers to test for text encoding (e.g., -2,-6,-8). Default: -2 (penultimate)",
    )
    parser.add_argument(
        "--vl-hidden-layers",
        type=str,
        default=None,
        help="Comma-separated hidden layers for VL embedding extraction (e.g., -6,-8). Enables VL embedding mode.",
    )

    # Token mode selection (for VL embeddings)
    parser.add_argument(
        "--token-modes",
        type=str,
        default=None,
        help="Comma-separated token modes for VL: full,text_only,image_only,image_no_markers. Requires --vl-hidden-layers.",
    )

    # VL embedding mode
    parser.add_argument(
        "--use-vl-embeddings",
        action="store_true",
        help="Use VL embeddings instead of text encoding. Requires --vl-hidden-layers.",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/caption_length_study",
        help="Output directory",
    )

    # Flags
    parser.add_argument("--dry-run", action="store_true", help="Show what would run")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config: RuntimeConfig | None = None
    if args.config:
        config = load_runtime_config(args)
        logger.info(f"Loaded config from {args.config} (profile: {args.profile})")

    # Resolve model path
    model_path = args.model_path
    if not model_path and config:
        model_path = config.model_path
    if not args.dry_run and not model_path:
        parser.error("--model-path is required unless using --dry-run or --config")

    # Resolve VL model path from config first, CLI overrides
    vl_model_path = args.vl_model_path
    if not vl_model_path and config and hasattr(config, "vl_model_path"):
        vl_model_path = config.vl_model_path
        if vl_model_path:
            logger.info(f"Using VL model path from config: {vl_model_path}")

    # Resolve devices from config first, CLI overrides
    text_encoder_device = args.text_encoder_device
    if config and hasattr(config, "encoder_device") and args.text_encoder_device == "cpu":
        # Only use config if CLI wasn't explicitly set (default is "cpu")
        text_encoder_device = config.encoder_device

    dit_device = args.dit_device
    if config and hasattr(config, "dit_device") and args.dit_device == "cuda":
        dit_device = config.dit_device

    vae_device = args.vae_device
    if config and hasattr(config, "vae_device") and args.vae_device == "cuda":
        vae_device = config.vae_device

    # Resolve hidden layer from config first, CLI overrides
    default_hidden_layer = -2
    if config and hasattr(config, "hidden_layer"):
        default_hidden_layer = config.hidden_layer
        logger.info(f"Using encoder hidden layer from config: {default_hidden_layer}")

    # Resolve VL hidden layer from config
    default_vl_hidden_layer = -2
    if config and hasattr(config, "vl_hidden_layer"):
        default_vl_hidden_layer = config.vl_hidden_layer
        logger.info(f"Using VL hidden layer from config: {default_vl_hidden_layer}")

    # Parse lists
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    target_lengths = [int(t.strip()) for t in args.target_lengths.split(",")]
    fill_modes = [m.strip() for m in args.fill_modes.split(",")]
    compression_modes = [m.strip() for m in args.compression_modes.split(",")]

    # Hidden layers: use config default if CLI wasn't explicitly changed from default
    if args.hidden_layers == "-2":
        # CLI is at default, check if config has different value
        hidden_layers = [default_hidden_layer]
    else:
        # CLI was explicitly set, use those values
        hidden_layers = [int(h.strip()) for h in args.hidden_layers.split(",")]

    # Parse VL-specific parameters
    vl_hidden_layers = None
    token_modes = None
    use_vl_embeddings = args.use_vl_embeddings

    if args.vl_hidden_layers:
        vl_hidden_layers = [int(h.strip()) for h in args.vl_hidden_layers.split(",")]
        use_vl_embeddings = True  # Implied by providing VL hidden layers

    if args.token_modes:
        token_modes = [m.strip() for m in args.token_modes.split(",")]
        valid_modes = {"full", "text_only", "image_only", "image_no_markers"}
        for mode in token_modes:
            if mode not in valid_modes:
                parser.error(f"Invalid token mode: {mode}. Must be one of: {valid_modes}")

    # Validation
    if use_vl_embeddings and not vl_hidden_layers:
        vl_hidden_layers = [default_vl_hidden_layer]  # Use config default VL layer
        logger.info(f"VL embeddings enabled with config layer {default_vl_hidden_layer}")

    if token_modes and not use_vl_embeddings:
        parser.error("--token-modes requires --use-vl-embeddings or --vl-hidden-layers")

    # Run study
    study = CaptionLengthStudy(
        model_path=model_path or "",
        vl_model_path=vl_model_path,
        output_dir=Path(args.output_dir),
        text_encoder_device=text_encoder_device,
        dit_device=dit_device,
        vae_device=vae_device,
        target_lengths=target_lengths,
        fill_modes=fill_modes,
        compression_modes=compression_modes,
        hidden_layers=hidden_layers,
        vl_hidden_layers=vl_hidden_layers,
        token_modes=token_modes,
        use_vl_embeddings=use_vl_embeddings,
        dry_run=args.dry_run,
    )

    results = study.run_study(prompts=args.prompts, seeds=seeds)

    # Summary
    successful = len([r for r in results if r.error is None])
    failed = len([r for r in results if r.error is not None])
    print(f"\nStudy complete: {successful} successful, {failed} failed")
    print(f"Results saved to: {args.output_dir}")

    # Print SSIM summary if available
    ssim_scores = [r.ssim_vs_source for r in results if r.ssim_vs_source is not None]
    if ssim_scores:
        print(f"SSIM range: {min(ssim_scores):.3f} - {max(ssim_scores):.3f}")
        print(f"SSIM mean: {sum(ssim_scores)/len(ssim_scores):.3f}")


if __name__ == "__main__":
    main()
