#!/usr/bin/env python
"""
Integration test for QwenImageDiffusersPipeline wrapper.

Tests:
1. Pipeline loading with CPU offload
2. Image decomposition
3. Layer editing (requires edit model download - uses Qwen-Image-Edit-2511)
4. Multi-image editing (2511 feature)
5. Device management
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add coderef diffusers to path BEFORE importing anything
coderef_diffusers = Path(__file__).parent.parent.parent / "coderef" / "diffusers" / "src"
sys.path.insert(0, str(coderef_diffusers))

# Add experiments to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from PIL import Image

from experiments.utils import save_image_grid, save_metadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_pipeline_import():
    """Test that the pipeline can be imported."""
    logger.info("=" * 60)
    logger.info("TEST: Pipeline Import")
    logger.info("=" * 60)

    from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline
    logger.info("SUCCESS: QwenImageDiffusersPipeline imported")
    return True


def test_pipeline_loading(model_path: str):
    """Test pipeline loading with CPU offload."""
    logger.info("=" * 60)
    logger.info("TEST: Pipeline Loading")
    logger.info("=" * 60)

    from llm_dit.pipelines.qwen_image_diffusers import QwenImageDiffusersPipeline

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        vram_before = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"VRAM before: {vram_before:.2f} GB")

    start = time.time()
    pipe = QwenImageDiffusersPipeline.from_pretrained(
        model_path,
        cpu_offload=True,
        load_edit_model=False,
    )
    load_time = time.time() - start

    if torch.cuda.is_available():
        vram_after = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"VRAM after: {vram_after:.2f} GB")

    logger.info(f"Pipeline loaded in {load_time:.1f}s")
    logger.info(f"Device: {pipe.device}")
    logger.info(f"Dtype: {pipe.dtype}")
    logger.info(f"Has edit model: {pipe.has_edit_model}")

    return pipe


def test_decomposition(pipe, input_image: Image.Image, output_dir: Path):
    """Test image decomposition."""
    logger.info("=" * 60)
    logger.info("TEST: Image Decomposition")
    logger.info("=" * 60)

    start = time.time()
    layers = pipe.decompose(
        image=input_image,
        prompt="A colorful scene with objects",
        layer_num=4,
        resolution=640,
        num_inference_steps=30,
        cfg_scale=4.0,
        seed=42,
    )
    decompose_time = time.time() - start

    logger.info(f"Decomposition completed in {decompose_time:.1f}s")
    logger.info(f"Generated {len(layers)} layers")

    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"Peak VRAM: {peak_vram:.2f} GB")

    # Save layers
    decompose_dir = output_dir / "decompose"
    decompose_dir.mkdir(parents=True, exist_ok=True)

    for i, layer in enumerate(layers):
        path = decompose_dir / f"layer_{i:02d}.png"
        layer.save(path)
        logger.info(f"  Saved: {path} ({layer.size}, mode={layer.mode})")

    return layers


def test_layer_editing(pipe, layer: Image.Image, output_dir: Path):
    """Test layer editing (requires edit model download)."""
    logger.info("=" * 60)
    logger.info("TEST: Layer Editing")
    logger.info("=" * 60)

    logger.info("This test will download the edit model if not already present...")
    logger.info("Edit model: Qwen/Qwen-Image-Edit-2511 (~20 GB)")

    start = time.time()
    edited = pipe.edit_layer(
        layer_image=layer,
        instruction="Change the colors to be more vibrant",
        num_inference_steps=30,
        cfg_scale=4.0,
        seed=42,
    )
    edit_time = time.time() - start

    logger.info(f"Layer edit completed in {edit_time:.1f}s")
    logger.info(f"Edited layer: {edited.size}, mode={edited.mode}")

    # Save edited layer
    edit_dir = output_dir / "edit"
    edit_dir.mkdir(parents=True, exist_ok=True)

    original_path = edit_dir / "original_layer.png"
    layer.save(original_path)
    logger.info(f"  Saved original: {original_path}")

    edited_path = edit_dir / "edited_layer.png"
    edited.save(edited_path)
    logger.info(f"  Saved edited: {edited_path}")

    return edited


def test_edit_status(pipe):
    """Test edit model status."""
    logger.info("=" * 60)
    logger.info("TEST: Edit Model Status")
    logger.info("=" * 60)

    has_edit = pipe.has_edit_model
    edit_path = getattr(pipe, '_edit_model_path', None)

    logger.info(f"Has edit model: {has_edit}")
    logger.info(f"Edit model path: {edit_path}")

    return has_edit


def test_multi_image_editing(pipe, images: list, output_dir: Path):
    """Test multi-image editing (Qwen-Image-Edit-2511 feature).

    Combines multiple images based on text instruction.
    """
    logger.info("=" * 60)
    logger.info("TEST: Multi-Image Editing (2511 Feature)")
    logger.info("=" * 60)

    if len(images) < 2:
        logger.warning("Need at least 2 images for multi-edit test. Skipping.")
        return None

    logger.info(f"Combining {len(images)} images...")
    logger.info("This test will download the edit model if not already present...")
    logger.info("Edit model: Qwen/Qwen-Image-Edit-2511 (~20 GB)")

    start = time.time()
    combined = pipe.edit_multi(
        images=images,
        instruction="Place both subjects side by side in a natural setting",
        num_inference_steps=40,
        cfg_scale=4.0,
        seed=42,
    )
    edit_time = time.time() - start

    logger.info(f"Multi-edit completed in {edit_time:.1f}s")
    logger.info(f"Combined image: {combined.size}, mode={combined.mode}")

    # Save result
    multi_dir = output_dir / "multi_edit"
    multi_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(images):
        input_path = multi_dir / f"input_{i:02d}.png"
        img.save(input_path)
        logger.info(f"  Saved input {i}: {input_path}")

    combined_path = multi_dir / "combined.png"
    combined.save(combined_path)
    logger.info(f"  Saved combined: {combined_path}")

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Integration test for QwenImageDiffusersPipeline"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(Path.home() / "Storage" / "Qwen_Qwen-Image-Layered"),
        help="Path to Qwen-Image-Layered model",
    )
    parser.add_argument(
        "--input-image",
        type=str,
        default="experiments/inputs/test_scene.png",
        help="Input image to decompose",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/qwen_layered_test/wrapper_test",
        help="Output directory for test results",
    )
    parser.add_argument(
        "--skip-edit",
        action="store_true",
        help="Skip layer editing test (avoids downloading edit model)",
    )
    parser.add_argument(
        "--skip-multi-edit",
        action="store_true",
        help="Skip multi-image editing test (2511 feature)",
    )
    parser.add_argument(
        "--second-image",
        type=str,
        default=None,
        help="Second input image for multi-edit test (optional, creates dummy if not provided)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Test 1: Import
    try:
        results["import"] = test_pipeline_import()
    except Exception as e:
        logger.error(f"Import test failed: {e}")
        results["import"] = False

    # Test 2: Loading
    pipe = None
    try:
        pipe = test_pipeline_loading(args.model_path)
        results["loading"] = True
    except Exception as e:
        logger.error(f"Loading test failed: {e}")
        results["loading"] = False
        return 1

    # Load input image
    input_path = Path(args.input_image)
    if input_path.exists():
        input_image = Image.open(input_path).convert("RGBA")
        logger.info(f"Loaded input image: {input_path} ({input_image.size})")
    else:
        logger.warning(f"Input image not found, creating dummy: {input_path}")
        input_image = Image.new("RGBA", (640, 640), (128, 128, 128, 255))

    # Test 3: Decomposition
    layers = None
    try:
        layers = test_decomposition(pipe, input_image, output_dir)
        results["decomposition"] = True
    except Exception as e:
        logger.error(f"Decomposition test failed: {e}")
        import traceback
        traceback.print_exc()
        results["decomposition"] = False

    # Test 4: Layer Editing (optional)
    if not args.skip_edit and layers and len(layers) > 1:
        try:
            test_layer_editing(pipe, layers[1], output_dir)
            results["editing"] = True
        except Exception as e:
            logger.error(f"Editing test failed: {e}")
            import traceback
            traceback.print_exc()
            results["editing"] = False
    else:
        logger.info("Skipping layer editing test (--skip-edit or no layers)")
        results["editing"] = "skipped"

    # Test 5: Edit Status
    try:
        test_edit_status(pipe)
        results["edit_status"] = True
    except Exception as e:
        logger.error(f"Edit status test failed: {e}")
        results["edit_status"] = False

    # Test 6: Multi-Image Editing (2511 feature, optional)
    if not args.skip_edit and not args.skip_multi_edit:
        try:
            # Prepare images for multi-edit test
            multi_images = [input_image.convert("RGB")]

            # Load or create second image
            if args.second_image and Path(args.second_image).exists():
                second_img = Image.open(args.second_image).convert("RGB")
                logger.info(f"Loaded second image: {args.second_image}")
            else:
                # Create a dummy second image with different color
                second_img = Image.new("RGB", (640, 640), (200, 100, 50))
                logger.info("Created dummy second image for multi-edit test")

            multi_images.append(second_img)

            test_multi_image_editing(pipe, multi_images, output_dir)
            results["multi_edit"] = True
        except Exception as e:
            logger.error(f"Multi-edit test failed: {e}")
            import traceback
            traceback.print_exc()
            results["multi_edit"] = False
    else:
        logger.info("Skipping multi-edit test (--skip-edit or --skip-multi-edit)")
        results["multi_edit"] = "skipped"

    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    all_passed = True
    for test_name, result in results.items():
        status = "PASS" if result is True else ("SKIP" if result == "skipped" else "FAIL")
        logger.info(f"  {test_name}: {status}")
        if result is False:
            all_passed = False

    logger.info("=" * 60)

    # Save test results metadata
    save_metadata(
        output_dir / "test_results.json",
        model_path=args.model_path,
        input_image=str(args.input_image),
        skip_edit=args.skip_edit,
        test_results=results,
        all_passed=all_passed,
    )

    logger.info(f"Output saved to: {output_dir}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
