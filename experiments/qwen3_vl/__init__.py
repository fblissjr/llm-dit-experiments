"""
Qwen3-VL Vision Conditioning for Z-Image.

This module provides tools for using Qwen3-VL vision embeddings to condition
Z-Image generation. Key discovery: Qwen3-VL's text model hidden states
(after processing an image) are compatible with Z-Image because both use
Qwen3-4B architecture (hidden_size=2560).

Main scripts:
    - extract_embeddings.py: Extract VL embeddings from reference images
    - blend_and_generate.py: Blend VL + text embeddings and generate
    - run_comparison.py: Run comprehensive comparison experiments

Key parameters:
    - alpha: Interpolation ratio (0.0=text, 1.0=VL, recommended: 0.3)
    - hidden_layer: Which layer to extract (-2 recommended)
    - image_tokens_only: Use only image tokens vs full sequence
    - text_with_image: Include text description with reference image

See README.md and CONDITIONING_GUIDE.md for detailed documentation.
"""
