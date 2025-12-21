"""
Training dataset for Z-Image models.

Supports:
- JSON/JSONL/CSV metadata formats
- Image preprocessing with resolution control
- Template-based prompts
- Optional cached embeddings

Usage:
    from llm_dit.training.data import TrainingDataset

    dataset = TrainingDataset(
        metadata_path="data/train.jsonl",
        image_dir="data/images",
        max_resolution=1024,
    )

    for item in dataset:
        print(item["prompt"], item["image"].size)
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from PIL import Image

logger = logging.getLogger(__name__)


class TrainingDataset(Dataset):
    """
    Dataset for training Z-Image models.

    Supports:
        - JSON/JSONL/CSV metadata formats
        - Image preprocessing with resolution control
        - Template-based prompts
        - Dataset repetition for small datasets
        - Custom transform functions

    Metadata Format:
        JSONL (recommended):
        ```
        {"prompt": "A cat sleeping", "image": "images/001.jpg"}
        {"prompt": "A dog playing", "image": "images/002.jpg", "template": "photorealistic"}
        ```

        JSON:
        ```json
        [
            {"prompt": "A cat sleeping", "image": "images/001.jpg"},
            {"prompt": "A dog playing", "image": "images/002.jpg"}
        ]
        ```

        CSV:
        ```
        prompt,image,template
        "A cat sleeping","images/001.jpg",
        "A dog playing","images/002.jpg","photorealistic"
        ```

    Attributes:
        data: List of metadata dictionaries
        image_dir: Base directory for images
        max_resolution: Maximum image dimension
        resolution_divisor: Resolution must be divisible by this
        repeat: Number of times to repeat dataset
    """

    def __init__(
        self,
        metadata_path: Union[str, Path],
        image_dir: Union[str, Path],
        max_resolution: int = 1024,
        resolution_divisor: int = 16,
        repeat: int = 1,
        transform: Optional[Callable[[Image.Image], Image.Image]] = None,
        shuffle: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            metadata_path: Path to metadata file (JSON/JSONL/CSV)
            image_dir: Base directory for training images
            max_resolution: Maximum image resolution (maintains aspect ratio)
            resolution_divisor: Resolution must be divisible by this (VAE constraint)
            repeat: Number of times to repeat dataset per epoch
            transform: Optional image transform function
            shuffle: Shuffle data on load
        """
        self.image_dir = Path(image_dir)
        self.max_resolution = max_resolution
        self.resolution_divisor = resolution_divisor
        self.repeat = repeat
        self.transform = transform

        # Load metadata
        self.data = self._load_metadata(metadata_path)

        if shuffle:
            import random
            random.shuffle(self.data)

        logger.info(
            f"Loaded {len(self.data)} items from {metadata_path} "
            f"(x{repeat} repeat = {len(self)} total)"
        )

    def _load_metadata(self, path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Load metadata from JSON/JSONL/CSV file.

        Args:
            path: Path to metadata file

        Returns:
            List of metadata dictionaries
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Metadata file not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, dict):
                # Handle {"items": [...]} format
                data = data.get("items", data.get("data", [data]))
            return data

        elif suffix == ".jsonl":
            data = []
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data

        elif suffix == ".csv":
            try:
                import pandas as pd
                df = pd.read_csv(path)
                return df.to_dict("records")
            except ImportError:
                raise ImportError(
                    "pandas is required for CSV metadata. "
                    "Install with: pip install pandas"
                )

        else:
            raise ValueError(
                f"Unsupported metadata format: {suffix}. "
                "Supported: .json, .jsonl, .csv"
            )

    def _snap_to_multiple(self, value: int, multiple: int) -> int:
        """Snap value to nearest multiple."""
        return round(value / multiple) * multiple

    def _process_image(self, image: Image.Image) -> tuple:
        """
        Process image: resize and ensure valid dimensions.

        Args:
            image: Input image

        Returns:
            Tuple of (processed_image, width, height)
        """
        width, height = image.size

        # Calculate scale factor to fit within max_resolution
        max_dim = max(width, height)
        if max_dim > self.max_resolution:
            scale = self.max_resolution / max_dim
            width = int(width * scale)
            height = int(height * scale)

        # Snap to divisor (VAE constraint)
        width = self._snap_to_multiple(width, self.resolution_divisor)
        height = self._snap_to_multiple(height, self.resolution_divisor)

        # Ensure minimum size
        width = max(width, self.resolution_divisor)
        height = max(height, self.resolution_divisor)

        # Resize if needed
        if (width, height) != image.size:
            image = image.resize((width, height), Image.LANCZOS)

        return image, width, height

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get dataset item.

        Args:
            idx: Item index

        Returns:
            Dictionary with keys:
                - prompt: Text prompt
                - image: PIL Image
                - width: Image width
                - height: Image height
                - template: Optional template name
                - system_prompt: Optional system prompt
                - thinking_content: Optional thinking content
        """
        # Handle repeat
        actual_idx = idx % len(self.data)
        item = self.data[actual_idx]

        # Load image
        image_path = self.image_dir / item["image"]
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")

        # Process image
        image, width, height = self._process_image(image)

        # Apply optional transform
        if self.transform is not None:
            image = self.transform(image)

        # Build output
        result = {
            "prompt": item["prompt"],
            "image": image,
            "width": width,
            "height": height,
        }

        # Optional fields
        for key in ["template", "system_prompt", "thinking_content"]:
            if key in item:
                result[key] = item[key]

        return result

    def __len__(self) -> int:
        """Return total length including repeats."""
        return len(self.data) * self.repeat


class CachedEmbeddingDataset(Dataset):
    """
    Dataset with pre-computed embeddings.

    Use this for faster training when embeddings are expensive to compute.
    Embeddings should be saved as .pt files with matching names to images.

    Example:
        data/
            images/
                001.jpg
                002.jpg
            embeddings/
                001.pt  # Contains {"prompt_embeds": tensor, "latents": tensor}
                002.pt
    """

    def __init__(
        self,
        metadata_path: Union[str, Path],
        embedding_dir: Union[str, Path],
        repeat: int = 1,
    ):
        """
        Initialize cached embedding dataset.

        Args:
            metadata_path: Path to metadata file
            embedding_dir: Directory containing .pt embedding files
            repeat: Number of times to repeat dataset
        """
        self.embedding_dir = Path(embedding_dir)
        self.repeat = repeat

        # Load metadata
        base_dataset = TrainingDataset(
            metadata_path=metadata_path,
            image_dir=".",  # Not used
            repeat=1,
        )
        self.data = base_dataset.data

        logger.info(f"Loaded {len(self.data)} cached embeddings")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get cached embeddings.

        Returns:
            Dictionary with:
                - input_latents: Encoded image latents
                - prompt_embeds: Text embeddings
        """
        actual_idx = idx % len(self.data)
        item = self.data[actual_idx]

        # Derive embedding filename from image name
        image_name = Path(item["image"]).stem
        embedding_path = self.embedding_dir / f"{image_name}.pt"

        if not embedding_path.exists():
            raise FileNotFoundError(f"Cached embeddings not found: {embedding_path}")

        return torch.load(embedding_path, weights_only=True)

    def __len__(self) -> int:
        return len(self.data) * self.repeat
