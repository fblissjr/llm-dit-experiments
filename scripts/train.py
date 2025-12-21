#!/usr/bin/env python
"""
Z-Image LoRA training script.

Train LoRA adapters or full models on custom datasets.

Usage:
    # LoRA training (recommended for RTX 4090)
    uv run scripts/train.py \
        --model-path /path/to/z-image-turbo \
        --use-lora \
        --lora-rank 32 \
        --dataset-metadata data/train.jsonl \
        --dataset-images data/images \
        --num-epochs 10 \
        --learning-rate 1e-4 \
        --output-dir ./checkpoints/lora_style

    # Full transformer training (requires A100 80GB)
    uv run scripts/train.py \
        --model-path /path/to/z-image-turbo \
        --trainable-models transformer \
        --dataset-metadata data/train.jsonl \
        --dataset-images data/images \
        --num-epochs 3 \
        --learning-rate 1e-6 \
        --output-dir ./checkpoints/full_dit

    # Multi-GPU training
    accelerate launch --multi_gpu scripts/train.py \
        --model-path /path/to/z-image-turbo \
        --use-lora \
        --dataset-metadata data/train.jsonl \
        --dataset-images data/images

Dataset Format:
    JSONL (recommended):
    ```
    {"prompt": "A cat sleeping", "image": "images/001.jpg"}
    {"prompt": "A dog playing", "image": "images/002.jpg", "template": "photorealistic"}
    ```

    CSV:
    ```
    prompt,image,template
    "A cat sleeping","images/001.jpg",
    "A dog playing","images/002.jpg","photorealistic"
    ```
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs

from llm_dit.training import (
    ZImageTrainingModule,
    TrainingDataset,
    ModelLogger,
    TensorBoardLogger,
    TrainingConfig,
    launch_training_task,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Z-Image LoRA or full model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to Z-Image model weights",
    )
    parser.add_argument(
        "--trainable-models",
        type=str,
        default=None,
        help="Comma-separated list of models to train (e.g., transformer,context_refiner)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="sft",
        choices=["sft", "distill"],
        help="Training task type",
    )

    # LoRA
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA adapters for efficient fine-tuning",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=32,
        help="LoRA rank (lower = fewer params, higher = more capacity)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha (defaults to rank)",
    )
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="to_q,to_k,to_v,to_out.0",
        help="Comma-separated target modules for LoRA",
    )
    parser.add_argument(
        "--lora-checkpoint",
        type=str,
        default=None,
        help="Path to resume LoRA training from checkpoint",
    )

    # Dataset
    parser.add_argument(
        "--dataset-metadata",
        type=str,
        required=True,
        help="Path to dataset metadata file (JSON/JSONL/CSV)",
    )
    parser.add_argument(
        "--dataset-images",
        type=str,
        required=True,
        help="Base directory for training images",
    )
    parser.add_argument(
        "--dataset-repeat",
        type=int,
        default=1,
        help="Times to repeat dataset per epoch",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=1024,
        help="Maximum image resolution",
    )

    # Training
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm (0 to disable)",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="constant",
        choices=["constant", "constant_with_warmup", "cosine", "cosine_with_warmup", "linear", "linear_with_warmup"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Number of warmup steps",
    )

    # Optimization
    parser.add_argument(
        "--use-gradient-checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing (default: True)",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing",
    )
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision mode",
    )

    # Checkpointing
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps (None = epoch only)",
    )
    parser.add_argument(
        "--save-epochs",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log metrics every N steps",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging",
    )

    # Hardware
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def main():
    """Main training entry point."""
    args = parse_args()

    # Handle gradient checkpointing flag
    use_gradient_checkpointing = args.use_gradient_checkpointing
    if args.no_gradient_checkpointing:
        use_gradient_checkpointing = False

    # Parse trainable models
    trainable_models = None
    if args.trainable_models:
        trainable_models = [m.strip() for m in args.trainable_models.split(",")]

    # Parse LoRA target modules
    lora_target_modules = [m.strip() for m in args.lora_target_modules.split(",")]

    # Build config
    config = TrainingConfig(
        model_path=args.model_path,
        task=args.task,
        trainable_models=trainable_models,
        use_lora=args.use_lora,
        lora_base_model="transformer",
        lora_target_modules=lora_target_modules,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha or args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        dataset_metadata_path=args.dataset_metadata,
        dataset_image_dir=args.dataset_images,
        dataset_repeat=args.dataset_repeat,
        max_resolution=args.max_resolution,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        use_gradient_checkpointing=use_gradient_checkpointing,
        mixed_precision=args.mixed_precision,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        save_epochs=args.save_epochs,
        logging_steps=args.logging_steps,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    # Create accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with="tensorboard" if args.tensorboard else None,
        project_dir=args.output_dir if args.tensorboard else None,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=False)
        ],
    )

    # Set seed
    if config.seed is not None:
        torch.manual_seed(config.seed)

    # Create dataset
    dataset = TrainingDataset(
        metadata_path=config.dataset_metadata_path,
        image_dir=config.dataset_image_dir,
        max_resolution=config.max_resolution,
        repeat=config.dataset_repeat,
    )

    # Create training module
    model = ZImageTrainingModule(
        config=config,
        device=accelerator.device,
        dtype=torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16,
    )

    # Create logger
    model_logger = ModelLogger(
        output_dir=config.output_dir,
        remove_prefix="pipe.transformer.",
    )

    # Optional TensorBoard logger
    tb_logger = None
    if args.tensorboard:
        tb_logger = TensorBoardLogger(f"{config.output_dir}/tensorboard")

    # Log training info
    if accelerator.is_main_process:
        logger.info(f"Training {model.num_trainable_params():,} parameters")
        logger.info(f"Dataset: {len(dataset)} items")
        logger.info(f"Output: {config.output_dir}")

    # Launch training
    launch_training_task(
        accelerator=accelerator,
        dataset=dataset,
        model=model,
        model_logger=model_logger,
        config=config,
        tb_logger=tb_logger,
    )


if __name__ == "__main__":
    main()
