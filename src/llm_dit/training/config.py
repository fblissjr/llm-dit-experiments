"""
Training configuration dataclasses.

Usage:
    from llm_dit.training import TrainingConfig, LoRAConfig

    config = TrainingConfig(
        model_path="/path/to/model",
        use_lora=True,
        lora_rank=32,
    )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class LoRAConfig:
    """
    LoRA (Low-Rank Adaptation) configuration.

    LoRA adds trainable low-rank matrices to frozen model weights,
    enabling efficient fine-tuning with minimal memory overhead.

    Attributes:
        base_model: Name of model to apply LoRA to (e.g., "transformer")
        target_modules: Module names to apply LoRA to (e.g., ["q", "k", "v", "o"])
        rank: LoRA rank (lower = fewer parameters, higher = more capacity)
        alpha: LoRA alpha scaling factor (typically equal to rank)
        dropout: LoRA dropout probability
        checkpoint: Path to existing LoRA checkpoint to resume from
    """
    base_model: str = "transformer"
    target_modules: List[str] = field(
        default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"]
    )
    rank: int = 32
    alpha: int = 32
    dropout: float = 0.0
    checkpoint: Optional[str] = None

    @classmethod
    def from_dict(cls, config_dict: dict) -> "LoRAConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """
    Complete training configuration.

    Attributes:
        model_path: Path to base model weights
        task: Training task type (sft, distill, consistency)
        trainable_models: List of model components to train

        use_lora: Whether to use LoRA adapters
        lora_base_model: Model to apply LoRA to
        lora_target_modules: Modules to apply LoRA to
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_checkpoint: Path to resume LoRA training

        dataset_metadata_path: Path to dataset metadata file
        dataset_image_dir: Base directory for training images
        dataset_repeat: Times to repeat dataset per epoch

        num_epochs: Number of training epochs
        batch_size: Training batch size
        gradient_accumulation_steps: Gradient accumulation steps
        learning_rate: Optimizer learning rate
        weight_decay: AdamW weight decay
        adam_beta1: AdamW beta1
        adam_beta2: AdamW beta2
        adam_epsilon: AdamW epsilon
        max_grad_norm: Gradient clipping norm
        lr_scheduler: Learning rate scheduler type
        warmup_steps: Number of warmup steps

        use_gradient_checkpointing: Enable gradient checkpointing
        use_gradient_checkpointing_offload: Offload checkpoints to CPU
        mixed_precision: Mixed precision mode (no, fp16, bf16)

        output_dir: Directory for checkpoints
        save_steps: Save checkpoint every N steps
        save_epochs: Save checkpoint every N epochs
        logging_steps: Log every N steps
        remove_prefix_in_ckpt: Prefix to remove from state dict keys

        device: Training device
        num_workers: DataLoader workers
        seed: Random seed
    """
    # Model
    model_path: str = ""
    task: str = "sft"
    trainable_models: Optional[List[str]] = None

    # LoRA
    use_lora: bool = False
    lora_base_model: str = "transformer"
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"]
    )
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_checkpoint: Optional[str] = None

    # Dataset
    dataset_metadata_path: Optional[str] = None
    dataset_image_dir: Optional[str] = None
    dataset_repeat: int = 1
    max_resolution: int = 1024

    # Training
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lr_scheduler: str = "constant"
    warmup_steps: int = 0

    # Optimization
    use_gradient_checkpointing: bool = True
    use_gradient_checkpointing_offload: bool = False
    mixed_precision: str = "bf16"

    # Checkpointing
    output_dir: str = "./checkpoints"
    save_steps: Optional[int] = None
    save_epochs: int = 1
    logging_steps: int = 10
    remove_prefix_in_ckpt: Optional[str] = "pipe.transformer."

    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42

    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_toml(cls, path: Union[str, Path]) -> "TrainingConfig":
        """
        Load config from TOML file.

        Args:
            path: Path to TOML config file

        Returns:
            TrainingConfig instance
        """
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        with open(path, "rb") as f:
            config = tomllib.load(f)

        return cls.from_dict(config.get("training", {}))

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def get_lora_config(self) -> Optional[LoRAConfig]:
        """
        Get LoRA configuration if enabled.

        Returns:
            LoRAConfig if use_lora is True, else None
        """
        if not self.use_lora:
            return None

        return LoRAConfig(
            base_model=self.lora_base_model,
            target_modules=self.lora_target_modules,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
            checkpoint=self.lora_checkpoint,
        )
