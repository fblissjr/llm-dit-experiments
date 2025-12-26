"""
Z-Image training module.

Provides training functionality for Z-Image models including:
- Full DiT training
- LoRA fine-tuning
- Context refiner training

Usage:
    from llm_dit.training import ZImageTrainingModule, TrainingConfig

    config = TrainingConfig(
        model_path="/path/to/z-image",
        use_lora=True,
        lora_rank=32,
    )

    module = ZImageTrainingModule(config)
    loss = module(data)
    loss.backward()
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

from llm_dit.training.base import BaseTrainingModule
from llm_dit.training.config import LoRAConfig, TrainingConfig
from llm_dit.training.losses import FlowMatchSFTLoss, DirectDistillLoss

logger = logging.getLogger(__name__)


class ZImageTrainingModule(BaseTrainingModule):
    """
    Training module for Z-Image models.

    Supports:
        - Full DiT training
        - LoRA fine-tuning
        - Context refiner training
        - Text encoder fine-tuning (optional)

    Training Tasks:
        - sft: Supervised fine-tuning with flow matching loss
        - distill: Direct distillation from teacher model

    Attributes:
        pipe: ZImagePipeline instance
        task: Training task type
        use_gradient_checkpointing: Whether to use gradient checkpointing

    Example:
        >>> from llm_dit.training import ZImageTrainingModule, TrainingConfig
        >>>
        >>> config = TrainingConfig(
        ...     model_path="/path/to/z-image",
        ...     use_lora=True,
        ...     lora_rank=32,
        ...     lora_target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        ... )
        >>>
        >>> module = ZImageTrainingModule(config)
        >>> print(f"Trainable params: {module.num_trainable_params():,}")
        >>>
        >>> # Training loop
        >>> for data in dataloader:
        ...     loss = module(data)
        ...     loss.backward()
        ...     optimizer.step()
        ...     optimizer.zero_grad()
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        model_path: Optional[str] = None,
        trainable_models: Optional[List[str]] = None,
        lora_config: Optional[LoRAConfig] = None,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        task: str = "sft",
    ):
        """
        Initialize the Z-Image training module.

        Args:
            config: Training configuration (overrides other args if provided)
            model_path: Path to Z-Image model weights
            trainable_models: List of model names to train (e.g., ["transformer"])
            lora_config: LoRA configuration (if using LoRA)
            use_gradient_checkpointing: Enable gradient checkpointing
            use_gradient_checkpointing_offload: Offload checkpoints to CPU
            device: Training device
            dtype: Training dtype
            task: Training task type
        """
        # Use config values if provided
        if config is not None:
            model_path = model_path or config.model_path
            trainable_models = trainable_models or config.trainable_models
            lora_config = lora_config or config.get_lora_config()
            use_gradient_checkpointing = config.use_gradient_checkpointing
            use_gradient_checkpointing_offload = config.use_gradient_checkpointing_offload
            task = config.task

        super().__init__(config=config, lora_config=lora_config)

        if not model_path:
            raise ValueError("model_path is required")

        self.task = task
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.device = device
        self.dtype = dtype

        # Load pipeline
        self._load_pipeline(model_path, device, dtype)

        # Configure training mode
        self._setup_training_mode(trainable_models, lora_config)

        logger.info(
            f"ZImageTrainingModule initialized: "
            f"task={task}, "
            f"trainable_params={self.num_trainable_params():,}, "
            f"gradient_checkpointing={use_gradient_checkpointing}"
        )

    def _load_pipeline(
        self,
        model_path: str,
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ) -> None:
        """Load the Z-Image pipeline."""
        from llm_dit.pipelines import ZImagePipeline

        logger.info(f"Loading Z-Image pipeline from: {model_path}")

        self.pipe = ZImagePipeline.from_pretrained(
            model_path,
            torch_dtype=dtype,
            use_custom_scheduler=True,
        )

        # Move to device
        self.pipe.to(device)

        # Set training timesteps for scheduler
        self.pipe.scheduler.set_timesteps(
            self.pipe.scheduler.num_train_timesteps,
            device=device,
        )

    def _setup_training_mode(
        self,
        trainable_models: Optional[List[str]],
        lora_config: Optional[LoRAConfig],
    ) -> None:
        """Configure which models to train and add LoRA if needed."""
        # Start by freezing everything
        self.freeze_all()

        # Determine what to train
        if lora_config is not None:
            # LoRA training
            base_model = getattr(self.pipe, lora_config.base_model, None)
            if base_model is None:
                raise ValueError(
                    f"Model '{lora_config.base_model}' not found in pipeline"
                )

            # Add LoRA adapters
            base_model = self.add_lora_to_model(
                base_model,
                target_modules=lora_config.target_modules,
                lora_rank=lora_config.rank,
                lora_alpha=lora_config.alpha,
                lora_dropout=lora_config.dropout,
            )
            setattr(self.pipe, lora_config.base_model, base_model)

            # Load checkpoint if provided
            if lora_config.checkpoint:
                self.load_lora_checkpoint(base_model, lora_config.checkpoint)

            self.trainable_model_names.add(lora_config.base_model)

        elif trainable_models:
            # Full model training
            self.freeze_except(trainable_models)

        else:
            # Default: train transformer only
            logger.warning(
                "No trainable_models specified and no LoRA config. "
                "Defaulting to training transformer."
            )
            self.freeze_except(["transformer"])

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encode image to latents for training target.

        Args:
            image: PIL Image to encode

        Returns:
            Latent tensor (scaled by VAE factor)
        """
        with torch.no_grad():
            # Preprocess image
            pixel_values = self.pipe.image_processor.preprocess(image)
            pixel_values = pixel_values.to(
                device=self.device,
                dtype=self.dtype,
            )

            # Encode to latents
            latents = self.pipe.vae.encode(pixel_values).latent_dist.sample()
            latents = (latents - self.pipe.vae.config.shift_factor) * self.pipe.vae.config.scaling_factor

        return latents

    def encode_prompt(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        thinking_content: Optional[str] = None,
        template: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Encode prompt to embeddings.

        Args:
            prompt: Text prompt
            system_prompt: Optional system prompt
            thinking_content: Optional thinking content
            template: Optional template name

        Returns:
            Prompt embeddings tensor
        """
        with torch.no_grad():
            prompt_embeds = self.pipe.encode_prompt(
                prompt=prompt,
                system_prompt=system_prompt,
                thinking_content=thinking_content,
                template=template,
            )

        return prompt_embeds

    def get_pipeline_inputs(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert dataset item to pipeline inputs.

        Expected data keys:
            - prompt: Text prompt (required)
            - image: PIL Image (required)
            - system_prompt: Optional system prompt
            - thinking_content: Optional thinking content
            - template: Optional template name
            - width: Image width (optional, derived from image)
            - height: Image height (optional, derived from image)

        Args:
            data: Dataset item dictionary

        Returns:
            Dictionary with:
                - input_latents: Encoded image latents
                - prompt_embeds: Text embeddings
                - use_gradient_checkpointing: Checkpointing flag
                - use_gradient_checkpointing_offload: Offload flag
        """
        # Encode image to latents
        image = data["image"]
        input_latents = self.encode_image(image)

        # Encode prompt
        prompt_embeds = self.encode_prompt(
            prompt=data["prompt"],
            system_prompt=data.get("system_prompt"),
            thinking_content=data.get("thinking_content"),
            template=data.get("template"),
        )

        return {
            "input_latents": input_latents,
            "prompt_embeds": prompt_embeds,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }

    def forward(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        Forward pass computing training loss.

        Args:
            data: Dataset item with 'prompt' and 'image' keys

        Returns:
            Loss tensor
        """
        # Get pipeline inputs
        inputs = self.get_pipeline_inputs(data)

        # Compute loss based on task
        if self.task == "sft":
            loss = FlowMatchSFTLoss(
                self.pipe,
                input_latents=inputs["input_latents"],
                prompt_embeds=inputs["prompt_embeds"],
                use_gradient_checkpointing=inputs["use_gradient_checkpointing"],
                use_gradient_checkpointing_offload=inputs["use_gradient_checkpointing_offload"],
            )
        elif self.task == "distill":
            loss = DirectDistillLoss(
                self.pipe,
                input_latents=inputs["input_latents"],
                prompt_embeds=inputs["prompt_embeds"],
            )
        else:
            raise ValueError(f"Unknown task: {self.task}")

        return loss

    @classmethod
    def from_config(
        cls,
        config: TrainingConfig,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "ZImageTrainingModule":
        """
        Create module from configuration.

        Args:
            config: Training configuration
            device: Training device
            dtype: Training dtype

        Returns:
            Initialized training module
        """
        return cls(
            config=config,
            device=device,
            dtype=dtype,
        )
