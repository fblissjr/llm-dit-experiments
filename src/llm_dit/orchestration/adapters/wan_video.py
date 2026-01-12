"""
Wan Video Pipeline Adapter for orchestration.

Last Updated: 2026-01-12

Wraps WanVideoPipeline as a PipelineStep for orchestration.
Supports T2V, TA, and TIA modes.

Example:
    pool = ModelPool()
    pool.register("humo-transformer", ModelSpec(path="~/Storage/HuMo/HuMo-17B"))
    pool.register("wan-vae", ModelSpec(path="~/Storage/Wan2.1-T2V-1.3B/"))
    pool.register("umt5-xxl", ModelSpec(path="~/Storage/Wan2.1-T2V-1.3B/"))

    orchestrator = Orchestrator(pool)
    orchestrator.add_step(WanVideoAdapter())

    result = orchestrator.run({
        "prompt": "A woman dancing",
        "audio": "music.wav",
        "num_frames": 97,
    })
"""

from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from ..outputs import VideoOutput as OrchVideoOutput
from ..steps import PipelineStep, StepInput, StepOutput


class WanVideoAdapter(PipelineStep):
    """
    Adapter for WanVideoPipeline as an orchestration step.

    Modes:
    - T2V: Text-to-video (no audio, no image)
    - TA: Text-audio (audio provided)
    - TIA: Text-image-audio (audio + reference image)

    Inputs:
        prompt: str - Text prompt describing desired video
        audio: Optional[str] - Audio file path for conditioning
        reference_image: Optional[Image] - Reference image for TIA mode
        num_frames: int - Number of frames (default 97)
        height: int - Video height (default 720)
        width: int - Video width (default 1280)
        guidance_scale: float - Text guidance (default 5.0)
        audio_scale: float - Audio guidance (default 1.0 if audio provided)
        seed: Optional[int] - Random seed

    Outputs:
        video: VideoOutput - Generated video frames

    Models (self-managed via WanVideoPipeline):
        - HuMo-17B or HuMo-1.7B transformer
        - Wan 2.1 VAE for decoding
        - UMT5-XXL text encoder
    """

    name = "WanVideoAdapter"
    description = "Generate video from text/audio using HuMo transformer"

    inputs = [
        StepInput("prompt", str, description="Text prompt for video"),
        StepInput("negative_prompt", str, default="", description="Negative prompt"),
        StepInput(
            "audio",
            (str, np.ndarray, type(None)),
            required=False,
            description="Audio file path or waveform",
        ),
        StepInput(
            "reference_image",
            (Image.Image, type(None)),
            required=False,
            description="Reference image for I2V",
        ),
        StepInput("num_frames", int, default=97, description="Number of frames"),
        StepInput("height", int, default=720, description="Video height"),
        StepInput("width", int, default=1280, description="Video width"),
        StepInput("num_inference_steps", int, default=50, description="Diffusion steps"),
        StepInput("guidance_scale", float, default=5.0, description="Text guidance"),
        StepInput("audio_scale", float, default=0.0, description="Audio guidance"),
        StepInput("seed", int, required=False, description="Random seed"),
    ]

    outputs = [
        StepOutput("video", OrchVideoOutput, description="Generated video"),
    ]

    # Note: This adapter manages its own models via WanVideoPipeline.from_pretrained()
    # It doesn't use models from the ModelPool. Set to empty to avoid loading errors.
    # Future: Could be refactored to accept models from pool for sharing across adapters.
    required_models = []

    def __init__(
        self,
        humo_path: Optional[str] = None,
        wan_path: Optional[str] = None,
        humo_variant: str = "17B",
        enable_cpu_offload: bool = True,
        **config,
    ):
        """
        Initialize adapter.

        Args:
            humo_path: Override path for HuMo base directory
            wan_path: Override path for Wan weights
            humo_variant: HuMo model variant ("17B" or "1.7B")
            enable_cpu_offload: Whether to offload models to CPU when not in use
            **config: Additional pipeline config
        """
        super().__init__(**config)
        self._humo_path = humo_path
        self._wan_path = wan_path
        self._humo_variant = humo_variant
        self._enable_cpu_offload = enable_cpu_offload
        self._pipeline = None

    def execute(
        self,
        inputs: Dict[str, Any],
        models: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute video generation.

        Args:
            inputs: Validated inputs (prompt, audio, etc.)
            models: Model instances from pool

        Returns:
            Dict with "video" key containing VideoOutput
        """
        # Get pipeline (creates via from_pretrained on first call)
        pipeline = self._get_pipeline()

        # Prepare inputs for pipeline
        audio = inputs.get("audio")
        reference_image = inputs.get("reference_image")

        # Auto-set audio scale if audio provided
        audio_scale = inputs.get("audio_scale", 0.0)
        if audio is not None and audio_scale == 0.0:
            audio_scale = 1.0

        # Generate
        result = pipeline(
            prompt=inputs["prompt"],
            negative_prompt=inputs.get("negative_prompt", ""),
            audio=audio,
            image=reference_image,
            height=inputs.get("height", 720),
            width=inputs.get("width", 1280),
            num_frames=inputs.get("num_frames", 97),
            num_inference_steps=inputs.get("num_inference_steps", 50),
            guidance_scale=inputs.get("guidance_scale", 5.0),
            audio_scale=audio_scale,
            seed=inputs.get("seed"),
        )

        # Convert to orchestration VideoOutput
        video_output = OrchVideoOutput(
            frames=result.frames,
            fps=result.fps,
            audio=result.audio,
            audio_sr=result.audio_sample_rate if hasattr(result, "audio_sample_rate") else 16000,
            seed=inputs.get("seed"),
        )

        return {"video": video_output}

    def _get_pipeline(self):
        """Get or create the underlying pipeline."""
        if self._pipeline is None:
            from llm_dit.pipelines.wan_video import WanVideoPipeline

            humo_path = self._humo_path or self.config.get(
                "humo_path", "~/Storage/HuMo"
            )
            wan_path = self._wan_path or self.config.get(
                "wan_path", "~/Storage/Wan2.1-T2V-1.3B"
            )
            humo_variant = self._humo_variant or self.config.get(
                "humo_variant", "17B"
            )

            self._pipeline = WanVideoPipeline.from_pretrained(
                humo_path=humo_path,
                wan_path=wan_path,
                humo_variant=humo_variant,
                enable_cpu_offload=self._enable_cpu_offload,
            )

        return self._pipeline


class WanTextEncoderStep(PipelineStep):
    """
    Low-level step for just the Wan text encoder.

    Use this for fine-grained control over text encoding.
    """

    name = "WanTextEncoderStep"
    inputs = [StepInput("prompt", str)]
    outputs = [StepOutput("text_embeddings", object)]  # TextEmbeddings
    required_models = ["umt5-xxl"]

    def execute(self, inputs: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
        encoder = models["umt5-xxl"]
        embeddings = encoder.encode(inputs["prompt"])
        return {"text_embeddings": embeddings}


class WanVAEDecodeStep(PipelineStep):
    """
    Low-level step for just the Wan VAE decoder.

    Use this for fine-grained control over latent decoding.
    """

    name = "WanVAEDecodeStep"
    inputs = [StepInput("latents", object)]  # torch.Tensor
    outputs = [StepOutput("video_frames", object)]  # np.ndarray
    required_models = ["wan-vae"]

    def execute(self, inputs: Dict[str, Any], models: Dict[str, Any]) -> Dict[str, Any]:
        vae = models["wan-vae"]
        frames = vae.decode(inputs["latents"])
        return {"video_frames": frames}
