"""
llm-dit backend implementation.
"""

import gc
import logging
import time
from pathlib import Path
from typing import Optional

import torch

from .protocol import Backend, GenerationConfig, GenerationResult, GenerationStats

logger = logging.getLogger(__name__)

def _cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class LLMDitBackend(Backend):
    def __init__(self, model_path: str = "models/LTX-2"):
        self.model_path = Path(model_path)

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
        from llm_dit.pipelines.generate import (
            GenerationConfig as LLMDitConfig,
            generate_video_with_offloading,
        )

        config.validate()

        # Stats tracking
        stats = GenerationStats()
        start_time = time.time()
        stage_times = {}
        stage_start = 0

        def progress_callback(stage: str, step: int, total: int):
            nonlocal stage_start
            if step == 0:
                stage_start = time.time()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            elif step == total:
                stage_times[stage] = time.time() - stage_start
                if torch.cuda.is_available():
                    mem = torch.cuda.max_memory_allocated() / 1024**3
                    if stage == "text_encoder":
                        stats.text_encoder_peak_memory = mem
                    elif stage == "transformer":
                        stats.transformer_peak_memory = mem
                    elif stage == "vae":
                        stats.vae_peak_memory = mem

        # Convert config
        llm_dit_config = LLMDitConfig(
            num_frames=config.num_frames,
            height=config.height,
            width=config.width,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            seed=config.seed,
        )

        video = generate_video_with_offloading(
            prompt=prompt,
            config=llm_dit_config,
            model_path=self.model_path,
            quantize=config.fp8,
            precision="fp8-native" if config.fp8 else "bf16",
            dtype=config.dtype,
            callback=progress_callback,
        )

        stats.total_time = time.time() - start_time
        stats.text_encoder_time = stage_times.get("text_encoder", 0)
        stats.transformer_time = stage_times.get("transformer", 0)
        stats.vae_time = stage_times.get("vae", 0)

        # Populate result
        stats.actual_num_frames = video.shape[0]
        stats.actual_height = video.shape[1]
        stats.actual_width = video.shape[2]

        result = GenerationResult(
            video=video, prompt=prompt, config=config, stats=stats, backend_name=self.name
        )

        if output_dir:
            output_dir = Path(output_dir)
            if save_video:
                result.save_video(output_dir / "video.mp4")
            result.save_metadata(output_dir / "metadata.json")

        return result

    def encode_text(
        self,
        prompt: str,
        output_dir: Optional[Path] = None,
        debug_trace: bool = False,
    ) -> torch.Tensor:
        """
        Encode text prompt to embeddings.

        Args:
            prompt: Text prompt to encode
            output_dir: Optional directory to save diagnostics
            debug_trace: If True, save detailed connector diagnostics

        Returns:
            Text embeddings tensor [1, seq_len, dim]
        """
        from llm_dit.encoders import Gemma3Encoder

        encoder = Gemma3Encoder(
            model_id=str(self.model_path / "text_encoder"),
            load_in_8bit=True,
            device="cuda",
        )

        # Force model loading to ensure connector is available for hooks
        # (Gemma3Encoder uses lazy loading)
        encoder._load_model()

        # Setup diagnostics collector if debug_trace enabled
        diagnostics_collector = None
        if debug_trace:
            from .diagnostics import ConnectorDiagnosticsCollector

            diagnostics_collector = ConnectorDiagnosticsCollector()
            # Attach hooks to the connector inside the encoder
            if encoder._embeddings_connector is not None:
                diagnostics_collector.attach_hooks(encoder._embeddings_connector)
                logger.debug(
                    f"Attached diagnostics hooks to connector with "
                    f"{len(encoder._embeddings_connector.transformer_blocks)} blocks"
                )
            else:
                logger.warning("Connector not available for diagnostics hooks")

        # Run encoding
        out = encoder.encode(prompt)

        # Collect and save diagnostics
        if diagnostics_collector is not None:
            diagnostics = diagnostics_collector.collect()

            # Print summary
            logger.info(diagnostics.summary())

            # Check for anomalies
            warnings = diagnostics.check_for_anomalies()
            for w in warnings:
                logger.warning(f"ANOMALY: {w}")

            # Save to file if output_dir provided
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                diagnostics.save(output_dir / "connector_diagnostics.json")

            # Cleanup hooks
            diagnostics_collector.remove_hooks()

        # [1, seq_len, dim]
        embeddings = out.embeddings[0].unsqueeze(0).cpu()

        del encoder
        _cleanup_memory()
        return embeddings

    def cleanup(self):
        _cleanup_memory()
