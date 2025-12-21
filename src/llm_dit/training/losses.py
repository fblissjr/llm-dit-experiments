"""
Training loss functions for flow matching models.

Based on DiffSynth-Studio implementation (Apache 2.0 license).

Usage:
    from llm_dit.training.losses import FlowMatchSFTLoss

    loss = FlowMatchSFTLoss(
        pipe,
        input_latents=target_latents,
        prompt_embeds=prompt_embeds,
    )
"""

import torch
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from llm_dit.pipelines import ZImagePipeline


def FlowMatchSFTLoss(
    pipe: "ZImagePipeline",
    input_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    min_timestep_boundary: float = 0.0,
    max_timestep_boundary: float = 1.0,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Flow matching supervised fine-tuning loss.

    Samples a random timestep, adds noise to latents, predicts the velocity,
    and computes MSE loss with timestep weighting.

    The loss is:
        L = w(t) * MSE(v_pred, v_target)

    where:
        - v_pred is the model's velocity prediction
        - v_target = noise - sample (flow matching velocity)
        - w(t) is the timestep weight (Gaussian-weighted)

    Args:
        pipe: ZImagePipeline instance with scheduler and transformer
        input_latents: Target latents (encoded from training images)
        prompt_embeds: Text embeddings from encoder
        min_timestep_boundary: Minimum timestep fraction (0.0-1.0)
        max_timestep_boundary: Maximum timestep fraction (0.0-1.0)
        use_gradient_checkpointing: Enable gradient checkpointing
        use_gradient_checkpointing_offload: Offload checkpoints to CPU
        **kwargs: Additional arguments passed to transformer

    Returns:
        Weighted MSE loss tensor

    Example:
        >>> # Encode training image
        >>> latents = pipe.vae.encode(image).latent_dist.sample()
        >>> latents = latents * pipe.vae.config.scaling_factor
        >>>
        >>> # Encode prompt
        >>> prompt_embeds = pipe.encode_prompt("A cat sleeping")
        >>>
        >>> # Compute loss
        >>> loss = FlowMatchSFTLoss(pipe, latents, prompt_embeds)
        >>> loss.backward()
    """
    from llm_dit.training.gradient_checkpoint import gradient_checkpoint_forward

    scheduler = pipe.scheduler
    device = input_latents.device
    dtype = input_latents.dtype

    # Sample random timestep index
    num_timesteps = len(scheduler.timesteps)
    min_idx = int(min_timestep_boundary * num_timesteps)
    max_idx = int(max_timestep_boundary * num_timesteps)

    # Ensure valid range
    min_idx = max(0, min_idx)
    max_idx = min(num_timesteps, max_idx)
    if min_idx >= max_idx:
        min_idx = 0
        max_idx = num_timesteps

    timestep_idx = torch.randint(min_idx, max_idx, (1,), device=device)
    timestep = scheduler.timesteps[timestep_idx].to(dtype=dtype, device=device)

    # Add noise to latents
    noise = torch.randn_like(input_latents)
    noisy_latents = scheduler.add_noise(input_latents, noise, timestep)

    # Get training target (velocity)
    training_target = scheduler.training_target(input_latents, noise, timestep)

    # Forward pass through transformer
    if use_gradient_checkpointing or use_gradient_checkpointing_offload:
        model_output = gradient_checkpoint_forward(
            pipe.transformer,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            hidden_states=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            **kwargs,
        )
    else:
        model_output = pipe.transformer(
            hidden_states=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            **kwargs,
        )

    # Handle output format (may be tuple or object with .sample)
    if hasattr(model_output, 'sample'):
        model_output = model_output.sample
    elif isinstance(model_output, tuple):
        model_output = model_output[0]

    # Compute MSE loss
    loss = torch.nn.functional.mse_loss(
        model_output.float(),
        training_target.float(),
    )

    # Apply timestep weighting
    weight = scheduler.training_weight(timestep)
    loss = loss * weight

    return loss


def DirectDistillLoss(
    pipe: "ZImagePipeline",
    input_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    num_inference_steps: int = 8,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Direct distillation loss for few-step generation.

    Runs the full denoising process and compares final output to target.
    Used for distilling teacher models into faster student models.

    Args:
        pipe: ZImagePipeline instance
        input_latents: Target latents
        prompt_embeds: Text embeddings
        num_inference_steps: Number of denoising steps
        **kwargs: Additional arguments

    Returns:
        MSE loss between generated and target latents
    """
    scheduler = pipe.scheduler
    device = input_latents.device
    dtype = input_latents.dtype

    # Set up scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Start from pure noise
    latents = torch.randn_like(input_latents)

    # Denoise
    for timestep in scheduler.timesteps:
        timestep = timestep.unsqueeze(0).to(dtype=dtype, device=device)

        model_output = pipe.transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            **kwargs,
        )

        if hasattr(model_output, 'sample'):
            model_output = model_output.sample
        elif isinstance(model_output, tuple):
            model_output = model_output[0]

        latents = scheduler.step(model_output, timestep, latents).prev_sample

    # Compare final latents to target
    loss = torch.nn.functional.mse_loss(
        latents.float(),
        input_latents.float(),
    )

    return loss


def ConsistencyLoss(
    pipe: "ZImagePipeline",
    input_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    timestep_pairs: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Consistency loss for training consistency models.

    Enforces that the model output at adjacent timesteps maps to
    the same clean sample.

    Args:
        pipe: ZImagePipeline instance
        input_latents: Target latents
        prompt_embeds: Text embeddings
        timestep_pairs: Optional pairs of timesteps to compare
        **kwargs: Additional arguments

    Returns:
        Consistency loss
    """
    scheduler = pipe.scheduler
    device = input_latents.device
    dtype = input_latents.dtype

    if timestep_pairs is None:
        # Generate random adjacent timestep pair
        num_timesteps = len(scheduler.timesteps)
        idx1 = torch.randint(0, num_timesteps - 1, (1,), device=device)
        idx2 = idx1 + 1
        t1 = scheduler.timesteps[idx1].to(dtype=dtype, device=device)
        t2 = scheduler.timesteps[idx2].to(dtype=dtype, device=device)
    else:
        t1, t2 = timestep_pairs[0], timestep_pairs[1]

    # Add noise at both timesteps
    noise = torch.randn_like(input_latents)
    noisy1 = scheduler.add_noise(input_latents, noise, t1)
    noisy2 = scheduler.add_noise(input_latents, noise, t2)

    # Get predictions at both timesteps
    out1 = pipe.transformer(
        hidden_states=noisy1,
        timestep=t1,
        encoder_hidden_states=prompt_embeds,
        **kwargs,
    )
    if hasattr(out1, 'sample'):
        out1 = out1.sample

    out2 = pipe.transformer(
        hidden_states=noisy2,
        timestep=t2,
        encoder_hidden_states=prompt_embeds,
        **kwargs,
    )
    if hasattr(out2, 'sample'):
        out2 = out2.sample

    # Consistency: outputs should predict same target
    loss = torch.nn.functional.mse_loss(out1.float(), out2.float())

    return loss
