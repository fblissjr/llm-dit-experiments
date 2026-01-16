#!/usr/bin/env python3
"""
LTX-2 Token Layer Router Training Scaffold

Last Updated: 2026-01-16

This script trains the TokenLayerRouter to learn per-token layer selection
for optimal text-image alignment. The router learns which Gemma layers are
most relevant for each type of token (style words, objects, actions, etc.).

Training Strategy:
    1. Freeze Gemma text encoder + LTX-2 DiT + VAE
    2. Only train router (~250K params)
    3. Optimize for SigLIP score on generated images
    4. Optional: Add sparsity loss to reduce compute

Loss Function:
    L = -SigLIP(prompt, image) + λ * SparsityLoss(router_weights)

The SigLIP loss is non-differentiable through the DiT, so we use:
    - REINFORCE (policy gradient) with SigLIP as reward
    - Or: Pre-compute layer contributions, train router as classifier

Usage:
    # Quick test (small dataset, few steps)
    uv run python experiments/ltx2/train_router.py --quick

    # Full training
    uv run python experiments/ltx2/train_router.py \\
        --prompts experiments/ltx2/prompts.py \\
        --epochs 10 \\
        --batch-size 4

    # Resume from checkpoint
    uv run python experiments/ltx2/train_router.py --resume checkpoints/router_epoch5.pt
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants for Gemma-2 9B (used in LTX-2)
GEMMA_HIDDEN_DIM = 3840
GEMMA_NUM_LAYERS = 49


def load_router():
    """Load the TokenLayerRouter."""
    from llm_dit.router import TokenLayerRouter
    return TokenLayerRouter(
        hidden_dim=GEMMA_HIDDEN_DIM,
        num_layers=GEMMA_NUM_LAYERS,
        bottleneck_dim=64,
        temperature=1.0,
        routing_mode="soft",
        init_uniform=True,  # Start with uniform routing
    )


def load_text_encoder(model_path: str = "models/LTX-2"):
    """Load Gemma text encoder from LTX-2 pipeline."""
    from transformers import AutoTokenizer, AutoModel

    # LTX-2 uses Gemma-2 9B
    logger.info("Loading Gemma text encoder...")

    # The text encoder is inside the LTX-2 pipeline
    # For training the router, we need direct access to hidden states
    from diffusers import LTX2Pipeline
    pipe = LTX2Pipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )

    # Extract components
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # Freeze text encoder
    for param in text_encoder.parameters():
        param.requires_grad = False

    return tokenizer, text_encoder


def get_token_embeddings(
    tokenizer,
    text_encoder,
    prompts: list[str],
    device: str = "cuda",
) -> torch.Tensor:
    """Get per-layer hidden states from Gemma for training.

    Returns:
        hidden_states: [B, T, D, L] tensor of all layer outputs
    """
    # Tokenize
    inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=1024,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    # Get hidden states from all layers
    with torch.no_grad():
        outputs = text_encoder(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
        )

    # Stack hidden states: list of [B, T, D] → [B, T, D, L]
    hidden_states = torch.stack(outputs.hidden_states, dim=-1)

    return hidden_states, inputs.attention_mask


class RouterTrainer:
    """Training loop for TokenLayerRouter.

    This is a scaffold - the actual reward computation requires generation
    which is slow. For prototyping, you can:

    1. Pre-compute: Run layer_blend_sweep first, cache (token_embeds, siglip_scores)
       pairs for different layer configs, then train router as a regressor.

    2. Online: Generate images during training (very slow, but most accurate).

    3. Proxy reward: Use a faster proxy metric during training, validate with SigLIP.
    """

    def __init__(
        self,
        router: nn.Module,
        tokenizer,
        text_encoder,
        learning_rate: float = 1e-4,
        sparsity_weight: float = 0.01,
        device: str = "cuda",
    ):
        self.router = router.to(device)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder.to(device)
        self.device = device

        self.optimizer = AdamW(router.parameters(), lr=learning_rate)
        self.sparsity_weight = sparsity_weight

        # Optional sparsity loss
        from llm_dit.router.token_layer_router import SparsityLoss
        self.sparsity_loss = SparsityLoss(target_sparsity=8.0, loss_weight=sparsity_weight)

    def compute_reward(
        self,
        prompts: list[str],
        layer_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reward for router outputs.

        This is the key function to implement. Options:

        1. Full generation + SigLIP (slow but accurate):
           - Generate video with weighted layers
           - Score with SigLIP
           - Return as reward

        2. Proxy reward (fast but approximate):
           - Use pre-computed layer contribution scores
           - Reward = sum(weights * layer_contribution_scores)

        3. Learned reward model:
           - Train a small model to predict SigLIP from layer weights
           - Use as fast proxy during training

        Args:
            prompts: List of text prompts
            layer_weights: Router output [B, T, L]

        Returns:
            rewards: [B] tensor of rewards (higher is better)
        """
        # TODO: Implement reward computation
        # For now, return dummy rewards for scaffolding
        batch_size = len(prompts)
        return torch.zeros(batch_size, device=self.device)

    def train_step(
        self,
        prompts: list[str],
    ) -> dict:
        """Single training step.

        Uses REINFORCE to optimize router for non-differentiable reward.
        """
        self.router.train()
        self.optimizer.zero_grad()

        # Get token embeddings from Gemma
        hidden_states, attention_mask = get_token_embeddings(
            self.tokenizer,
            self.text_encoder,
            prompts,
            self.device,
        )

        # Use last layer hidden state as input to router
        # (could also use pooled or average across layers)
        router_input = hidden_states[:, :, :, -1]  # [B, T, D]

        # Get layer weights from router
        layer_weights = self.router(router_input, attention_mask)  # [B, T, L]

        # Compute reward
        rewards = self.compute_reward(prompts, layer_weights)

        # REINFORCE loss: -reward * log_prob
        # Since softmax outputs are probabilities, we can use entropy
        log_probs = (layer_weights + 1e-8).log()
        policy_loss = -(rewards.unsqueeze(-1).unsqueeze(-1) * log_probs).mean()

        # Sparsity loss
        sparsity_loss = self.sparsity_loss(layer_weights)

        # Total loss
        total_loss = policy_loss + sparsity_loss

        # Backward
        total_loss.backward()
        self.optimizer.step()

        # Stats
        routing_stats = self.router.get_routing_stats(layer_weights.detach())

        return {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "mean_reward": rewards.mean().item(),
            "routing_entropy": routing_stats["entropy"],
            "routing_sparsity": routing_stats["sparsity"],
        }

    def save_checkpoint(self, path: Path, epoch: int, stats: dict):
        """Save training checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "router_state_dict": self.router.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "stats": stats,
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path) -> int:
        """Load training checkpoint. Returns epoch number."""
        checkpoint = torch.load(path, map_location=self.device)
        self.router.load_state_dict(checkpoint["router_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return checkpoint["epoch"]


def load_prompts(quick: bool = False) -> list[str]:
    """Load training prompts."""
    from experiments.ltx2.prompts import get_all_prompts

    prompts_dict = get_all_prompts(quick=quick)
    return list(prompts_dict.values())


def main():
    parser = argparse.ArgumentParser(
        description="Train TokenLayerRouter for LTX-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", default="models/LTX-2", help="Path to LTX-2 model")
    parser.add_argument("--output-dir", default="experiments/results/router_training", help="Output directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--sparsity-weight", type=float, default=0.01, help="Sparsity loss weight")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--device", default="cuda", help="Device")

    args = parser.parse_args()

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LTX-2 Token Layer Router Training")
    logger.info("=" * 60)
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {args.device}")

    # Load components
    router = load_router()
    logger.info(f"Router parameters: {sum(p.numel() for p in router.parameters()):,}")

    tokenizer, text_encoder = load_text_encoder(args.model_path)

    # Create trainer
    trainer = RouterTrainer(
        router=router,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        learning_rate=args.lr,
        sparsity_weight=args.sparsity_weight,
        device=args.device,
    )

    # Resume if specified
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(Path(args.resume))

    # Load prompts
    prompts = load_prompts(quick=args.quick)
    logger.info(f"Training prompts: {len(prompts)}")

    # Training loop
    all_stats = []

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Simple batching
        epoch_stats = []
        for i in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[i:i + args.batch_size]
            stats = trainer.train_step(batch_prompts)
            epoch_stats.append(stats)

            logger.info(
                f"  Step {i // args.batch_size + 1}: "
                f"loss={stats['loss']:.4f}, "
                f"reward={stats['mean_reward']:.4f}, "
                f"entropy={stats['routing_entropy']:.2f}"
            )

        # Epoch summary
        avg_stats = {
            k: sum(s[k] for s in epoch_stats) / len(epoch_stats)
            for k in epoch_stats[0].keys()
        }
        all_stats.append({"epoch": epoch + 1, **avg_stats})

        logger.info(f"Epoch {epoch + 1} avg: loss={avg_stats['loss']:.4f}")

        # Save checkpoint
        checkpoint_path = output_dir / f"router_epoch{epoch + 1}.pt"
        trainer.save_checkpoint(checkpoint_path, epoch + 1, avg_stats)

    # Save final stats
    stats_path = output_dir / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    logger.info(f"\nTraining complete. Results in {output_dir}")
    logger.info("=" * 60)

    # Print routing analysis
    logger.info("\nFinal Routing Analysis:")
    with torch.no_grad():
        test_prompts = prompts[:2]
        hidden_states, attention_mask = get_token_embeddings(
            tokenizer, text_encoder, test_prompts, args.device
        )
        router_input = hidden_states[:, :, :, -1]
        weights = router.to(args.device)(router_input, attention_mask)
        stats = router.get_routing_stats(weights)

    logger.info(f"  Routing entropy: {stats['entropy']:.2f}")
    logger.info(f"  Effective layers: {stats['sparsity']:.1f}")
    logger.info(f"  Top layer usage: {stats['top_layer_distribution'][:5]}")


if __name__ == "__main__":
    main()
